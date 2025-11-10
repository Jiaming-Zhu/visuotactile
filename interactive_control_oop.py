"""
一个将 PyBullet 仿真与真实机械臂控制解耦的 OOP 版本，并提供命令行交互层。

核心思想：
- SimulationInterface 仅负责仿真加载、IK、前向运动学读取与关节下发。
- RealRobotInterface 仅负责真实机械臂连接、读取角度、下发角度（度）。
- ArmController 统一调度两者：计算 IK、下发真实机械臂、基于真实角度做到位判定。
- run_cli 提供简单命令行交互，覆盖 connect/current/home/move 等常用命令。

到位判定逻辑：
- 若已连接真实机械臂：周期性读取真实关节角（度）→ 转弧度 → 写入仿真关节 → 用仿真 FK 得到末端实际位置 → 与目标做差。
- 未连接时：回退到仿真到位。
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import threading
import sys as _sys
import subprocess
import socket
import json
from threading import Event

import numpy as np
import pybullet as p
import pybullet_data
import torch

try:
    from lerobot.common.robot_devices.robots.configs import So101RobotConfig
    from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot

    LEROBOT_AVAILABLE = True
except ImportError:
    ManipulatorRobot = None  # type: ignore[assignment]
    So101RobotConfig = None  # type: ignore[assignment]
    LEROBOT_AVAILABLE = False


@dataclass
class ReachResult:
    reached: bool
    error: float
    final_position: Tuple[float, float, float]
    samples: List[Tuple[float, float, float]]


class SimulationInterface:
    """轻量的仿真封装：暴露 IK、FK（通过 get_end_effector_position）、以及关节控制。

    说明：
    - 仅收集前 6 个可旋转关节作为“控制关节”（大多数 6 DoF 机械臂）。
    - end_effector 通过 link 名包含 'gripper_frame' 来查找，找不到则取最后一个 link。
    """

    def __init__(
        self,
        gui: bool = True,
        urdf_filename: str = "so101_new_calib.urdf",
        base_position: Sequence[float] = (0.0, 0.0, 0.0),
        gravity: Sequence[float] = (0.0, 0.0, -10.0),
    ) -> None:
        self.client_id = self._connect(gui)
        self.urdf_path = str(Path(__file__).with_name(urdf_filename))
        self.base_position = base_position
        self.gravity = gravity

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(*self.gravity)

        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(
            self.urdf_path,
            basePosition=self.base_position,
            baseOrientation=p.getQuaternionFromEuler((0.0, 0.0, 0.0)),
            useFixedBase=True,
        )

        self.end_effector_index = self._find_end_effector()
        self.control_joint_indices = self._collect_control_joints()

    @staticmethod
    def _connect(gui: bool) -> int:
        """连接 PyBullet，优先 GUI，失败则回退 DIRECT。"""
        if gui:
            try:
                return p.connect(p.GUI)
            except Exception:
                pass
        return p.connect(p.DIRECT)

    def _find_end_effector(self) -> int:
        """查找末端执行器的 link 索引。"""
        end_effector = -1
        num_joints = p.getNumJoints(self.robot_id)
        for index in range(num_joints):
            info = p.getJointInfo(self.robot_id, index)
            link_name = info[12].decode("utf-8")
            if "gripper_frame" in link_name.lower():
                end_effector = index
                break
        return end_effector if end_effector >= 0 else num_joints - 1

    def _collect_control_joints(self) -> List[int]:
        """收集用于控制的关节索引（最多 6 个 REVOLUTE）。"""
        indices: List[int] = []
        for index in range(p.getNumJoints(self.robot_id)):
            joint_type = p.getJointInfo(self.robot_id, index)[2]
            if joint_type == p.JOINT_REVOLUTE:
                indices.append(index)
            if len(indices) == 6:
                break
        return indices

    def calculate_ik(self, target_position: Sequence[float]) -> List[float]:
        """计算 IK，返回所有关节的解（pybullet 的顺序与数量）。"""
        return list(
            p.calculateInverseKinematics(
                self.robot_id,
                self.end_effector_index,
                targetPosition=target_position,
                maxNumIterations=1000,
                residualThreshold=1e-4,
            )
        )

    def command_joint_positions(self, joint_positions: Sequence[float], force: float = 500.0) -> None:
        """按 pybullet 的关节索引写入目标，带健壮性回退。

        - 优先用 joint_idx 直接取目标；
        - 若 IK 输出长度小于最大索引，则退化为使用 control 顺序索引；
        - 仍不足则跳过该关节。
        """
        joint_array = list(joint_positions)
        for offset, joint_idx in enumerate(self.control_joint_indices):
            if joint_idx < len(joint_array):
                target = joint_array[joint_idx]
            elif offset < len(joint_array):
                target = joint_array[offset]
            else:
                # Not enough IK outputs; skip this joint safely.
                continue
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target,
                force=force,
                maxVelocity=5.0,
            )

    def reset_joint_states(self, joint_positions: Sequence[float]) -> None:
        """直接重置（不经电机控制）仿真中的控制关节角，用于计算 FK。"""
        for offset, joint_idx in enumerate(self.control_joint_indices):
            if offset < len(joint_positions):
                p.resetJointState(self.robot_id, joint_idx, joint_positions[offset])

    def get_end_effector_position(self) -> Tuple[float, float, float]:
        """读取末端位姿中的位置（x, y, z）。"""
        link_state = p.getLinkState(self.robot_id, self.end_effector_index)
        return tuple(link_state[0])

    def command_control_joint_positions(self, control_joint_positions: Sequence[float], force: float = 500.0) -> None:
        """仅按“控制关节顺序”（最多 6 个）下发目标角。"""
        for i, joint_idx in enumerate(self.control_joint_indices):
            target = control_joint_positions[i] if i < len(control_joint_positions) else 0.0
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=float(target),
                force=force,
                maxVelocity=5.0,
            )

    def step(self, steps: int = 1, sleep: float = 1.0 / 240.0) -> None:
        for _ in range(steps):
            p.stepSimulation()
            time.sleep(sleep)

    def shutdown(self) -> None:
        if p.isConnected(self.client_id):
            p.disconnect(self.client_id)


class RealRobotInterface:
    """封装与真实机械臂（LeRobot）的读写与连接管理。"""

    def __init__(self, calibration_dir: str, zero_offset_deg: Optional[Sequence[float]] = None) -> None:
        if not LEROBOT_AVAILABLE:
            raise RuntimeError("LeRobot is not available; cannot control the physical arm.")

        self.calibration_dir = calibration_dir
        self.zero_offset_deg = np.zeros(6) if zero_offset_deg is None else np.array(zero_offset_deg, dtype=float)
        self.robot: Optional[ManipulatorRobot] = None
        self._follower_name: Optional[str] = None
        self._io_lock = threading.Lock()

    @property
    def is_connected(self) -> bool:
        return self.robot is not None

    def connect(self) -> None:
        """建立与 follower 臂的连接并加载校准。"""
        config = So101RobotConfig(
            calibration_dir=self.calibration_dir,
            leader_arms={},
            cameras={},
        )
        with self._io_lock:
            self.robot = ManipulatorRobot(config)
            self.robot.connect()
            self._follower_name = next(iter(self.robot.follower_arms))
            # 启动即设为最大加速度/速度（满速），保证交互顺畅
            # try:
            #     follower = self.robot.follower_arms[self._follower_name]
            #     follower.write("Acceleration", 254)   # 建议最大值
            #     follower.write("Goal_Speed", 1023)    # 100% 速度幅值
            # except Exception:
            #     # 某些固件/接口不支持时忽略
            #     pass


    def disconnect(self, disable_torque: bool = True) -> None:
        """断开连接，可选禁用力矩（安全）。"""
        if not self.robot:
            return
        if disable_torque:
            for name in self.robot.follower_arms:
                try:
                    with self._io_lock:
                        self.robot.follower_arms[name].write("Torque_Enable", 0)
                except Exception:
                    pass
        with self._io_lock:
            self.robot.disconnect()
            self.robot = None
        self._follower_name = None

    def write_goal_positions(self, joint_angles_deg: Sequence[float]) -> None:
        """写入目标角（度）。长度根据电机数量自动补齐/截断，并加零点偏移。"""
        if not self.robot or not self._follower_name:
            raise RuntimeError("Physical arm is not connected.")
        desired = np.array(joint_angles_deg, dtype=float)
        follower = self.robot.follower_arms[self._follower_name]
        motor_count = len(follower.motor_names)
        if len(desired) < motor_count:
            desired = np.pad(desired, (0, motor_count - len(desired)), mode="constant")
        desired = desired[:motor_count]
        zero_offset = self.zero_offset_deg[:motor_count]
        if len(zero_offset) < motor_count:
            zero_offset = np.pad(zero_offset, (0, motor_count - len(zero_offset)), mode="constant")
        desired += zero_offset
        command = torch.from_numpy(desired.astype(np.float32))
        with self._io_lock:
            follower.write("Goal_Position", command.numpy())

    def read_joint_positions(self) -> np.ndarray:
        """读取当前角度（度），长度对齐零点偏移。"""
        if not self.robot or not self._follower_name:
            raise RuntimeError("Physical arm is not connected.")
        follower = self.robot.follower_arms[self._follower_name]
        with self._io_lock:
            present = follower.read("Present_Position")
        values = np.array(present, dtype=float)
        zero_offset = self.zero_offset_deg[: len(values)]
        if len(zero_offset) < len(values):
            zero_offset = np.pad(zero_offset, (0, len(values) - len(zero_offset)), mode="constant")
        return values[: len(zero_offset)] - zero_offset


class ArmController:
    """高层控制器：统一使用仿真与真实臂，提供 move_to/直接关节下发等能力。"""

    def __init__(
        self,
        use_gui: bool = True,
        calibration_dir: Optional[str] = None,
        zero_offset_deg: Optional[Sequence[float]] = None,
    ) -> None:
        self.sim = SimulationInterface(gui=use_gui)
        self.real: Optional[RealRobotInterface] = None
        self.last_commanded_control_angles_deg: Optional[List[float]] = None
        if calibration_dir and LEROBOT_AVAILABLE:
            self.real = RealRobotInterface(calibration_dir, zero_offset_deg)

    def connect_real_robot(self) -> None:
        if not self.real:
            raise RuntimeError("Real robot interface was not initialised with a calibration directory.")
        if not self.real.is_connected:
            self.real.connect()

    def disconnect_real_robot(self) -> None:
        if self.real and self.real.is_connected:
            self.real.disconnect()

    def move_to(
        self,
        target_position: Sequence[float],
        wait_timeout: float = 5.0,
        tolerance: float = 0.015,
        poll_interval: float = 0.1,
    ) -> ReachResult:
        """移动到目标位姿（仅位置），内部计算 IK 并到位等待。"""
        target = np.array(target_position, dtype=float)
        ik_solution = self.sim.calculate_ik(target)
        control_angles = [
            ik_solution[idx] for idx in self.sim.control_joint_indices if idx < len(ik_solution)
        ]
        # 记录“仿真目标角”（用于对比显示），以度为单位
        self.last_commanded_control_angles_deg = np.degrees(control_angles).tolist()
        self.sim.command_control_joint_positions(control_angles)

        samples: List[Tuple[float, float, float]] = []
        if self.real and self.real.is_connected:
            if not control_angles:
                raise RuntimeError("No IK outputs available for the controllable joints.")
            joint_angles_deg = np.rad2deg(control_angles)
            self.real.write_goal_positions(joint_angles_deg)
            result = self._wait_for_real_robot(target, wait_timeout, tolerance, poll_interval, samples)
        else:
            result = self._wait_for_simulation(target, wait_timeout, tolerance, poll_interval, samples, ik_solution)
        return result

    def set_joint_angles_deg(
        self,
        joint_angles_deg: Sequence[float],
        wait_timeout: float = 3.0,
        tolerance: float = 0.02,
        poll_interval: float = 0.1,
    ) -> ReachResult:
        """直接按控制关节顺序下发角度（度），并等待到位。"""
        joint_angles_deg = list(joint_angles_deg)
        # 记录“仿真目标角”（用于对比显示）
        self.last_commanded_control_angles_deg = joint_angles_deg[:]
        # Send to real robot (if any)
        if self.real and self.real.is_connected:
            self.real.write_goal_positions(joint_angles_deg)

        # Command simulation
        joint_angles_rad = np.deg2rad(joint_angles_deg).tolist()
        self.sim.command_control_joint_positions(joint_angles_rad)

        # Define target as the FK from the desired joints (for error calc)
        # Reset joint states temporarily to compute FK target
        self.sim.reset_joint_states(joint_angles_rad)
        target_pos = np.array(self.sim.get_end_effector_position(), dtype=float)

        samples: List[Tuple[float, float, float]] = []
        if self.real and self.real.is_connected:
            return self._wait_for_real_robot(target_pos, wait_timeout, tolerance, poll_interval, samples)
        else:
            # Use the same commanded values for sim waiting
            return self._wait_for_simulation(target_pos, wait_timeout, tolerance, poll_interval, samples, joint_angles_rad)

    # ----------------------
    # 角度对比相关工具方法
    # ----------------------
    def get_sim_joint_angles_deg(self) -> List[float]:
        """读取仿真“目标角”（优先使用最近一次下发的命令），否则读取当前仿真角。"""
        if self.last_commanded_control_angles_deg is not None:
            return self.last_commanded_control_angles_deg[:]
        degs: List[float] = []
        for joint_idx in self.sim.control_joint_indices:
            state = p.getJointState(self.sim.robot_id, joint_idx)
            degs.append(np.degrees(state[0]))
        return degs

    def get_real_joint_angles_deg(self) -> Optional[List[float]]:
        """读取真实机械臂的角度（度），长度与控制关节对齐。未连接返回 None。"""
        if not (self.real and self.real.is_connected):
            return None
        real_vals = self.real.read_joint_positions().tolist()
        n = len(self.sim.control_joint_indices)
        return real_vals[:n]

    def angle_comparison(self) -> Optional[Tuple[List[str], List[float], List[float], List[float]]]:
        """返回关节名称、仿真角、真实角与误差（真实-仿真），未连接时返回 None。"""
        real = self.get_real_joint_angles_deg()
        if real is None:
            return None
        sim = self.get_sim_joint_angles_deg()
        n = min(len(sim), len(real))
        sim = sim[:n]
        real = real[:n]
        errors = (np.array(real) - np.array(sim)).tolist()
        names = [p.getJointInfo(self.sim.robot_id, j)[1].decode("utf-8") for j in self.sim.control_joint_indices[:n]]
        return names, sim, real, errors

    def print_angle_comparison(self) -> None:
        """打印关节角度对比表。未连接真实机械臂时输出提示。"""
        data = self.angle_comparison()
        if data is None:
            print("(未连接真实机械臂，无法对比关节角)")
            return
        names, sim, real, errors = data
        print("-" * 80)
        print(f"{'关节名称':20s} | {'目标角(仿真)':>11s} | {'真实角度':>9s} | {'误差':>8s}")
        print("-" * 80)
        for i, name in enumerate(names):
            print(f"{name:20s} | {sim[i]:11.2f}° | {real[i]:9.2f}° | {errors[i]:+7.2f}°")
        print("-" * 80)

    def move_to_with_live_comparison(
        self,
        target_position: Sequence[float],
        wait_timeout: float = 6.0,
        tolerance: float = 0.02,
        poll_interval: float = 0.2,
    ) -> ReachResult:
        """移动到位并实时输出关节角对比（若连接真实机械臂）。

        说明：本方法不做闭环微调，只在开始时下发一次目标，随后实时显示真实角与误差，
        当误差小于阈值或超时结束。
        """
        target = np.array(target_position, dtype=float)
        ik_solution = self.sim.calculate_ik(target)
        control_angles = [
            ik_solution[idx] for idx in self.sim.control_joint_indices if idx < len(ik_solution)
        ]
        self.last_commanded_control_angles_deg = np.degrees(control_angles).tolist()
        self.sim.command_control_joint_positions(control_angles)

        samples: List[Tuple[float, float, float]] = []

        # 若未连接真实机械臂，则直接走仿真等待（打印一次对比提示）
        if not (self.real and self.real.is_connected):
            print("(未连接真实机械臂，实时对比不可用。)\n")
            return self._wait_for_simulation(target, wait_timeout, tolerance, poll_interval, samples, ik_solution)

        # 已连接真实机械臂：写入一次目标
        self.real.write_goal_positions(self.last_commanded_control_angles_deg)

        start_time = time.time()
        last_error = float("inf")
        last_position = tuple(self.sim.get_end_effector_position())

        print("实时关节角对比：按 Ctrl+C 中断显示\n")
        try:
            while time.time() - start_time <= wait_timeout:
                # 刷新真实角 → 用仿真 FK 计算末端误差
                joint_deg_full = self.real.read_joint_positions()
                n_ctrl = len(self.sim.control_joint_indices)
                real_deg_now = joint_deg_full[:n_ctrl]
                self.sim.reset_joint_states(np.deg2rad(real_deg_now))
                actual = self.sim.get_end_effector_position()
                error = float(np.linalg.norm(np.array(actual) - target))
                last_error = error
                last_position = actual

                # 打印对比表
                names, sim_deg, real_deg_show, errs = self.angle_comparison() or ([], [], [], [])
                print("\033[2J\033[H", end="")  # 清屏
                print(f"目标: [{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}]  |  误差: {error*1000:.1f} mm")
                print("-" * 80)
                print(f"{'关节名称':20s} | {'目标角(仿真)':>11s} | {'真实角度':>9s} | {'误差':>8s}")
                print("-" * 80)
                for i, name in enumerate(names):
                    print(f"{name:20s} | {sim_deg[i]:11.2f}° | {real_deg_show[i]:9.2f}° | {errs[i]:+7.2f}°")
                print("-" * 80)
                if error <= tolerance:
                    return ReachResult(True, error, actual, samples)
                time.sleep(poll_interval)
        except KeyboardInterrupt:
            pass
        return ReachResult(False, last_error, last_position, samples)

    def _wait_for_real_robot(
        self,
        target: np.ndarray,
        wait_timeout: float,
        tolerance: float,
        poll_interval: float,
        samples: List[Tuple[float, float, float]],
    ) -> ReachResult:
        assert self.real is not None
        start_time = time.time()
        last_error = float("inf")
        last_position = tuple(self.sim.get_end_effector_position())

        while time.time() - start_time <= wait_timeout:
            joint_deg = self.real.read_joint_positions()
            joint_rad = np.deg2rad(joint_deg)
            self.sim.reset_joint_states(joint_rad)
            actual = self.sim.get_end_effector_position()
            samples.append(actual)
            error = float(np.linalg.norm(np.array(actual) - target))
            last_error = error
            last_position = actual
            if error <= tolerance:
                return ReachResult(True, error, actual, samples)
            time.sleep(poll_interval)
        return ReachResult(False, last_error, last_position, samples)

    def _wait_for_simulation(
        self,
        target: np.ndarray,
        wait_timeout: float,
        tolerance: float,
        poll_interval: float,
        samples: List[Tuple[float, float, float]],
        ik_solution: Sequence[float],
    ) -> ReachResult:
        start_time = time.time()
        last_error = float("inf")
        last_position = tuple(self.sim.get_end_effector_position())

        while time.time() - start_time <= wait_timeout:
            self.sim.command_joint_positions(ik_solution)
            self.sim.step(steps=4, sleep=poll_interval / 4.0)
            actual = self.sim.get_end_effector_position()
            samples.append(actual)
            error = float(np.linalg.norm(np.array(actual) - target))
            last_error = error
            last_position = actual
            if error <= tolerance:
                return ReachResult(True, error, actual, samples)
        return ReachResult(False, last_error, last_position, samples)

    def shutdown(self) -> None:
        self.disconnect_real_robot()
        self.sim.shutdown()


def _format_vec(v: Sequence[float]) -> str:
    return f"[{v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f}]"

def help_message() -> None:
    print("支持命令: \n"
          "  connect                连接真实机械臂\n"
          "  disconnect             断开真实机械臂\n"
          "  current                显示当前末端位置（若连接则基于真实角度）\n"
          "  home                   关节归零（前 N 个控制关节）\n"
          "  save                   保存当前位置\n"
          "  list                   列出已保存位置\n"
          "  goto N                 移动到第 N 个保存位置\n"
          "  load                   移动到最后一次保存的位置\n"
          "  move x y z             移动到目标位置（单位 m）\n"
          "  x y z                  直接输入三数等同于 move\n"
          "  metrics start [interval] [window]  启动实时波形GUI (默认 0.1s / 20s)\n"
          "  metrics stop                      关闭实时波形GUI\n"
          "  veldebug start [interval]        开启速度调试输出（符号位/幅值/解码）\n"
          "  veldebug stop                    停止速度调试输出\n"
          "  help                   显示帮助\n"
          "  quit/exit              退出\n")


def run_cli() -> None:
    """命令行交互层：复用 ArmController 提供 connect/current/home/move 等命令。"""
    controller = ArmController(
        use_gui=True,
        calibration_dir=str(Path(__file__).parent),
        zero_offset_deg=[0, 0, 0, 0, 0, 0],
    )
    saved_positions: List[List[float]] = []
    # 实时波形 GUI 子进程（避免后端要求在主线程更新 GUI 的限制）
    metrics_proc: Optional[subprocess.Popen] = None
    telemetry_thread: Optional[threading.Thread] = None
    telemetry_stop = Event()
    TELEMETRY_PORT = 8765
    # 速度调试线程
    veldebug_thread: Optional[threading.Thread] = None
    veldebug_stop = Event()
    veldebug_interval = 0.2

    def telemetry_loop():
        # 简单 TCP 服务器：每次仅服务一个客户端，JSONL 推送
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", TELEMETRY_PORT))
        srv.listen(1)
        try:
            while not telemetry_stop.is_set():
                srv.settimeout(0.5)
                try:
                    conn, _ = srv.accept()
                except socket.timeout:
                    continue
                with conn:
                    f = conn.makefile("w")
                    # 初次发送名称
                    try:
                        if controller.real and controller.real.is_connected:
                            follower = next(iter(controller.real.robot.follower_arms.values()))  # type: ignore
                            names = list(follower.motor_names)
                        else:
                            names = []
                        header = {"names": names}
                        f.write(json.dumps(header) + "\n"); f.flush()
                        # 速度：同时采集 raw_bit15（寄存器原始速度解码为 deg/s）与由位置导数计算的 deg/s
                        prev_pos_arr = None
                        prev_t = None
                        while not telemetry_stop.is_set():
                            if not (controller.real and controller.real.is_connected):
                                break
                            follower = next(iter(controller.real.robot.follower_arms.values()))  # type: ignore
                            # 串口访问必须串行化：使用 RealRobotInterface 的 IO 锁
                            with controller.real._io_lock:  # type: ignore[attr-defined]
                                load = (
                                    follower.read_decoded("Present_Load")
                                    if hasattr(follower, "read_decoded")
                                    else follower.read("Present_Load")
                                )
                                current = (
                                    follower.read_decoded("Present_Current")
                                    if hasattr(follower, "read_decoded")
                                    else follower.read("Present_Current")
                                )
                                position = follower.read("Present_Position")
                                speed_raw_words = follower.read("Present_Speed")
                            # 由位置导数估计速度（deg/s）
                            now_t = time.perf_counter()
                            pos_arr = np.asarray(position, dtype=np.float32)
                            if prev_pos_arr is None or prev_t is None:
                                speed_deg = np.zeros_like(pos_arr)
                            else:
                                dt = now_t - float(prev_t)
                                if dt <= 1e-6:
                                    speed_deg = np.zeros_like(pos_arr)
                                else:
                                    speed_deg = (pos_arr - prev_pos_arr) / dt
                            prev_pos_arr = pos_arr
                            prev_t = now_t

                            # 原始寄存器速度 raw_bit15 → 带符号计数/秒，再按每路分辨率换算成 deg/s
                            raw_arr = np.asarray(speed_raw_words, dtype=np.int32)
                            neg = (raw_arr & 0x8000) != 0
                            mag = (raw_arr & 0x7FFF).astype(np.int32)
                            signed_counts = np.where(neg, -mag, mag).astype(np.float32)
                            # 逐电机分辨率
                            try:
                                names_local = list(follower.motor_names)
                                res_list = []
                                for nm in names_local:
                                    idx, model = follower.motors[nm]
                                    res_list.append(float(getattr(follower, "model_resolution", {}).get(model, 4096)))
                                res = np.asarray(res_list, dtype=np.float32)
                            except Exception:
                                res = np.full_like(signed_counts, 4096.0, dtype=np.float32)
                            speed_deg_raw = signed_counts / res * 360.0

                            pkt = {
                                "load": list(map(float, load)),
                                "current": list(map(float, current)),
                                "position": list(map(float, position)),
                                # 实时显示使用 raw_bit15（寄存器解码）的速度
                                "speed": list(map(float, speed_deg_raw)),
                                # 同时传递导数速度，便于外部对照或后续需求
                                "speed_derived": list(map(float, speed_deg)),
                            }
                            f.write(json.dumps(pkt) + "\n"); f.flush()
                            if telemetry_stop.wait(0.1):
                                break
                    except Exception:
                        # 客户端异常断开，继续等待下一个连接
                        pass
        finally:
            srv.close()

    def veldebug_loop():
        print("[veldebug] 速度调试已开启：raw_bit15 与 由位置导数 的速度 (deg/s)")
        prev_pos = None
        prev_t = None
        while not veldebug_stop.is_set():
            try:
                if not (controller.real and controller.real.is_connected):
                    print("[veldebug] 未连接真实机械臂")
                    if veldebug_stop.wait(1.0):
                        break
                    continue
                assert controller.real and controller.real.robot is not None
                follower = next(iter(controller.real.robot.follower_arms.values()))
                names = list(follower.motor_names)
                with controller.real._io_lock:  # type: ignore[attr-defined]
                    pos = follower.read("Present_Position")
                    speed_raw_words = follower.read("Present_Speed")
                now = time.perf_counter()
                pos_arr = np.asarray(pos, dtype=np.float32)
                if prev_pos is None or prev_t is None:
                    vel = np.zeros_like(pos_arr)
                else:
                    dt = now - float(prev_t)
                    if dt <= 1e-6:
                        vel = np.zeros_like(pos_arr)
                    else:
                        vel = (pos_arr - prev_pos) / dt
                prev_pos = pos_arr
                prev_t = now

                # raw_bit15 速度
                raw_arr = np.asarray(speed_raw_words, dtype=np.int32)
                neg = (raw_arr & 0x8000) != 0
                mag = (raw_arr & 0x7FFF).astype(np.int32)
                signed_counts = np.where(neg, -mag, mag).astype(np.float32)
                try:
                    res_list = []
                    for nm in names:
                        _, model = follower.motors[nm]
                        res_list.append(float(getattr(follower, "model_resolution", {}).get(model, 4096)))
                    res = np.asarray(res_list, dtype=np.float32)
                except Exception:
                    res = np.full_like(signed_counts, 4096.0, dtype=np.float32)
                vel_raw = signed_counts / res * 360.0
                print("-" * 80)
                for i, nm in enumerate(names):
                    print(f"{nm:16s} | pos={pos_arr[i]:8.3f} deg | raw_bit15={vel_raw[i]:+9.3f} deg/s | deriv={vel[i]:+9.3f} deg/s")
            except Exception as e:
                print(f"[veldebug] 读取失败: {e}")
            if veldebug_stop.wait(veldebug_interval):
                break

    print("=" * 80)
    print("机械臂交互式控制 (OOP 版本)")
    print("=" * 80)
    help_message()

    try:
        while True:
            cmd = input("\n> ").strip()
            if not cmd:
                continue
            low = cmd.lower()

            if low in ("quit", "exit"):
                print("退出程序...")
                break

            if low == "help":
                help_message()
                continue

            if low == "connect":
                try:
                    controller.connect_real_robot()
                    print("✅ 真实机械臂已连接（follower 臂）")
                    # 连接成功后：导出一次当前控制表（等效于运行 find_defaultSetting.py）
                    try:
                        # 动态按文件路径加载同目录下的 find_defaultSetting.py，避免包导入问题
                        import importlib.util as _ilu
                        _fds_path = Path(__file__).with_name("find_defaultSetting.py")
                        spec = _ilu.spec_from_file_location("find_defaultSetting", str(_fds_path))
                        if spec is None or spec.loader is None:
                            raise ImportError(f"spec not found for {_fds_path}")
                        _fds = _ilu.module_from_spec(spec)
                        spec.loader.exec_module(_fds)  # type: ignore[attr-defined]
                        result = {"arms": {}}
                        # 读取 follower 与（若存在）leader 的控制表
                        if controller.real and controller.real.robot is not None:
                            # 使用相同串口连接，避免重复连接导致冲突
                            with controller.real._io_lock:  # type: ignore[attr-defined]
                                for name, bus in controller.real.robot.follower_arms.items():  # type: ignore
                                    result["arms"][f"follower:{name}"] = _fds._read_all_from_bus(bus)  # type: ignore[attr-defined]
                                for name, bus in controller.real.robot.leader_arms.items():  # type: ignore
                                    result["arms"][f"leader:{name}"] = _fds._read_all_from_bus(bus)  # type: ignore[attr-defined]
                        out_dir = Path(__file__).parent / "outputs"
                        out_dir.mkdir(parents=True, exist_ok=True)
                        out_path = out_dir / f"default_settings_{int(time.time())}.json"
                        with open(out_path, "w", encoding="utf-8") as f:
                            json.dump(result, f, ensure_ascii=False, indent=2)
                        print(f"ℹ️ 已导出当前控制表到 {out_path}")
                    except Exception as ee:
                        print(f"⚠️ 控制表导出失败：{ee}")
                    # 调试：自动以 10Hz 启动 metrics（如未启动）
                    if metrics_proc is None or metrics_proc.poll() is not None:
                        # 启动遥测线程
                        if telemetry_thread is None or not telemetry_thread.is_alive():
                            telemetry_stop.clear()
                            telemetry_thread = threading.Thread(target=telemetry_loop, daemon=True)
                            telemetry_thread.start()
                        # 启动 viewer 进程（TCP 客户端）
                        viewer_path = str(Path(__file__).with_name("realtime_metrics_viewer.py"))
                        cmd = [
                            _sys.executable,
                            viewer_path,
                            "--mode", "tcp",
                            "--host", "127.0.0.1",
                            "--port", str(TELEMETRY_PORT),
                            "--interval", "0.1",
                            "--window", "20",
                        ]
                        try:
                            metrics_proc = subprocess.Popen(cmd)
                            print("ℹ️ 已自动启动实时波形GUI (10Hz)")
                        except Exception as e:
                            print(f"⚠️ 无法自动启动波形GUI: {e}")
                except Exception as e:
                    print(f"❌ 连接失败: {e}")
                continue

            if low == "disconnect":
                try:
                    # 断开前关闭波形 GUI 子进程
                    if metrics_proc is not None and metrics_proc.poll() is None:
                        try:
                            metrics_proc.terminate()
                            metrics_proc.wait(timeout=2.0)
                        except Exception:
                            try:
                                metrics_proc.kill()
                            except Exception:
                                pass
                        metrics_proc = None
                    controller.disconnect_real_robot()
                    print("✅ 真实机械臂已断开")
                except Exception as e:
                    print(f"❌ 断开失败: {e}")
                continue

            if low.startswith("metrics"):
                parts = low.split()
                if len(parts) >= 2 and parts[1] == "start":
                    if not (controller.real and controller.real.is_connected):
                        print("⚠️ 请先 connect 连接真实机械臂后再启动波形GUI")
                        continue
                    if metrics_proc is not None and metrics_proc.poll() is None:
                        print("⚠️ 实时波形GUI已在运行")
                        continue
                    try:
                        interval = float(parts[2]) if len(parts) >= 3 else 0.1
                        window = float(parts[3]) if len(parts) >= 4 else 20.0
                    except ValueError:
                        print("用法: metrics start [interval] [window]")
                        continue
                    # 启动遥测服务线程（共享本连接，避免串口竞争）
                    if telemetry_thread is None or not telemetry_thread.is_alive():
                        telemetry_stop.clear()
                        telemetry_thread = threading.Thread(target=telemetry_loop, daemon=True)
                        telemetry_thread.start()
                    # 启动独立进程运行 viewer（TCP 客户端模式）
                    viewer_path = str(Path(__file__).with_name("realtime_metrics_viewer.py"))
                    calib_dir = str(Path(__file__).parent)
                    cmd = [
                        _sys.executable,
                        viewer_path,
                        "--mode", "tcp",
                        "--host", "127.0.0.1",
                        "--port", str(TELEMETRY_PORT),
                        "--interval",
                        str(interval),
                        "--window",
                        str(window),
                    ]
                    try:
                        metrics_proc = subprocess.Popen(cmd)
                    except Exception as e:
                        print(f"❌ 启动波形GUI失败: {e}")
                        metrics_proc = None
                        continue
                    print(f"✅ 已启动实时波形GUI interval={interval}s, window={window}s")
                elif len(parts) >= 2 and parts[1] == "stop":
                    if metrics_proc is None or metrics_proc.poll() is not None:
                        print("⚠️ 实时波形GUI未运行")
                    else:
                        try:
                            metrics_proc.terminate()
                            metrics_proc.wait(timeout=2.0)
                        except Exception:
                            try:
                                metrics_proc.kill()
                            except Exception:
                                pass
                        metrics_proc = None
                        telemetry_stop.set()
                        if telemetry_thread and telemetry_thread.is_alive():
                            telemetry_thread.join(timeout=2.0)
                        telemetry_thread = None
                        print("✅ 已关闭实时波形GUI")
                else:
                    print("用法: metrics start [interval] [window] | metrics stop")
                continue

            # PID 调参：pid get [joint] | pid set (all|joint) P I D
            if low.startswith("pid"):
                parts = cmd.split()
                if not (controller.real and controller.real.is_connected):
                    print("⚠️ 未连接真实机械臂，无法设置/读取 PID")
                    continue
                # 取 follower 总线
                try:
                    follower_bus = next(iter(controller.real.robot.follower_arms.values()))  # type: ignore
                except Exception:
                    print("❌ 未找到 follower 总线")
                    continue
                bus_mod = getattr(follower_bus.__class__, "__module__", "")
                is_feetech = "feetech" in bus_mod
                is_dynamixel = "dynamixel" in bus_mod
                if len(parts) >= 2 and parts[1] == "get":
                    joint = parts[2] if len(parts) >= 3 else None
                    try:
                        if is_feetech:
                            if joint:
                                pval = follower_bus.read("P_Coefficient", joint)
                                ival = follower_bus.read("I_Coefficient", joint)
                                dval = follower_bus.read("D_Coefficient", joint)
                                print(f"[PID:{joint}] P={pval} I={ival} D={dval}")
                            else:
                                pval = follower_bus.read("P_Coefficient")
                                ival = follower_bus.read("I_Coefficient")
                                dval = follower_bus.read("D_Coefficient")
                                names = list(follower_bus.motors.keys())
                                for i, nm in enumerate(names):
                                    pv = pval[i] if i < len(pval) else pval
                                    iv = ival[i] if i < len(ival) else ival
                                    dv = dval[i] if i < len(dval) else dval
                                    print(f"[PID:{nm}] P={pv} I={iv} D={dv}")
                        elif is_dynamixel:
                            if joint:
                                pval = follower_bus.read("Position_P_Gain", joint)
                                ival = follower_bus.read("Position_I_Gain", joint)
                                dval = follower_bus.read("Position_D_Gain", joint)
                                print(f"[PID:{joint}] P={pval} I={ival} D={dval}")
                            else:
                                pval = follower_bus.read("Position_P_Gain")
                                ival = follower_bus.read("Position_I_Gain")
                                dval = follower_bus.read("Position_D_Gain")
                                names = list(follower_bus.motors.keys())
                                for i, nm in enumerate(names):
                                    pv = pval[i] if i < len(pval) else pval
                                    iv = ival[i] if i < len(ival) else ival
                                    dv = dval[i] if i < len(dval) else dval
                                    print(f"[PID:{nm}] P={pv} I={iv} D={dv}")
                        else:
                            print(f"⚠️ 未识别的总线类型：{bus_mod}")
                    except Exception as e:
                        print(f"❌ 读取 PID 失败: {e}")
                    continue
                elif len(parts) >= 2 and parts[1] == "set":
                    if len(parts) < 6:
                        print("用法: pid set (all|JOINT_NAME) P I D")
                        continue
                    target = parts[2]
                    try:
                        p_new = int(float(parts[3]))
                        i_new = int(float(parts[4]))
                        d_new = int(float(parts[5]))
                    except ValueError:
                        print("❌ P/I/D 必须是数字")
                        continue
                    try:
                        # 安全：写前尽量关力矩
                        try:
                            follower_bus.write("Torque_Enable", 0)
                        except Exception:
                            pass
                        if is_feetech:
                            if target.lower() == "all":
                                follower_bus.write("P_Coefficient", p_new)
                                follower_bus.write("I_Coefficient", i_new)
                                follower_bus.write("D_Coefficient", d_new)
                            else:
                                follower_bus.write("P_Coefficient", p_new, target)
                                follower_bus.write("I_Coefficient", i_new, target)
                                follower_bus.write("D_Coefficient", d_new, target)
                        elif is_dynamixel:
                            if target.lower() == "all":
                                follower_bus.write("Position_P_Gain", p_new)
                                follower_bus.write("Position_I_Gain", i_new)
                                follower_bus.write("Position_D_Gain", d_new)
                            else:
                                follower_bus.write("Position_P_Gain", p_new, target)
                                follower_bus.write("Position_I_Gain", i_new, target)
                                follower_bus.write("Position_D_Gain", d_new, target)
                        else:
                            print(f"⚠️ 未识别的总线类型：{bus_mod}")
                        # 写后开力矩
                        try:
                            follower_bus.write("Torque_Enable", 1)
                        except Exception:
                            pass
                        print(f"✅ PID 已写入 ({target}): P={p_new} I={i_new} D={d_new}")
                    except Exception as e:
                        print(f"❌ 写入 PID 失败: {e}")
                    continue
                else:
                    print("用法: pid get [JOINT] | pid set (all|JOINT) P I D")
                    continue

            # 速度调试：开启/停止
            if low.startswith("veldebug"):
                parts = low.split()
                if len(parts) >= 2 and parts[1] == "start":
                    if len(parts) >= 3:
                        try:
                            veldebug_interval = max(0.02, float(parts[2]))
                        except ValueError:
                            print("interval 参数无效，使用默认 0.2s")
                            veldebug_interval = 0.2
                    if veldebug_thread and veldebug_thread.is_alive():
                        print("veldebug 已在运行")
                    else:
                        veldebug_stop.clear()
                        veldebug_thread = threading.Thread(target=veldebug_loop, daemon=True)
                        veldebug_thread.start()
                        print(f"veldebug started (interval={veldebug_interval}s)")
                elif len(parts) >= 2 and parts[1] == "stop":
                    if veldebug_thread and veldebug_thread.is_alive():
                        veldebug_stop.set()
                        veldebug_thread.join(timeout=2.0)
                        print("veldebug stopped")
                    else:
                        print("veldebug 未在运行")
                else:
                    print("用法: veldebug start [interval] | veldebug stop")
                continue

            if low == "current":
                try:
                    if controller.real and controller.real.is_connected:
                        # 用真实角度刷新仿真后读位置
                        joint_deg = controller.real.read_joint_positions()
                        controller.sim.reset_joint_states(np.deg2rad(joint_deg))
                    pos = controller.sim.get_end_effector_position()
                    print(f"当前末端位置: {_format_vec(pos)}")
                except Exception as e:
                    print(f"❌ 读取失败: {e}")
                continue

            if low == "home":
                try:
                    n = len(controller.sim.control_joint_indices)
                    result = controller.set_joint_angles_deg([0.0] * n)
                    status = "到达" if result.reached else "未到达"
                    print(f"Home: {status}，误差 {result.error*1000:.1f} mm")
                except Exception as e:
                    print(f"❌ 执行失败: {e}")
                continue

            if low == "save":
                pos = controller.sim.get_end_effector_position()
                saved_positions.append(list(pos))
                print(f"✅ 已保存位置 #{len(saved_positions)}: {_format_vec(pos)}")
                continue

            if low == "list":
                if not saved_positions:
                    print("没有保存的位置")
                else:
                    for i, pos in enumerate(saved_positions, 1):
                        print(f"  {i}. {_format_vec(pos)}")
                continue

            if low.startswith("goto "):
                parts = low.split()
                if len(parts) == 2 and parts[1].isdigit():
                    idx = int(parts[1]) - 1
                    if 0 <= idx < len(saved_positions):
                        target = saved_positions[idx]
                        result = controller.move_to_with_live_comparison(target)
                        status = "到达" if result.reached else "未到达"
                        print(f"Move: {status}，误差 {result.error*1000:.1f} mm")
                    else:
                        print(f"错误: 位置编号需要在 1-{len(saved_positions)} 之间")
                else:
                    print("用法: goto N")
                continue

            if low == "load":
                if not saved_positions:
                    print("没有保存的位置")
                else:
                    target = saved_positions[-1]
                    result = controller.move_to_with_live_comparison(target)
                    status = "到达" if result.reached else "未到达"
                    print(f"Move: {status}，误差 {result.error*1000:.1f} mm")
                continue

            def try_parse_xyz(text: str) -> Optional[Tuple[float, float, float]]:
                parts = text.strip().split()
                if len(parts) == 3:
                    try:
                        return float(parts[0]), float(parts[1]), float(parts[2])
                    except ValueError:
                        return None
                if len(parts) == 4 and parts[0] == "move":
                    try:
                        return float(parts[1]), float(parts[2]), float(parts[3])
                    except ValueError:
                        return None
                return None

            xyz = try_parse_xyz(cmd)
            if xyz is not None:
                # 使用实时对比版本
                result = controller.move_to_with_live_comparison(xyz)
                status = "到达" if result.reached else "未到达"
                print(f"Target: {_format_vec(xyz)} | {status} | 误差 {result.error*1000:.1f} mm")
                
                continue

            print("无法识别的命令，输入 help 查看帮助")
    finally:
        try:
            if metrics_proc is not None and metrics_proc.poll() is None:
                metrics_proc.terminate()
                try:
                    metrics_proc.wait(timeout=2.0)
                except Exception:
                    metrics_proc.kill()
            telemetry_stop.set()
            if telemetry_thread and telemetry_thread.is_alive():
                telemetry_thread.join(timeout=2.0)
            veldebug_stop.set()
            if veldebug_thread and veldebug_thread.is_alive():
                veldebug_thread.join(timeout=2.0)
        except Exception:
            pass
        controller.shutdown()


if __name__ == "__main__":
    run_cli()
