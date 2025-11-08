"""
自动化数据采集脚本（SO101：标准按压动作）
=================================================

核心目标
- 使用 LeRobot 原生接口控制 SO101（Feetech 总线），严格采用速度 API（Goal_Speed/Acceleration）。
- 实现两阶段运动：
  1) 快速接近 SAFE（不记录）；2) 慢速按压到 TARGET 或出现“平台”后慢速返回 SAFE（全程记录）。
- 以 LeRobotDataset 标准格式、高频同步记录：position/velocity/load/current 与 action。

运行环境
- 已安装可用的 LeRobot 本地库；串口正确；SO101 已装配与完成标定。

重要说明
- 速度控制使用底层寄存器：Goal_Speed(46)、Acceleration(41)。速度百分比映射到 0..1023 幅值。

使用方法（示例）
  python learn_PyBullet/collect_pressing_dataset.py \
    --label wood_block \
    --n 10 \
    --fps 200 \
    --dataset-path data/my_physical_dataset \
    --tol-deg 2 \
    --reach mean \

参数总览（可选）
- 基础运行
  - --fps <int> 默认 200：采样频率（Hz）
  - --dataset-path <path> 默认 data/my_physical_dataset
  - --calibration-dir <path> 默认为脚本目录
- 到位判定（目标角度）
  - --tol-deg <float> 默认 1.0：到位角度公差（度）
  - --reach {all|mean|any} 默认 all：到位方式（严格/较宽/最宽）
  - --segment-timeout <sec> 默认 10：每段动作最大等待时间
- 平台判定（真实按压用，无法到达 target 时提前结束）
  - --end-condition {target|plateau|either} 默认 either：结束条件
  - --plateau-window-ms <ms> 默认 300：平台判定窗口（按 fps 转帧数）
  - --plateau-delta-deg <float> 默认 0.2：窗口内关节位移阈值（度）
  - --plateau-mode {mean|max} 默认 mean：位移度量
  - --plateau-min-load <pct> 默认 -1：窗口平均|load|阈值；<0 不启用
- 速度写入稳态（避免“突然很快”）
  - --speed-retries <int> 默认 2：寄存器写重试次数
  - --speed-verify-sleep-ms <ms> 默认 20：写后读回校验前等待
  - --post-speed-sleep-ms <ms> 默认 30：切速成功后发位姿前等待

python learn_PyBullet/collect_pressing_dataset.py --label wood_block --n 10 --fps 200 --dataset-path data/so101_press --end-condition either --plateau-window-ms 500 --plateau-delta-deg 0.001 --plateau-min-load 90 

"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from collections import deque
from pathlib import Path
from typing import Dict, List, Sequence, Optional
import threading
import socket
import json
import subprocess

import numpy as np
import torch

# LeRobot imports (local source)
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robot_devices.robots.configs import So101RobotConfig
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.robot_devices.utils import busy_wait


# 模块级遥测共享数据（由主循环写入，遥测线程只读取，避免串口竞争）
_TELEMETRY_LOCK = threading.Lock()
_TELEMETRY_PACKET: Optional[dict] = None

# 终端彩色输出辅助
_CLR_RESET = "\033[0m"
_CLR_RED = "\033[31m"
_CLR_YELLOW = "\033[33m"
_CLR_GREEN = "\033[32m"
_CLR_CYAN = "\033[36m"
_CLR_BOLD = "\033[1m"
def _c(text: str, color: str) -> str:
    try:
        return f"{color}{text}{_CLR_RESET}"
    except Exception:
        return text

# 轻量 IK/FK 封装：使用 PyBullet 进行末端位置 IK/FK（DIRECT，无 GUI）
class _Kinematics:
    def __init__(self, urdf_filename: str = "so101_new_calib.urdf") -> None:
        import pybullet as p
        import pybullet_data

        self.p = p
        self.client_id = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.robot_id = p.loadURDF(
            str(Path(__file__).with_name(urdf_filename)),
            basePosition=[0, 0, 0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
        )
        self.end_effector_index = self._find_end_effector()
        self.control_joint_indices = self._collect_control_joints()

    def _find_end_effector(self) -> int:
        end_effector = -1
        for index in range(self.p.getNumJoints(self.robot_id)):
            info = self.p.getJointInfo(self.robot_id, index)
            link_name = info[12].decode("utf-8")
            if "gripper_frame" in link_name.lower():
                end_effector = index
                break
        return end_effector if end_effector >= 0 else self.p.getNumJoints(self.robot_id) - 1

    def _collect_control_joints(self) -> List[int]:
        indices: List[int] = []
        for index in range(self.p.getNumJoints(self.robot_id)):
            joint_type = self.p.getJointInfo(self.robot_id, index)[2]
            if joint_type == self.p.JOINT_REVOLUTE:
                indices.append(index)
            if len(indices) == 6:
                break
        return indices

    def set_joints_deg(self, joint_angles_deg: Sequence[float]) -> None:
        arr = list(joint_angles_deg)
        for i, joint_idx in enumerate(self.control_joint_indices):
            val = arr[i] if i < len(arr) else 0.0
            self.p.resetJointState(self.robot_id, joint_idx, float(np.deg2rad(val)))

    def get_end_effector_position(self) -> np.ndarray:
        pos = self.p.getLinkState(self.robot_id, self.end_effector_index)[0]
        return np.asarray(pos, dtype=np.float32)

    def ik_to_joints_deg(self, target_position_xyz: Sequence[float]) -> List[float]:
        sol = self.p.calculateInverseKinematics(
            self.robot_id,
            self.end_effector_index,
            targetPosition=list(map(float, target_position_xyz)),
            maxNumIterations=1000,
            residualThreshold=1e-5,
        )
        # 映射为控制关节顺序长度
        out: List[float] = []
        for idx in self.control_joint_indices:
            if idx < len(sol):
                out.append(float(np.rad2deg(sol[idx])))
            else:
                out.append(0.0)
        return out

    def shutdown(self) -> None:
        try:
            if self.p.isConnected(self.client_id):
                self.p.disconnect(self.client_id)
        except Exception:
            pass


# =====================
# 关键配置（顶部集中）
# =====================

# 1) 关节角定义（度）：按 follower 实际电机顺序填写（长度需与电机数量一致）
#    建议先通过交互脚本/示波工具确认并填写。
HOME_JOINTS_DEG: List[float] = [0.0,-104.68,32.17,95.54,0.62, 0.0]
SAFE_JOINTS_DEG: List[float] = [-0.35,-66.97,42.71,31.38,0.0,0.0]
TARGET_JOINTS_DEG: List[float] = [-0.53, -19.42, 54.32, 25.31, 0.62, 0.0]  # 比 SAFE 更“下压”一些

# 2) 速度定义（百分比→原始幅值映射）
FAST_SPEED_PERCENT = 100
SLOW_SPEED_PERCENT = 10
ACCEL_FAST = 254  # Feetech 加速度寄存器建议范围 0..254
ACCEL_SLOW = 50

# Feetech 速度幅值上限（常见 0..1023 为 10bit 幅值），可按需调整。
GOAL_SPEED_MAX_UNITS = 1023

# 3) 数据集默认参数
DEFAULT_DATASET_PATH = "data/my_physical_dataset"  # 形如: data/<repo_id>
DEFAULT_FPS = 200  # 记录频率（Hz）

# 5) 速度推导与平滑（默认使用“由位置推导”的方式，提升分辨率）
DEFAULT_VEL_SOURCE = "derived"       # present | derived
DEFAULT_VEL_WINDOW = 5               # 用于差分的帧间隔（>=1）
DEFAULT_VEL_SMOOTH = "none"          # none | ma（移动平均）
DEFAULT_VEL_SMOOTH_WINDOW = 5        # 移动平均窗口（>=1）

# 4) 判定与控制常量（默认稳健值）
JOINT_TOL_DEG = 2.0         # 仅用于少数关节到位场景
SEGMENT_TIMEOUT_S = 15.0    # 单段超时（秒）

# 末端到位与平台（米）
EE_TOL_M = 0.035            # 末端到位容差（约 30 mm）
PLATEAU_WINDOW_MS = 400.0   # 平台窗口（毫秒）
PLATEAU_DELTA_M = 0.0015    # 平台净位移阈值（米，窗口首尾差约 1.5 mm）
PLATEAU_MIN_LOAD = 5.0      # 平均 |load| 最低百分比；<0 则不启用
MIN_PRESS_MS = 300.0        # 平台判定前的最短按压时长（毫秒）
END_CONDITION = "either"    # target 或 plateau 任一满足

# 切速稳态参数（写后读回 + 小延时）
SPEED_RETRIES = 3
SPEED_VERIFY_SLEEP_S = 0.05
POST_SPEED_SLEEP_S = 0.05


# =====================
# 辅助数据结构
# =====================

@dataclass
class EpisodeResult:
    frames: int
    duration_s: float


# =====================
# 工具函数
# =====================

def _percent_to_goal_speed_units(percent: float) -> int:
    """将 0..100 的百分比映射到 Goal_Speed 原生单位（0..GOAL_SPEED_MAX_UNITS）。"""
    pct = float(np.clip(percent, 0.0, 100.0))
    return int(round(pct / 100.0 * GOAL_SPEED_MAX_UNITS))


def _ensure_len(vec: Sequence[float], n: int) -> np.ndarray:
    """将关节角列表裁剪/补零到指定长度。"""
    arr = np.array(vec, dtype=np.float32)
    if arr.size < n:
        arr = np.pad(arr, (0, n - arr.size))
    return arr[:n]


def _reached(a: np.ndarray, b: np.ndarray, tol: float, mode: str = "all") -> bool:
    """判定是否到位：
    - all  : 所有关节误差均 <= tol（严格）
    - mean : 平均绝对误差 <= tol（宽松）
    - any  : 任一关节误差 <= tol（最宽松）
    """
    err = np.abs(a - b)
    if mode == "all":
        return np.all(err <= tol)
    if mode == "mean":
        return float(np.mean(err)) <= tol
    if mode == "any":
        return np.any(err <= tol)
    # 默认回退到严格模式
    return np.all(err <= tol)


def _create_or_load_dataset(dataset_path: Path, fps: int, num_motors: int, motor_names: List[str]) -> LeRobotDataset:
    """创建或加载 LeRobotDataset（仅包含我们需要的 4 个观测 + action）。

    约定（对齐 LeRobotDataset 的 __init__/create 语义）：
    - 已有数据集：使用 repo_id=dataset_path.name, root=dataset_path.parent 进行加载。
    - 新建数据集：直接在 dataset_path 下创建（root=dataset_path）。
    - 如目录存在但不是有效数据集（缺少 meta/info.json）或加载失败，则自动重命名到带时间戳的新目录后创建。
    """
    repo_id = dataset_path.name
    root_base = dataset_path.parent

    info_path = dataset_path / "meta" / "info.json"
    load_failed = False
    if info_path.exists():
        # 加载已存在数据集（本地）
        try:
            return LeRobotDataset(repo_id=repo_id, root=root_base)
        except Exception as e:
            print(f"⚠️ 现有数据集加载失败，将重建：{e}")
            load_failed = True

    # 自定义特征（注意：LeRobot 会自动补充默认列 DEFAULT_FEATURES）
    features = {
        "observation.position": {
            "dtype": "float32",
            "shape": (num_motors,),
            "names": list(motor_names),
        },
        "observation.velocity": {
            "dtype": "float32",
            "shape": (num_motors,),
            "names": list(motor_names),
        },
        # 新增：同时保留两种速度来源（由位置求导 & 原始寄存器 bit15 解码），便于对照分析
        "observation.velocity_derived": {
            "dtype": "float32",
            "shape": (num_motors,),
            "names": list(motor_names),
        },
        "observation.velocity_raw_bit15": {
            "dtype": "float32",
            "shape": (num_motors,),
            "names": list(motor_names),
        },
        # 兼容旧字段名：raw_present_speed（将写入与 velocity_raw_bit15 相同的值）
        "observation.raw_present_speed": {
            "dtype": "float32",
            "shape": (num_motors,),
            "names": list(motor_names),
        },
        "observation.load": {
            "dtype": "float32",
            "shape": (num_motors,),
            "names": list(motor_names),
        },
        "observation.current": {
            "dtype": "float32",
            "shape": (num_motors,),
            "names": list(motor_names),
        },
        "action": {
            "dtype": "float32",
            "shape": (num_motors,),
            "names": list(motor_names),
        },
    }

    # 智能重命名：若目标目录已存在（但不是有效数据集），自动生成新目录
    target_root = dataset_path
    if target_root.exists() and (load_failed or not info_path.exists()):
        ts = time.strftime("%Y%m%d_%H%M%S")
        base = target_root.name
        cand = target_root.parent / f"{base}_{ts}"
        idx = 1
        while cand.exists() and idx < 1000:
            cand = target_root.parent / f"{base}_{ts}_{idx}"
            idx += 1
        print(f"信息：数据集目录已存在，自动改名为 {cand}")
        target_root = cand
        repo_id = target_root.name

    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=target_root,
        features=features,
        use_videos=False,
    )
    return ds


def _set_speed_percent(
    follower_bus,
    speed_percent: float,
    accel: int,
    retries: int = 2,
    verify_sleep_s: float = 0.02,
    tolerance_units: int = 3,
) -> bool:
    """为所有从臂电机设置速度百分比和加速度（原生 Feetech API），并读回校验。

    返回 True 表示校验通过；False 表示重试后仍不一致（会继续程序，但发出警告）。
    """
    goal_speed = _percent_to_goal_speed_units(speed_percent)

    def _as_array(v):
        if isinstance(v, np.ndarray):
            return v
        if isinstance(v, (list, tuple)):
            return np.asarray(v)
        try:
            return np.asarray([v])
        except Exception:
            return np.asarray([-1])

    for attempt in range(retries + 1):
        follower_bus.write("Acceleration", int(accel))
        follower_bus.write("Goal_Speed", int(goal_speed))
        # 给予设备一点时间应用寄存器
        if verify_sleep_s > 0:
            time.sleep(verify_sleep_s)

        try:
            accel_rb = _as_array(follower_bus.read("Acceleration"))
        except Exception:
            accel_rb = np.asarray([-1])
        try:
            speed_rb = _as_array(follower_bus.read("Goal_Speed"))
        except Exception:
            speed_rb = np.asarray([-1])

        accel_ok = np.all(accel_rb == int(accel))
        speed_ok = np.all(np.abs(speed_rb - int(goal_speed)) <= int(tolerance_units))

        if accel_ok and speed_ok:
            return True

    # 打印部分读回值用于诊断
    def _preview(arr, k=3):
        try:
            a = np.asarray(arr).tolist()
            return a[:k]
        except Exception:
            return [str(arr)]

    print(
        f"⚠️ 速度设置读回不一致：exp accel={accel} got={_preview(accel_rb)}, exp speed={goal_speed} got={_preview(speed_rb)}"
    )
    return False


def _wait_until_reached(
    follower_bus,
    goal_deg: np.ndarray,
    tol_deg: float,
    reach_mode: str = "all",
    fps: int | None = None,
    timeout_s: float = SEGMENT_TIMEOUT_S,
) -> bool:
    """阻塞等待直到到位或超时。可选按 fps 循环。

    返回 True 表示到位，False 表示超时。
    """
    start = time.perf_counter()
    while True:
        present = follower_bus.read("Present_Position")
        present = np.asarray(present, dtype=np.float32)
        if _reached(present, goal_deg.astype(np.float32), tol_deg, reach_mode):
            return True
        if time.perf_counter() - start > timeout_s:
            return False
        if fps is not None:
            busy_wait(1.0 / float(fps))


def _wait_until_reached_ee(
    follower_bus,
    kin: _Kinematics,
    target_xyz: np.ndarray,
    tol_m: float = EE_TOL_M,
    timeout_s: float = SEGMENT_TIMEOUT_S,
    poll_hz: float = 50.0,
) -> bool:
    """等待末端到达目标位置（米），基于真实关节 + FK 计算误差。"""
    start = time.perf_counter()
    period = 1.0 / float(poll_hz)
    target_xyz = np.asarray(target_xyz, dtype=np.float32)
    last_dbg = 0.0
    while True:
        pos_deg = np.asarray(follower_bus.read("Present_Position"), dtype=np.float32)
        kin.set_joints_deg(pos_deg.tolist())
        ee_now = kin.get_end_effector_position()
        err = float(np.linalg.norm(ee_now - target_xyz))
        # 调试：仅打印误差（毫米），限速输出（每 ≥1s 一次）
        now = time.perf_counter()
        if now - last_dbg >= 1.0:
            print(_c(f"err={err*1000:.1f} mm", _CLR_CYAN))
            last_dbg = now
        if err <= tol_m:
            # 到达目标：打印依据（误差与阈值）
            print(_c(
                f"Reached target: err={err*1000:.1f} mm <= tol={tol_m*1000:.1f} mm",
                _CLR_GREEN,
            ))
            return True
        if time.perf_counter() - start > timeout_s:
            return False
        busy_wait(period)


def _record_pressing_episode(
    robot: ManipulatorRobot,
    follower_name: str,
    dataset: LeRobotDataset,
    label: str,
    fps: int,
    slow_percent: float,
    slow_accel: int,
    kin: _Kinematics,
    safe_ee_xyz: np.ndarray,
    target_ee_xyz: np.ndarray,
    safe_deg: np.ndarray,
    target_deg: np.ndarray,
    tol_deg: float,
    reach_mode: str,
    end_condition: str,
    plateau_window_frames: int,
    plateau_delta_deg: float,
    plateau_mode: str,
    plateau_min_load_pct: float,
    speed_retries: int = SPEED_RETRIES,
    speed_verify_sleep_s: float = SPEED_VERIFY_SLEEP_S,
    post_speed_sleep_s: float = POST_SPEED_SLEEP_S,
    velocity_source: str = DEFAULT_VEL_SOURCE,
    vel_window_frames: int = DEFAULT_VEL_WINDOW,
    vel_smooth: str = DEFAULT_VEL_SMOOTH,
    vel_smooth_window: int = DEFAULT_VEL_SMOOTH_WINDOW,
) -> EpisodeResult:
    """执行慢速“下压-返回”并全程记录到 dataset。"""
    follower = robot.follower_arms[follower_name]

    # 切换慢速（带读回校验）
    _set_speed_percent(
        follower,
        slow_percent,
        slow_accel,
        retries=speed_retries,
        verify_sleep_s=speed_verify_sleep_s,
    )
    if post_speed_sleep_s > 0:
        time.sleep(post_speed_sleep_s)

    frames = 0
    t_start = time.perf_counter()

    # 1) 慢速下压：目标 target_ee（此处 target_deg 传入的即为 IK 解）
    follower.write("Goal_Position", target_deg)
    load_hist = []      # 每帧负载强度（mean |load|）
    ee_window: List[np.ndarray] = []  # 末端位置窗口
    plateau_triggered = False
    last_dbg = 0.0
    # 节流调试输出
    last_dbg = 0.0
    # 推导速度所需的缓存
    pos_buffer: deque[np.ndarray] = deque(maxlen=max(2, int(vel_window_frames) + 1))
    vel_ma_buffer: deque[np.ndarray] = deque(maxlen=max(1, int(vel_smooth_window)))

    while True:
        # 观测
        pos = np.asarray(follower.read("Present_Position"), dtype=np.float32)
        # 读取原始速度字并按 bit15 解码为 deg/s（raw_bit15）
        speed_words = np.asarray(follower.read("Present_Speed"), dtype=np.int32)
        neg = (speed_words & 0x8000) != 0
        mag = (speed_words & 0x7FFF).astype(np.int32)
        signed_counts = np.where(neg, -mag, mag).astype(np.float32)
        try:
            res_list = []
            for nm in list(follower.motor_names):
                _, model = follower.motors[nm]
                res_list.append(float(getattr(follower, "model_resolution", {}).get(model, 4096)))
            res = np.asarray(res_list, dtype=np.float32)
        except Exception:
            res = np.full_like(signed_counts, 4096.0, dtype=np.float32)
        vel_raw_bit15 = signed_counts / res * 360.0

        # derived：用位置推导速度
        pos_buffer.append(pos.copy())
        if len(pos_buffer) >= 2:
            k = min(int(vel_window_frames), len(pos_buffer) - 1)
            if k < 1:
                vel_derived = np.zeros_like(pos)
            else:
                dt = float(k) / float(fps)
                vel_inst = (pos_buffer[-1] - pos_buffer[-1 - k]) / max(dt, 1e-9)
                if vel_smooth == "ma":
                    vel_ma_buffer.append(vel_inst)
                    vel_derived = np.mean(np.stack(list(vel_ma_buffer), axis=0), axis=0).astype(np.float32)
                else:
                    vel_derived = vel_inst.astype(np.float32)
        else:
            vel_derived = np.zeros_like(pos)

        # 主 velocity 字段按配置来源选择（默认 derived）
        vel = vel_raw_bit15 if velocity_source == "present" else vel_derived
        load = np.asarray(getattr(follower, "read_decoded", follower.read)("Present_Load"), dtype=np.float32)
        # 电流存 mA：优先 read_decoded（6.5mA/LSB，bit15 符号）
        curr_read = getattr(follower, "read_decoded", follower.read)
        curr = np.asarray(curr_read("Present_Current"), dtype=np.float32)

        # 动作（当前段恒等于 target）
        action = target_deg.astype(np.float32)

        frame = {
            "observation.position": pos,
            "observation.velocity": vel,
            "observation.velocity_derived": vel_derived,
            "observation.velocity_raw_bit15": vel_raw_bit15,
            # 兼容旧列名：写入与 raw_bit15 相同值
            "observation.raw_present_speed": vel_raw_bit15,
            "observation.load": load,
            "observation.current": curr,
            "action": action,
            "task": label,
        }
        # 兼容旧数据集：若字段不匹配，依次回退去掉新增速度字段再写入
        try:
            dataset.add_frame(frame)
        except Exception:
            try:
                frame.pop("observation.velocity_raw_bit15", None)
                frame.pop("observation.velocity_derived", None)
                frame.pop("observation.raw_present_speed", None)
                dataset.add_frame(frame)
            except Exception:
                raise
        frames += 1

        # 更新遥测包（按压段）
        try:
            with _TELEMETRY_LOCK:
                globals()["_TELEMETRY_PACKET"] = {
                    "load": list(map(float, load)),
                    "current": list(map(float, curr)),
                    "position": list(map(float, pos)),
                    # 实时显示采用 raw_bit15
                    "speed": list(map(float, vel_raw_bit15)),
                    # 同时推送 derived 便于外部工具使用
                    "speed_derived": list(map(float, vel_derived)),
                }
        except Exception:
            pass

        # 结束条件 1：达到末端目标（仅用于调试，不作为结束依据）
        kin.set_joints_deg(pos.tolist())
        ee_now = kin.get_end_effector_position()
        err = float(np.linalg.norm(ee_now - target_ee_xyz))
        # 调试：按压阶段以表格形式输出每个关节 load、误差与窗口平台度量（限速，每 ≥1s 一次）
        mean_load = float(np.mean(np.abs(load)))
        now = time.perf_counter()
        if now - last_dbg >= 1.0:
            names = list(getattr(follower, "motor_names", [f"J{i+1}" for i in range(len(load))]))
            loads = [f"{abs(float(v)):.1f}" for v in load]
            header = _c("| Joint | " + " | ".join(names) + " |", _CLR_BOLD)
            values = _c("| Load% | " + " | ".join(loads) + " |", _CLR_CYAN)
            # 计算平台窗口净位移（米）
            if plateau_window_frames > 0 and len(ee_window) >= plateau_window_frames:
                move_metric = float(np.linalg.norm(ee_window[-1] - ee_window[0]))
                move_str = f"{move_metric*1000:.2f} mm"
            else:
                move_metric = None
                move_str = "n/a"
            # 彩色高亮误差和平台位移
            err_str = _c(f"{err*1000:.1f} mm", _CLR_YELLOW if err*1000 > 10 else _CLR_GREEN)
            move_col = _CLR_GREEN if (move_metric is not None and move_metric <= plateau_delta_deg) else _CLR_YELLOW
            summary = _c(f"| err | {err_str} | mean_load | {mean_load:.1f} | move | {move_str} |", move_col)
            sep = "+" + "-" * (len(header) - 2) + "+"
            print("\n".join([sep, header, values, summary, sep]))
            last_dbg = now

        # 结束条件 2：末端“平台”判定：窗口首尾净位移很小（可叠加负载门槛）
        load_hist.append(float(np.mean(np.abs(load))))
        if len(load_hist) > plateau_window_frames:
            load_hist.pop(0)
        ee_window.append(ee_now.copy())
        if len(ee_window) > plateau_window_frames:
            ee_window.pop(0)

        plateau_ok = False
        if plateau_window_frames > 0 and len(ee_window) >= plateau_window_frames:
            pos_start = ee_window[0]
            pos_now = ee_window[-1]
            move_metric = float(np.linalg.norm(pos_now - pos_start))
            load_ok = True if plateau_min_load_pct < 0 else (float(np.mean(load_hist)) >= plateau_min_load_pct)
            plateau_ok = (move_metric <= plateau_delta_deg) and load_ok

        # 前往 TARGET 时仅使用“平台”判定，忽略误差阈值
        if plateau_ok:
            plateau_triggered = True
            print(_c(
                f"Reached TARGET by plateau: move<={plateau_delta_deg*1000:.1f} mm",
                _CLR_GREEN,
            ))
            break

        # 节拍
        busy_wait(1.0 / float(fps))

    # 2) 慢速返回：目标 safe_deg（继续记录同一 episode）
    if plateau_triggered:
        print("信息：按压阶段以平台判定提前结束，开始慢速返回 SAFE。")
    # 再次确认慢速配置以避免中途被覆盖
    _set_speed_percent(
        follower,
        slow_percent,
        slow_accel,
        retries=speed_retries,
        verify_sleep_s=speed_verify_sleep_s,
    )
    if post_speed_sleep_s > 0:
        time.sleep(post_speed_sleep_s)
    follower.write("Goal_Position", safe_deg)
    last_dbg = 0.0
    last_dbg = 0.0
    # 重置推导速度缓存（返回段也使用相同策略）
    pos_buffer.clear()
    vel_ma_buffer.clear()
    while True:
        pos = np.asarray(follower.read("Present_Position"), dtype=np.float32)
        # 原始速度 raw_bit15 解码
        speed_words = np.asarray(follower.read("Present_Speed"), dtype=np.int32)
        neg = (speed_words & 0x8000) != 0
        mag = (speed_words & 0x7FFF).astype(np.int32)
        signed_counts = np.where(neg, -mag, mag).astype(np.float32)
        try:
            res_list = []
            for nm in list(follower.motor_names):
                _, model = follower.motors[nm]
                res_list.append(float(getattr(follower, "model_resolution", {}).get(model, 4096)))
            res = np.asarray(res_list, dtype=np.float32)
        except Exception:
            res = np.full_like(signed_counts, 4096.0, dtype=np.float32)
        vel_raw_bit15 = signed_counts / res * 360.0

        # 由位置导数估计
        pos_buffer.append(pos.copy())
        if len(pos_buffer) >= 2:
            k = min(int(vel_window_frames), len(pos_buffer) - 1)
            if k < 1:
                vel_derived = np.zeros_like(pos)
            else:
                dt = float(k) / float(fps)
                vel_inst = (pos_buffer[-1] - pos_buffer[-1 - k]) / max(dt, 1e-9)
                if vel_smooth == "ma":
                    vel_ma_buffer.append(vel_inst)
                    vel_derived = np.mean(np.stack(list(vel_ma_buffer), axis=0), axis=0).astype(np.float32)
                else:
                    vel_derived = vel_inst.astype(np.float32)
        else:
            vel_derived = np.zeros_like(pos)

        vel = vel_raw_bit15 if velocity_source == "present" else vel_derived
        load = np.asarray(getattr(follower, "read_decoded", follower.read)("Present_Load"), dtype=np.float32)
        curr_read = getattr(follower, "read_decoded", follower.read)
        curr = np.asarray(curr_read("Present_Current"), dtype=np.float32)

        action = safe_deg.astype(np.float32)

        frame = {
            "observation.position": pos,
            "observation.velocity": vel,
            "observation.velocity_derived": vel_derived,
            "observation.velocity_raw_bit15": vel_raw_bit15,
            # 兼容旧列名：写入与 raw_bit15 相同值
            "observation.raw_present_speed": vel_raw_bit15,
            "observation.load": load,
            "observation.current": curr,
            "action": action,
            "task": label,
        }
        try:
            dataset.add_frame(frame)
        except Exception:
            try:
                frame.pop("observation.velocity_raw_bit15", None)
                frame.pop("observation.velocity_derived", None)
                frame.pop("observation.raw_present_speed", None)
                dataset.add_frame(frame)
            except Exception:
                raise
        frames += 1

        # 更新遥测包（返回段）
        try:
            with _TELEMETRY_LOCK:
                globals()["_TELEMETRY_PACKET"] = {
                    "load": list(map(float, load)),
                    "current": list(map(float, curr)),
                    "position": list(map(float, pos)),
                    # 实时显示采用 raw_bit15
                    "speed": list(map(float, vel_raw_bit15)),
                    "speed_derived": list(map(float, vel_derived)),
                }
        except Exception:
            pass

        kin.set_joints_deg(pos.tolist())
        ee_now = kin.get_end_effector_position()
        err = float(np.linalg.norm(ee_now - safe_ee_xyz))
        # 调试：返回阶段仅打印误差（限速，每 ≥1s 一次）
        now = time.perf_counter()
        if now - last_dbg >= 1.0:
            print(_c(f"err={err*1000:.1f} mm", _CLR_CYAN))
            last_dbg = now
        if err <= EE_TOL_M:
            print(_c(
                f"Reached SAFE by error threshold: err={err*1000:.1f} mm <= tol={EE_TOL_M*1000:.1f} mm",
                _CLR_GREEN,
            ))
            break

        busy_wait(1.0 / float(fps))

    # episode 落盘
    dataset.save_episode()
    return EpisodeResult(frames=frames, duration_s=time.perf_counter() - t_start)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SO101 自动按压数据采集（两阶段 + LeRobotDataset）")
    parser.add_argument("--label", "-l", type=str, default="", help="物品标签（写入 task）")
    parser.add_argument("--run-name", "-r", type=str, default="", help="自定义保存到的数据子文件夹名（位于 --dataset-path 下）")
    parser.add_argument("--n", "--num", dest="num", type=int, required=True, help="采集的 episode 次数")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="记录频率（Hz）")
    parser.add_argument("--tol-deg", type=float, default=JOINT_TOL_DEG, help="到位判定角度公差（度）")
    parser.add_argument(
        "--segment-timeout",
        type=float,
        default=SEGMENT_TIMEOUT_S,
        help="单段动作最大等待时间（秒）",
    )
    parser.add_argument(
        "--reach",
        type=str,
        choices=["all", "mean", "any"],
        default="mean",
        help="到位判定方式：all(严格)/mean(较宽)/any(最宽)",
    )
    # 末端停滞（平台）判定相关参数
    parser.add_argument(
        "--end-condition",
        type=str,
        choices=["target", "plateau", "either"],
        default="either",
        help="按压结束条件：到达target / 出现平台 / 二者其一",
    )
    parser.add_argument(
        "--plateau-window-ms",
        type=float,
        default=300.0,
        help="平台判定窗口时长（毫秒），将按 fps 转换为帧数",
    )
    parser.add_argument(
        "--plateau-delta-deg",
        type=float,
        default=0.005,
        help="平台判定的末端净位移阈值（米，窗口首尾），建议 0.005 (5mm)",
    )
    parser.add_argument(
        "--plateau-mode",
        type=str,
        choices=["mean", "max"],
        default="mean",
        help="平台位移度量：mean(所有关节平均)|max(所有关节最大)",
    )
    parser.add_argument(
        "--plateau-min-load",
        type=float,
        default=-1.0,
        help="可选：窗口内平均|load|需不小于该百分比，<0 表示不启用",
    )
    # 速度来源与平滑选项
    parser.add_argument(
        "--velocity-source",
        type=str,
        choices=["present", "derived"],
        default=DEFAULT_VEL_SOURCE,
        help="速度来源：present=直接使用驱动器 Present_Speed；derived=由位置差分推导（更细的分辨率）",
    )
    parser.add_argument(
        "--vel-window-frames",
        type=int,
        default=DEFAULT_VEL_WINDOW,
        help="由位置推导速度时使用的差分帧间隔（>=1）",
    )
    parser.add_argument(
        "--vel-smooth",
        type=str,
        choices=["none", "ma"],
        default=DEFAULT_VEL_SMOOTH,
        help="派生速度的平滑方式：none(不平滑) / ma(移动平均)",
    )
    parser.add_argument(
        "--vel-smooth-window",
        type=int,
        default=DEFAULT_VEL_SMOOTH_WINDOW,
        help="移动平均窗口大小（>=1）",
    )
    parser.add_argument(
        "--speed-retries",
        type=int,
        default=2,
        help="速度/加速度写寄存器失败时的重试次数",
    )
    parser.add_argument(
        "--speed-verify-sleep-ms",
        type=float,
        default=20.0,
        help="写入后用于读回校验的等待时长（毫秒）",
    )
    parser.add_argument(
        "--post-speed-sleep-ms",
        type=float,
        default=30.0,
        help="切速成功后、发送位姿命令前的等待时长（毫秒）",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help="数据集根路径（形如 data/<repo_id>）",
    )
    parser.add_argument("--calibration-dir", type=str, default=str(Path(__file__).parent), help="标定目录")
    args = parser.parse_args(argv)

    # 交互式：启动时选择物品 label（若直接回车则采用当前默认）
    def _prompt_choice(prompt: str, default: str) -> str:
        try:
            s = input(f"{prompt} [{default}]: ").strip()
            return s if s else default
        except Exception:
            return default

    # 物品标签
    label_default = args.label or "item"
    label = _prompt_choice("物品标签(label)", label_default)
    # 保存子文件夹名（run-name）
    run_name_default = args.run_name or f"{label}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_name = _prompt_choice("保存子文件夹名(run-name)", run_name_default)

    label: str = label
    N: int = args.num
    fps: int = args.fps
    tol_deg: float = args.tol_deg
    segment_timeout: float = args.segment_timeout
    reach_mode: str = args.reach
    # 兼容旧 CLI：切速参数改为固定默认
    speed_retries: int = SPEED_RETRIES
    speed_verify_sleep_s: float = SPEED_VERIFY_SLEEP_S
    post_speed_sleep_s: float = POST_SPEED_SLEEP_S
    # 末端平台参数：窗口与阈值改为“米”单位
    end_condition: str = args.end_condition
    plateau_window_frames: int = int(max(1.0, args.plateau_window_ms / 1000.0) * fps)
    plateau_delta_deg: float = float(args.plateau_delta_deg)  # 这里视为“米”阈值
    plateau_mode: str = args.plateau_mode
    plateau_min_load_pct: float = -1.0
    dataset_path = Path(args.dataset_path)
    # 速度选项
    velocity_source: str = args.velocity_source
    vel_window_frames: int = max(1, int(args.vel_window_frames))
    vel_smooth: str = args.vel_smooth
    vel_smooth_window: int = max(1, int(args.vel_smooth_window))

    # 连接机器人（仅 follower，关闭相机以保持纯 headless）
    robot = ManipulatorRobot(
        So101RobotConfig(
            calibration_dir=args.calibration_dir,
            leader_arms={},
            cameras={},
        )
    )

    # 在整个任务周期内保持一次连接
    robot.connect()
    follower_name = next(iter(robot.follower_arms.keys()))
    follower = robot.follower_arms[follower_name]

    # 启动 10Hz 遥测与实时波形（四个信号：load/current/position/speed）
    TELEMETRY_PORT = 8766
    telemetry_stop = threading.Event()
    # 使用模块级共享包与锁（由数据采集循环更新，遥测线程仅转发）
    global _TELEMETRY_LOCK, _TELEMETRY_PACKET

    def _telemetry_server():
        print(f"[metrics] 启动 10Hz 遥测服务器在 127.0.0.1:{TELEMETRY_PORT}")
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
                    print("[metrics] 客户端已连接")
                    f = conn.makefile("w")
                    try:
                        names = list(follower.motor_names)
                        f.write(json.dumps({"names": names}) + "\n"); f.flush()
                        while not telemetry_stop.is_set():
                            # 从主循环采样的最新数据发送，不直接访问串口，避免端口竞争
                            with _TELEMETRY_LOCK:
                                pkt = _TELEMETRY_PACKET
                            if pkt is None:
                                pkt = {"load": [], "current": [], "position": [], "speed": []}
                            f.write(json.dumps(pkt) + "\n"); f.flush()
                            if telemetry_stop.wait(0.1):
                                break
                    except Exception:
                        pass
        finally:
            try:
                srv.close()
            except Exception:
                pass

    telemetry_thread = threading.Thread(target=_telemetry_server, daemon=True)
    telemetry_thread.start()

    # 启动 Viewer 进程（TCP 客户端模式，10Hz）
    try:
        viewer_path = str(Path(__file__).with_name("realtime_metrics_viewer.py"))
        metrics_proc = subprocess.Popen([
            sys.executable,
            viewer_path,
            "--mode", "tcp",
            "--host", "127.0.0.1",
            "--port", str(TELEMETRY_PORT),
            "--interval", "0.1",
            "--window", "20",
        ])
    except Exception:
        metrics_proc = None

    # 记录原始速度/加速度（用于退出时恢复）
    try:
        _orig_accel = np.asarray(follower.read("Acceleration")).copy()
    except Exception:
        _orig_accel = None
    try:
        _orig_speed = np.asarray(follower.read("Goal_Speed")).copy()
    except Exception:
        _orig_speed = None

    # 电机数量 & 名称
    motor_names = list(follower.motor_names)
    num_motors = len(motor_names)

    # 准备数据集（创建或加载）
    # 支持自定义保存的子文件夹名：data/<repo>/<run_name>
    if run_name:
        dataset_path = dataset_path / run_name
    dataset = _create_or_load_dataset(dataset_path, fps=fps, num_motors=num_motors, motor_names=motor_names)

    # Kinematics：将预设关节角转换为末端坐标（作为目标），并计算 IK 解用于下发
    kin = _Kinematics()
    kin.set_joints_deg(_ensure_len(SAFE_JOINTS_DEG, 6))
    safe_ee_xyz = kin.get_end_effector_position()
    kin.set_joints_deg(_ensure_len(TARGET_JOINTS_DEG, 6))
    target_ee_xyz = kin.get_end_effector_position()
    safe_cmd_list = kin.ik_to_joints_deg(safe_ee_xyz)
    target_cmd_list = kin.ik_to_joints_deg(target_ee_xyz)
    safe_deg = _ensure_len(safe_cmd_list, num_motors)
    target_deg = _ensure_len(target_cmd_list, num_motors)

    try:
        for i in range(1, N + 1):
            print(f"正在采集 '{label}'，第 [{i} / {N}] 次...")

            # -- 快速接近 SAFE（不记录）--
            _set_speed_percent(
                follower,
                FAST_SPEED_PERCENT,
                ACCEL_FAST,
                retries=speed_retries,
                verify_sleep_s=speed_verify_sleep_s,
            )
            if post_speed_sleep_s > 0:
                time.sleep(post_speed_sleep_s)
            follower.write("Goal_Position", safe_deg)
            ok = _wait_until_reached_ee(
                follower,
                kin,
                target_xyz=safe_ee_xyz,
                tol_m=EE_TOL_M,
                timeout_s=segment_timeout,
                poll_hz=50.0,
            )
            if not ok:
                print("⚠️ 快速接近阶段超时：未到达 SAFE，跳过该次 episode。")
                continue

            # -- 慢速下压并记录 --
            ep = _record_pressing_episode(
                robot=robot,
                follower_name=follower_name,
                dataset=dataset,
                label=label,
                fps=fps,
                slow_percent=SLOW_SPEED_PERCENT,
                slow_accel=ACCEL_SLOW,
                kin=kin,
                safe_ee_xyz=safe_ee_xyz,
                target_ee_xyz=target_ee_xyz,
                end_condition=end_condition,
                plateau_window_frames=plateau_window_frames,
                plateau_delta_deg=plateau_delta_deg,
                plateau_mode=plateau_mode,
                plateau_min_load_pct=plateau_min_load_pct,
                safe_deg=safe_deg,
                target_deg=target_deg,
                tol_deg=tol_deg,
                reach_mode=reach_mode,
                velocity_source=velocity_source,
                vel_window_frames=vel_window_frames,
                vel_smooth=vel_smooth,
                vel_smooth_window=vel_smooth_window,
            )
            print(f"✓ 保存 episode：{ep.frames} 帧，用时 {ep.duration_s:.2f}s")

    except KeyboardInterrupt:
        print("\n用户中断，执行安全停止...")
        try:
            # 尝试迅速减速并“原地刹停”，随后慢速回 HOME
            follower.write("Goal_Speed", 0)
            present = np.asarray(follower.read("Present_Position"), dtype=np.float32)
            follower.write("Goal_Position", present)
            # 小速度回 HOME（如有设置）
            home = _ensure_len(HOME_JOINTS_DEG, num_motors)
            _set_speed_percent(follower, 10.0, ACCEL_SLOW)
            follower.write("Goal_Position", home)
        except Exception as e:
            print(f"⚠️ 安全停止流程异常：{e}")
    finally:
        try:
            # 恢复原始速度/加速度设置（若可用）
            try:
                if _orig_accel is not None:
                    follower.write("Acceleration", _orig_accel)
            except Exception:
                pass
            try:
                if _orig_speed is not None:
                    follower.write("Goal_Speed", _orig_speed)
            except Exception:
                pass
            try:
                kin.shutdown()
            except Exception:
                pass
            robot.disconnect()
        except Exception:
            pass

        # 关闭调试遥测与波形
        try:
            telemetry_stop.set()
            if telemetry_thread.is_alive():
                telemetry_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            if metrics_proc is not None and metrics_proc.poll() is None:
                metrics_proc.terminate()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
