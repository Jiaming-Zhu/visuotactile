#!/usr/bin/env python3
"""
collect_teleop_multimodal.py
============================

本脚本用于实现“基于遥操作的多模态数据采集”。
它的目标是尽量复刻 `collect_custom_multimodal.py` 的数据记录能力和输出结构，
但把原脚本中的“固定轨迹回放”替换成“实时遥操作控制”。

===============================================================================
一、核心用途
===============================================================================
1. 使用 LeRobot 的主从遥操作链路，让 leader 实时控制 follower。
2. 在用户手动遥操作 follower 的过程中，高频记录 follower 的本体/触觉相关信号。
3. 每个 episode 开始前抓取一张静态 RGB 图像作为视觉锚点。
4. 每个 episode 结束后，将视觉、触觉、元数据保存到与旧脚本兼容的目录结构中。

===============================================================================
二、与 collect_custom_multimodal.py 的关系
===============================================================================
1. 相同点：
   - 都会在每个 episode 中保存：
     - `visual_anchor.jpg`
     - `tactile_data.pkl`
     - `metadata.json`
   - 都会记录 follower 的：
     - 关节位置
     - 关节负载
     - 关节电流
   - 都会在后处理中计算速度信息（joint velocity / gripper velocity）
2. 不同点：
   - 原脚本通过读取 `position_logs.json`，按固定轨迹逐段回放动作。
   - 本脚本不读取任何动作日志，而是直接进入实时遥操作循环。
   - 原脚本的 episode 边界由回放流程中的 recording 段决定。
   - 本脚本的 episode 边界由用户手动定义：开始后按回车结束当前 episode。

===============================================================================
三、实现原理
===============================================================================
本脚本的实现分为四个主要子系统：

1. 相机子系统（VisualAnchorCamera）
   - 启动时打开指定摄像头。
   - 每次 episode 开始时抓取一张图像。
   - 若图像尺寸与目标尺寸不一致，会自动 resize。

2. 采样子系统（SensorSampler）
   - 在独立线程中按 `sample_hz` 对 follower 进行高频采样。
   - 采样内容包括：
     - `Present_Position`
     - `Present_Load`
     - `Present_Current`
   - 同时从这些原始量中提取：
     - gripper 宽度（由 gripper 关节位置近似表示）
     - gripper 负载
     - lift 关节电流
   - 采样线程只负责记录原始流，不负责控制机械臂。

3. 遥操作子系统（EpisodeCollector.collect 中的 teleop 循环）
   - 主线程循环调用 `robot.teleop_step(record_data=False)`。
   - 每次调用会：
     - 读取 leader 当前关节位置
     - 将其作为目标位置写入 follower
   - 这里使用 `record_data=False`，表示只执行控制，不走 LeRobot 的观测记录逻辑，
     从而保留更轻量的纯遥操作路径。

4. 结束监听子系统（StopListener）
   - 使用非阻塞的标准输入轮询，而不是后台线程 `input()` 抢占 stdin。
   - 在每轮遥操作后检测是否有用户按下回车。
   - 一旦检测到回车，就结束当前 episode。
   - 这种实现方式比双线程抢占输入更稳，避免和主流程的交互输入冲突。

===============================================================================
四、为什么要加总线锁（bus lock）
===============================================================================
采样线程和遥操作循环都会访问 follower 的串口总线：

1. 采样线程会读取 follower 的传感寄存器。
2. `robot.teleop_step()` 会向 follower 写入目标位置，且在某些配置下也可能读取当前位置。

如果两边同时访问同一条总线，真机上容易出现通信冲突或读写异常。
因此本脚本使用同一把 `threading.Lock` 来串行化：

- 采样线程读 follower 时加锁
- 遥操作循环调用 `teleop_step()` 时也加锁

这样做的结果是：
- 峰值控制频率会略低于“完全无锁”的理想情况
- 但硬件通信更稳定，更适合长时间采集

===============================================================================
五、每个 Episode 的完整流程
===============================================================================
1. 用户在主循环中按回车，开始一个新的 episode。
2. 脚本生成唯一的 `episode_id`。
3. 立即拍摄一张视觉锚点图像。
4. 启动高频采样线程，开始记录 follower 触觉/本体数据。
5. 进入实时遥操作循环：
   - 用户移动 leader
   - follower 跟随 leader 动作
   - 采样线程持续记录 follower 状态
6. 用户按回车，结束当前 episode。
7. 停止采样线程。
8. 对采样数据做后处理，计算速度。
9. 将数据写入 episode 目录。

===============================================================================
六、保存的数据内容
===============================================================================
1. `visual_anchor.jpg`
   - 当前 episode 开始前拍摄的单帧 RGB 图像。
   - 用于表示抓取/操作前的视觉上下文。

2. `tactile_data.pkl`
   - 保存一个 Python 字典（pickle 格式），其中包括：
     - `timestamps`
     - `joint_position_profile`
     - `joint_load_profile`
     - `joint_current_profile`
     - `gripper_width_profile`
     - `load_profile`
     - `lift_current_profile`
     - `joint_velocity_profile`
     - `gripper_velocity_profile`

3. `metadata.json`
   - 保存本次 episode 的元信息，包括：
     - `episode_id`
     - `label`
     - `batch_progress`
     - `real_sampling_rate`
     - `num_samples`
     - `duration_s`
     - `log_file`
     - `missed_segments`
     - `joint_names`
   - 其中 `log_file` 固定为 `null`，`missed_segments` 固定为空列表，
     这是为了保持与旧脚本的格式兼容。

===============================================================================
七、参数与默认硬件配置
===============================================================================
1. 默认串口配置：
   - leader: `/dev/ttyACM2`
   - follower: `/dev/ttyACM1`
   这是基于当前机器上已验证可用的主从映射。
2. 默认校准目录：
   - `/home/martina/Y3_Project/lerobot/.cache/calibration/so101`
3. 默认采样率：
   - `--sample-hz 200`
4. 默认不限制遥操作循环频率：
   - 遥操作循环会尽可能快运行
   - 采样线程独立按 `sample_hz` 工作

===============================================================================
八、设计边界
===============================================================================
1. 本脚本只记录 follower 的传感数据，不记录 leader 动作流。
2. 本脚本不保存 follower 目标角度时间序列。
3. 本脚本不负责回放，不负责训练，不负责数据清洗。
4. 它的职责仅限于：
   - 遥操作
   - 采样
   - 保存

===============================================================================
九、异常处理策略
===============================================================================
1. 如果相机打不开，脚本会在正式采集前退出。
2. 如果采样数据不足（少于 2 帧），当前 episode 会被丢弃。
3. 如果遥操作过程中发生异常，当前 episode 会提前结束：
   - 已采到的数据仍会尝试保留
   - 元数据仍会写出
   - 日志中会提示是异常提前结束
4. 脚本退出时会尝试关闭 leader / follower 力矩，并断开连接。
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import select
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np


def _ensure_lerobot_on_path() -> None:
    """Allow running directly from the monorepo without extra PYTHONPATH."""
    try:
        import lerobot  # noqa: F401
        return
    except ImportError:
        pass

    repo_root = Path(__file__).resolve().parent.parent
    lerobot_root = repo_root / "lerobot"
    if lerobot_root.exists():
        sys.path.insert(0, str(lerobot_root))


_ensure_lerobot_on_path()

from lerobot.common.robot_devices.robots.configs import So101RobotConfig  # noqa: E402
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot  # noqa: E402


LOGGER = logging.getLogger("collect_teleop_multimodal")
_COLOR_RESET = "\033[0m"
_COLOR_MAP = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[35m",
}


class VisualAnchorCamera:
    def __init__(
        self,
        index: int,
        width: int,
        height: int,
        warmup_frames: int,
        refresh_frames: int,
    ) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.refresh_frames = max(1, refresh_frames)
        self.cap: Optional[cv2.VideoCapture] = None
        self._open_camera(max(1, warmup_frames))

    def _open_camera(self, warmup_frames: int) -> None:
        cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {self.index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        for _ in range(warmup_frames):
            cap.read()
        self.cap = cap

    def capture(self) -> np.ndarray:
        if self.cap is None:
            raise RuntimeError("摄像头未初始化")
        frame = None
        for _ in range(self.refresh_frames):
            ret, frame = self.cap.read()
            if not ret:
                continue
        if frame is None or frame.size == 0:
            raise RuntimeError("摄像头未返回有效图像帧")
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        return frame

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class SensorSampler:
    def __init__(
        self,
        follower_bus,
        gripper_index: int,
        lift_index: int,
        sample_hz: float,
        bus_lock: threading.Lock,
    ) -> None:
        self.follower = follower_bus
        self.gripper_index = gripper_index
        self.lift_index = lift_index
        self.period = 0.0 if sample_hz <= 0 else 1.0 / sample_hz
        self._bus_lock = bus_lock
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._buffer_lock = threading.Lock()
        self._buffer = {
            "timestamps": [],
            "joint_position_profile": [],
            "joint_load_profile": [],
            "joint_current_profile": [],
            "gripper_width_profile": [],
            "load_profile": [],
            "lift_current_profile": [],
        }

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _run(self) -> None:
        start = time.perf_counter()
        next_tick = start
        while not self._stop.is_set():
            now = time.perf_counter()
            stamp = now - start
            try:
                with self._bus_lock:
                    pos = np.asarray(self.follower.read("Present_Position"), dtype=np.float32)
                    load_reader = getattr(self.follower, "read_decoded", self.follower.read)
                    curr_reader = getattr(self.follower, "read_decoded", self.follower.read)
                    load = np.asarray(load_reader("Present_Load"), dtype=np.float32)
                    current = np.asarray(curr_reader("Present_Current"), dtype=np.float32)
            except Exception as exc:
                LOGGER.warning("采样读取失败：%s", exc)
                time.sleep(self.period or 0.005)
                continue

            grip_val = float(pos[self.gripper_index]) if self.gripper_index < pos.size else float("nan")
            grip_load = float(load[self.gripper_index]) if self.gripper_index < load.size else float("nan")
            lift_curr = float(current[self.lift_index]) if self.lift_index < current.size else float("nan")

            with self._buffer_lock:
                self._buffer["timestamps"].append(float(stamp))
                self._buffer["joint_position_profile"].append(pos.astype(float).tolist())
                self._buffer["joint_load_profile"].append(load.astype(float).tolist())
                self._buffer["joint_current_profile"].append(current.astype(float).tolist())
                self._buffer["gripper_width_profile"].append(grip_val)
                self._buffer["load_profile"].append(grip_load)
                self._buffer["lift_current_profile"].append(lift_curr)

            if self.period <= 0:
                continue
            next_tick += self.period
            sleep_s = next_tick - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)

    def export(self) -> Dict[str, List[Any]]:
        with self._buffer_lock:
            return {k: list(v) for k, v in self._buffer.items()}


class StopListener:
    """Use non-blocking stdin polling to end the current episode with Enter."""

    def __init__(self) -> None:
        self._stop_event = threading.Event()
        self._prompted = False

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event

    def start(self) -> None:
        self._stop_event.clear()
        if not self._prompted:
            print("采集中。按 Enter 结束当前 Episode...")
            self._prompted = True

    def poll(self) -> bool:
        if self._stop_event.is_set():
            return True
        if not self._stdin_ready():
            return False
        try:
            sys.stdin.readline()
        except Exception:
            return False
        self._stop_event.set()
        return True

    def stop(self) -> None:
        self._stop_event.set()

    @staticmethod
    def _stdin_ready() -> bool:
        try:
            readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        except (ValueError, OSError):
            return False
        return bool(readable)


@dataclass
class EpisodeContext:
    label: str
    batch_progress: str
    episode_id: str


class EpisodeCollector:
    def __init__(
        self,
        robot: ManipulatorRobot,
        follower,
        motor_names: Sequence[str],
        args: argparse.Namespace,
        camera: VisualAnchorCamera,
        dataset_root: Path,
    ) -> None:
        self.robot = robot
        self.follower = follower
        self.motor_names = list(motor_names)
        self.args = args
        self.camera = camera
        self.dataset_root = dataset_root
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        self._bus_lock = threading.Lock()
        self.gripper_index = self._require_index(args.gripper_name)
        self.lift_index = self._require_index(args.lift_name)

    def _require_index(self, joint_name: str) -> int:
        if joint_name not in self.motor_names:
            raise ValueError(f"电机列表中找不到 {joint_name}，可用：{self.motor_names}")
        return self.motor_names.index(joint_name)

    def collect(self, ctx: EpisodeContext) -> Optional[Path]:
        LOGGER.info("开始采集 Episode %s，标签 [%s]", ctx.episode_id, ctx.label)
        LOGGER.info("捕获视觉锚点中...")
        anchor = self.camera.capture()
        LOGGER.info("视觉锚点捕获完成，分辨率 %dx%d", anchor.shape[1], anchor.shape[0])

        sampler = SensorSampler(
            self.follower,
            self.gripper_index,
            self.lift_index,
            sample_hz=self.args.sample_hz,
            bus_lock=self._bus_lock,
        )
        stop_listener = StopListener()
        teleop_error: Optional[Exception] = None

        sampler.start()
        stop_listener.start()
        try:
            while not stop_listener.stop_event.is_set():
                try:
                    with self._bus_lock:
                        self.robot.teleop_step(record_data=False)
                except Exception as exc:
                    teleop_error = exc
                    LOGGER.error("遥操作过程中发生异常，提前结束当前 Episode：%s", exc)
                    break
                stop_listener.poll()
        finally:
            stop_listener.stop()
            sampler.stop()

        buffer = sampler.export()
        if len(buffer["timestamps"]) < 2:
            LOGGER.error("采样数据不足，丢弃本次采集。")
            return None

        derived = self._post_process(buffer)
        return self._persist(ctx, derived, anchor, teleop_error)

    def _post_process(self, buffer: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        timestamps = np.asarray(buffer["timestamps"], dtype=np.float64)
        joint_positions = np.asarray(buffer["joint_position_profile"], dtype=np.float64)
        if joint_positions.ndim != 2:
            raise ValueError("joint_position_profile 数据维度不正确。")

        frames, joints = joint_positions.shape
        vel_window = max(1, int(self.args.vel_window))
        velocities = np.zeros_like(joint_positions)
        if frames >= vel_window + 1:
            dt = (timestamps[vel_window:] - timestamps[:-vel_window]).reshape(-1, 1)
            dt[dt == 0] = np.finfo(np.float64).eps
            diffs = joint_positions[vel_window:] - joint_positions[:-vel_window]
            velocities[vel_window:] = diffs / dt

        smooth_window = max(1, int(self.args.vel_smooth))
        velocities = _moving_average(velocities, smooth_window)
        buffer["joint_velocity_profile"] = velocities.tolist()
        if self.gripper_index < joints:
            buffer["gripper_velocity_profile"] = velocities[:, self.gripper_index].tolist()
        else:
            buffer["gripper_velocity_profile"] = [0.0] * frames
        return buffer

    def _persist(
        self,
        ctx: EpisodeContext,
        tactile_buffer: Dict[str, List[Any]],
        anchor_image: np.ndarray,
        teleop_error: Optional[Exception],
    ) -> Path:
        timestamps = tactile_buffer["timestamps"]
        duration = timestamps[-1] - timestamps[0] if len(timestamps) >= 2 else 0.0
        samples = len(timestamps)
        real_fs = float(samples / duration) if duration > 0 else 0.0
        LOGGER.info(
            "本次触觉采样共 %d 帧，持续 %.2fs，估算采样率 %.1f Hz",
            samples,
            duration,
            real_fs,
        )

        episode_dir = self.dataset_root / ctx.episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)

        tactile_path = episode_dir / "tactile_data.pkl"
        with tactile_path.open("wb") as file_obj:
            pickle.dump(tactile_buffer, file_obj, protocol=pickle.HIGHEST_PROTOCOL)
        LOGGER.info("已保存触觉数据：%s", tactile_path)

        anchor_path = episode_dir / "visual_anchor.jpg"
        if not cv2.imwrite(str(anchor_path), anchor_image):
            raise RuntimeError(f"无法保存图像 {anchor_path}")
        LOGGER.info("已保存视觉锚点：%s", anchor_path)

        metadata = {
            "episode_id": ctx.episode_id,
            "label": ctx.label,
            "batch_progress": ctx.batch_progress,
            "real_sampling_rate": real_fs,
            "num_samples": samples,
            "duration_s": duration,
            "log_file": None,
            "missed_segments": [],
            "joint_names": list(self.motor_names),
        }
        metadata_path = episode_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info("已保存元数据：%s", metadata_path)
        if teleop_error is not None:
            LOGGER.warning("本次 Episode 因遥操作异常提前结束，但采样数据已保留。")
        print(f"✅ 已保存 {ctx.episode_id} -> {episode_dir}")
        return episode_dir


def _moving_average(data: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return data
    kernel = np.ones(window, dtype=np.float64) / float(window)
    out = np.empty_like(data)
    for joint_idx in range(data.shape[1]):
        out[:, joint_idx] = np.convolve(data[:, joint_idx], kernel, mode="same")
    return out


def _configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        print(f"⚠️ 未知日志等级 {level}，将使用 INFO。")
        numeric_level = logging.INFO
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    class _ColorFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            color = _COLOR_MAP.get(record.levelname, "")
            reset = _COLOR_RESET if color else ""
            base = super().format(record)
            return f"{color}{base}{reset}"

    formatter = _ColorFormatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logging.basicConfig(level=numeric_level, handlers=[handler])
    LOGGER.setLevel(numeric_level)


def _prompt_target_count() -> Optional[int]:
    raw = input("请输入本批次计划采集的 Episode 数量 (默认无限): ").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        print("⚠️ 输入无法解析，将视为无限。")
        return None
    if value <= 0:
        print("⚠️ 输入 <= 0，将视为无限。")
        return None
    return value


def _scan_available_cameras(max_index: int = 10) -> List[Dict[str, Any]]:
    available: List[Dict[str, Any]] = []
    print(f"\n🔍 正在扫描可用摄像头 (索引 0-{max_index - 1})...")

    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            backend = cap.getBackendName()
            cap.release()
            available.append(
                {
                    "index": idx,
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "backend": backend,
                }
            )
    return available


def _select_camera_interactive(
    available_cameras: List[Dict[str, Any]],
    default_index: int,
) -> Optional[int]:
    if not available_cameras:
        print("❌ 没有检测到可用的摄像头！")
        return None

    print("\n" + "=" * 60)
    print("  📷 检测到以下可用摄像头:")
    print("=" * 60)

    default_in_list = None
    for i, cam in enumerate(available_cameras):
        marker = ""
        if cam["index"] == default_index:
            marker = " [默认]"
            default_in_list = i
        fps_str = f"{cam['fps']:.1f}" if cam["fps"] > 0 else "N/A"
        print(
            f"  [{i + 1}] 索引 {cam['index']}: "
            f"{cam['width']}x{cam['height']} @ {fps_str} FPS ({cam['backend']}){marker}"
        )

    print("=" * 60)
    while True:
        prompt = f"请选择摄像头编号 (1-{len(available_cameras)})"
        if default_in_list is not None:
            prompt += f", 直接回车使用默认 [{default_index}]"
        prompt += ", 或输入 'q' 退出: "
        user_input = input(prompt).strip().lower()

        if user_input == "q":
            return None
        if user_input == "":
            if default_in_list is not None:
                return int(available_cameras[default_in_list]["index"])
            print(f"⚠️ 默认索引 {default_index} 不可用，将使用第一个可用摄像头。")
            return int(available_cameras[0]["index"])
        try:
            choice = int(user_input)
        except ValueError:
            print("⚠️ 无效输入，请输入数字编号。")
            continue
        if 1 <= choice <= len(available_cameras):
            return int(available_cameras[choice - 1]["index"])
        print(f"⚠️ 请输入 1 到 {len(available_cameras)} 之间的数字。")


def _preview_camera_and_confirm(
    camera_index: int,
    width: int,
    height: int,
    warmup_frames: int = 10,
) -> bool:
    print(f"\n📷 正在打开摄像头 {camera_index} 进行预览...")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"❌ 无法打开摄像头 {camera_index}，请检查连接或尝试其他索引。")
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    for _ in range(warmup_frames):
        cap.read()

    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print("⚠️ 摄像头读取失败，请检查连接。")
        return False

    overlay = frame.copy()
    text_lines = [
        f"Camera Index: {camera_index}",
        f"Resolution: {frame.shape[1]}x{frame.shape[0]}",
    ]
    y_offset = 30
    for line in text_lines:
        cv2.putText(
            overlay,
            line,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y_offset += 30

    gui_available = True
    try:
        window_name = f"Camera {camera_index} Preview"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width, height)
    except cv2.error:
        gui_available = False

    if gui_available:
        print("=" * 60)
        print(f"  摄像头预览窗口已打开 (索引: {camera_index})")
        print("  请确认这是否是你要使用的摄像头。")
        print("  确认: 按 Enter / Y / 空格")
        print("  拒绝: 按 Q / N / Esc")
        print("=" * 60)

        confirmed = False
        try:
            while True:
                cv2.imshow(window_name, overlay)
                key = cv2.waitKey(100) & 0xFF
                if key in (13, ord("y"), ord("Y"), 32):
                    confirmed = True
                    print("✅ 用户确认使用该摄像头。")
                    break
                if key in (ord("q"), ord("Q"), ord("n"), ord("N"), 27):
                    print("❌ 用户拒绝使用该摄像头。")
                    break
        except KeyboardInterrupt:
            print("\n用户中断预览。")
            confirmed = False
        finally:
            cv2.destroyAllWindows()
            for _ in range(5):
                cv2.waitKey(1)
        return confirmed

    preview_path = Path("camera_preview.jpg")
    cv2.imwrite(str(preview_path), overlay)
    print("=" * 60)
    print("  ⚠️ OpenCV 没有 GUI 支持，已保存预览图片。")
    print(f"  📁 预览图片路径: {preview_path.absolute()}")
    print(f"  📷 摄像头索引: {camera_index}")
    print(f"  📐 分辨率: {frame.shape[1]}x{frame.shape[0]}")
    print("=" * 60)
    print("  请打开图片查看，然后在此确认。")

    while True:
        user_input = input("  这是正确的摄像头吗？(y/n): ").strip().lower()
        if user_input in ("y", "yes", ""):
            print("✅ 用户确认使用该摄像头。")
            try:
                preview_path.unlink()
            except Exception:
                pass
            return True
        if user_input in ("n", "no", "q"):
            print("❌ 用户拒绝使用该摄像头。")
            try:
                preview_path.unlink()
            except Exception:
                pass
            return False
        print("  请输入 y 或 n")


def _disable_all_torque(robot: ManipulatorRobot) -> None:
    try:
        for name, arm in robot.follower_arms.items():
            LOGGER.info("禁用 follower arm %s 的力矩。", name)
            arm.write("Torque_Enable", 0)
        for name, arm in robot.leader_arms.items():
            LOGGER.info("禁用 leader arm %s 的力矩。", name)
            arm.write("Torque_Enable", 0)
    except Exception as exc:
        LOGGER.warning("禁用力矩时发生异常：%s", exc)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    default_calibration_dir = repo_root / "lerobot" / ".cache" / "calibration" / "so101"

    parser = argparse.ArgumentParser(description="基于遥操作的多模态数据采集脚本")
    parser.add_argument("--sample-hz", type=float, default=200.0, help="传感器采样频率 (Hz)")
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=default_calibration_dir,
        help="LeRobot 校准目录",
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("Plaintextdataset"), help="输出数据集根目录")
    parser.add_argument("--default-label", type=str, default="unlabeled", help="初始标签")
    parser.add_argument("--camera-index", type=int, default=0, help="摄像头索引")
    parser.add_argument("--camera-width", type=int, default=640, help="视觉锚点宽度")
    parser.add_argument("--camera-height", type=int, default=480, help="视觉锚点高度")
    parser.add_argument("--camera-warmup", type=int, default=5, help="采集前丢弃的预热帧数")
    parser.add_argument("--camera-refresh", type=int, default=3, help="每次采集前丢弃的帧数以刷新画面")
    parser.add_argument("--gripper-name", type=str, default="gripper", help="对应 gripper 的电机名称")
    parser.add_argument("--lift-name", type=str, default="shoulder_lift", help="Lift 对应电机名称")
    parser.add_argument("--log-level", type=str, default="INFO", help="日志等级 (DEBUG/INFO/WARNING/ERROR)")
    parser.add_argument("--vel-window", type=int, default=5, help="速度求导窗口（帧数，>=1）")
    parser.add_argument("--vel-smooth", type=int, default=10, help="速度移动平均窗口（帧数，>=1）")
    parser.add_argument("--leader-port", type=str, default="/dev/ttyACM2", help="leader 串口")
    parser.add_argument("--follower-port", type=str, default="/dev/ttyACM1", help="follower 串口")
    parser.add_argument(
        "--max-relative-target",
        type=int,
        default=None,
        help="可选安全限制，原样传递给 So101RobotConfig",
    )
    return parser.parse_args()


def build_robot(args: argparse.Namespace) -> ManipulatorRobot:
    config = So101RobotConfig(
        calibration_dir=str(args.calibration_dir),
        max_relative_target=args.max_relative_target,
        cameras={},
    )
    config.leader_arms["main"].port = args.leader_port
    config.follower_arms["main"].port = args.follower_port
    return ManipulatorRobot(config)


def main() -> int:
    args = parse_args()
    _configure_logging(args.log_level)

    available_cameras = _scan_available_cameras(max_index=10)
    selected_camera_index = _select_camera_interactive(
        available_cameras=available_cameras,
        default_index=args.camera_index,
    )
    if selected_camera_index is None:
        raise SystemExit("未选择摄像头，退出。")

    if not _preview_camera_and_confirm(
        camera_index=selected_camera_index,
        width=args.camera_width,
        height=args.camera_height,
        warmup_frames=args.camera_warmup,
    ):
        raise SystemExit("用户未确认摄像头，退出。")
    args.camera_index = selected_camera_index

    target_count = _prompt_target_count()

    robot: Optional[ManipulatorRobot] = None
    camera: Optional[VisualAnchorCamera] = None
    current_label = args.default_label
    current_count = 0
    try:
        robot = build_robot(args)
        robot.connect()

        follower_name = next(iter(robot.follower_arms.keys()))
        follower = robot.follower_arms[follower_name]
        motor_names = list(follower.motor_names)

        camera = VisualAnchorCamera(
            index=args.camera_index,
            width=args.camera_width,
            height=args.camera_height,
            warmup_frames=args.camera_warmup,
            refresh_frames=args.camera_refresh,
        )
        collector = EpisodeCollector(
            robot=robot,
            follower=follower,
            motor_names=motor_names,
            args=args,
            camera=camera,
            dataset_root=args.dataset_root,
        )

        goal_label = str(target_count) if target_count is not None else "∞"
        while True:
            if target_count is not None and current_count >= target_count:
                print(f"🎯 已达到目标 {target_count} 条 Episode，自动结束。")
                break
            print(f"\n[进度: {current_count + 1}/{goal_label}]")
            print(f"当前标签: [{current_label}]")
            user = input("按 Enter 开始，输入新标签修改，或 'q' 退出: ").strip()
            if user.lower() == "q":
                print("用户选择退出。")
                break
            if user:
                current_label = user

            episode_id = f"episode_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            batch_progress = f"{current_count + 1}/{goal_label}"
            ctx = EpisodeContext(label=current_label, batch_progress=batch_progress, episode_id=episode_id)
            saved_dir = collector.collect(ctx)
            if saved_dir is not None:
                current_count += 1
    except KeyboardInterrupt:
        print("\n收到中断信号，准备退出...")
    finally:
        if camera is not None:
            camera.close()
        if robot is not None:
            try:
                _disable_all_torque(robot)
            except Exception:
                pass
            try:
                robot.disconnect()
            except Exception:
                pass

    print(f"👋 结束采集，本次共完成 {current_count} 条 Episode。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
