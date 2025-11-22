#!/usr/bin/env python3
"""
collect_custom_multimodal.py
============================

Human-in-the-loop dataset recorder that replays logged arm motions, captures a visual
anchor, and stores high-frequency tactile streams with derived velocity.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from lerobot.common.robot_devices.robots.configs import So101RobotConfig
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot


LOGGER = logging.getLogger("collect_custom_multimodal")
_COLOR_RESET = "\033[0m"
_COLOR_MAP = {
    "DEBUG": "\033[36m",   # Cyan
    "INFO": "\033[32m",    # Green
    "WARNING": "\033[33m", # Yellow
    "ERROR": "\033[31m",   # Red
    "CRITICAL": "\033[35m",
}


# ----------------------
# Generic helper blocks
# ----------------------


def _percent_to_goal_speed_units(percent: float, max_units: int = 1023) -> int:
    pct = float(np.clip(percent, 0.0, 100.0))
    return int(round(pct / 100.0 * max_units))


def _as_array(value):
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        return np.asarray(value)
    try:
        return np.asarray([value])
    except Exception:
        return np.asarray([-1])


def _set_speed_percent(
    follower_bus,
    speed_percent: float,
    accel: int,
    retries: int = 2,
    verify_sleep_s: float = 0.02,
    tolerance_units: int = 3,
) -> bool:
    goal_speed = _percent_to_goal_speed_units(speed_percent)
    for attempt in range(retries + 1):
        follower_bus.write("Acceleration", int(accel))
        follower_bus.write("Goal_Speed", int(goal_speed))
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
    print(
        "⚠️ Goal_Speed/Acceleration 写入后读回不一致，仍将尝试继续执行 "
        f"(exp accel={accel}, exp speed={goal_speed})"
    )
    return False


def _ensure_joint_vector(
    joint_dict: Dict[str, float],
    motor_names: Sequence[str],
    fill_value: float = 0.0,
) -> np.ndarray:
    vec = []
    missing = []
    for name in motor_names:
        if name in joint_dict:
            vec.append(float(joint_dict[name]))
        else:
            vec.append(float(fill_value))
            missing.append(name)
    if missing:
        print(f"⚠️ 日志缺少以下关节字段，将使用 {fill_value}° 代替：{missing}")
    return np.asarray(vec, dtype=np.float32)


def _set_integral_gains(follower, motor_names: Sequence[str], value: int = 2) -> None:
    applied = 0
    for name in motor_names:
        if "gripper" in name.lower():
            continue
        try:
            follower.write("I_Coefficient", int(value), name)
            applied += 1
        except Exception as exc:
            LOGGER.warning("设置 I_Coefficient 到 %s 失败（关节 %s）：%s", value, name, exc)
    if applied:
        LOGGER.info("已将 %d 个非 gripper 关节的 I_Coefficient 设为 %d。", applied, value)
    else:
        LOGGER.warning("没有任何关节成功更新 I_Coefficient。")


def _load_logs(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError(f"{path} 内容不是 JSON 数组")
    return data


def _wait_until_reached(
    follower_bus,
    goal_deg: np.ndarray,
    tol_deg: float,
    timeout_s: float,
    poll_hz: float,
    bus_lock: Optional[threading.Lock] = None,
) -> tuple[bool, float]:
    start = time.perf_counter()
    period = 1.0 / poll_hz
    last_err = float("inf")
    while time.perf_counter() - start <= timeout_s:
        try:
            if bus_lock is None:
                present = np.asarray(follower_bus.read("Present_Position"), dtype=np.float32)
            else:
                with bus_lock:
                    present = np.asarray(follower_bus.read("Present_Position"), dtype=np.float32)
        except Exception as exc:
            print(f"⚠️ 读取 Present_Position 失败：{exc}")
            break
        err = np.abs(present - goal_deg[: present.size])
        last_err = float(np.max(err)) if err.size else 0.0
        if np.all(err <= tol_deg):
            return True, last_err
        time.sleep(period)
    return False, last_err


def _moving_average(data: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return data
    kernel = np.ones(window, dtype=np.float64) / float(window)
    out = np.empty_like(data)
    for j in range(data.shape[1]):
        out[:, j] = np.convolve(data[:, j], kernel, mode="same")
    return out


# --------------------
# Camera infrastructure
# --------------------


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


# --------------------
# Sampling subsystem
# --------------------


class SensorSampler:
    def __init__(
        self,
        follower_bus,
        gripper_index: int,
        lift_index: int,
        sample_hz: float,
        bus_lock: threading.Lock,
        motor_names: Sequence[str],
    ) -> None:
        self.follower = follower_bus
        self.gripper_index = gripper_index
        self.lift_index = lift_index
        self.period = 0.0 if sample_hz <= 0 else 1.0 / sample_hz
        self._bus_lock = bus_lock
        self.motor_names = list(motor_names)
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
                print(f"⚠️ 采样读取失败：{exc}")
                time.sleep(self.period or 0.005)
                continue
            pos_list = pos.astype(float).tolist()
            load_list = load.astype(float).tolist()
            curr_list = current.astype(float).tolist()
            grip_val = float(pos[self.gripper_index]) if self.gripper_index < pos.size else float("nan")
            grip_load = float(load[self.gripper_index]) if self.gripper_index < load.size else float("nan")
            lift_curr = float(current[self.lift_index]) if self.lift_index < current.size else float("nan")
            with self._buffer_lock:
                self._buffer["timestamps"].append(float(stamp))
                self._buffer["joint_position_profile"].append(pos_list)
                self._buffer["joint_load_profile"].append(load_list)
                self._buffer["joint_current_profile"].append(curr_list)
                self._buffer["gripper_width_profile"].append(grip_val)
                self._buffer["load_profile"].append(grip_load)
                self._buffer["lift_current_profile"].append(lift_curr)
            if self.period <= 0:
                continue
            next_tick += self.period
            sleep_s = next_tick - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)

    def export(self) -> Dict[str, List[float]]:
        with self._buffer_lock:
            return {k: list(v) for k, v in self._buffer.items()}


# --------------------
# Episode level orchestration
# --------------------


@dataclass
class EpisodeContext:
    label: str
    batch_progress: str
    episode_id: str


class EpisodeCollector:
    def __init__(
        self,
        follower,
        motor_names: Sequence[str],
        log_entries: List[dict],
        args: argparse.Namespace,
        camera: VisualAnchorCamera,
        dataset_root: Path,
    ) -> None:
        self.follower = follower
        self.motor_names = motor_names
        self.log_entries = log_entries
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
            motor_names=self.motor_names,
        )
        missed_segments, recorded = self._replay_sequence(sampler)
        sampler.stop()
        if not recorded:
            LOGGER.warning("recording 段未触发采样，本次不会生成数据。")
        if missed_segments:
            LOGGER.warning(
                "本次复刻有 %d 段未到位：%s（但仍将保留采集数据）",
                len(missed_segments),
                missed_segments,
            )
        buffer = sampler.export()
        if not recorded or len(buffer["timestamps"]) < 2:
            LOGGER.error("采样数据不足，丢弃本次采集。")
            return None
        derived = self._post_process(buffer)
        episode_dir = self._persist(ctx, derived, anchor, missed_segments)
        return episode_dir

    def _apply_speed_mode(self, mode: str, current_mode: Optional[str]) -> str:
        mode = "slow" if mode not in {"fast", "slow"} else mode
        if current_mode == mode:
            return current_mode
        if mode == "fast":
            percent = self.args.fast_speed_percent
            accel = self.args.fast_accel
        else:
            percent = self.args.speed_percent
            accel = self.args.accel
        LOGGER.info(
            "切换速度模式 -> %s (speed=%.1f%%, accel=%d)",
            mode,
            percent,
            accel,
        )
        with self._bus_lock:
            _set_speed_percent(
                self.follower,
                speed_percent=percent,
                accel=accel,
                retries=self.args.speed_retries,
                verify_sleep_s=self.args.speed_verify_sleep,
            )
        if self.args.post_speed_sleep > 0:
            time.sleep(self.args.post_speed_sleep)
        return mode

    def _replay_sequence(self, sampler: SensorSampler) -> tuple[List[int], bool]:
        poll_hz = 20.0
        missed: List[int] = []
        record_indices = [
            idx for idx, entry in enumerate(self.log_entries) if bool(entry.get("recording", False))
        ]
        if not record_indices:
            LOGGER.warning("日志中 recording 标志未开启，本次将不会记录触觉数据。")
        sampler_started = False
        recording_active = False
        current_speed_mode: Optional[str] = None
        for idx, entry in enumerate(self.log_entries):
            joint_dict = entry.get("joint_angles")
            if not isinstance(joint_dict, dict):
                LOGGER.warning("条目 #%d joint_angles 缺失或格式错误，跳过。", idx)
                missed.append(idx)
                continue
            goal_deg = _ensure_joint_vector(joint_dict, self.motor_names)
            requested_mode = str(entry.get("speed", "slow")).lower()
            current_speed_mode = self._apply_speed_mode(requested_mode, current_speed_mode)
            speed_mode = current_speed_mode or "slow"
            entry_recording = bool(entry.get("recording", False))
            if entry_recording and not recording_active:
                LOGGER.info(
                    "触觉采样开始（recording 段 #%d，目标频率 %.1f Hz）。",
                    idx,
                    self.args.sample_hz,
                )
                sampler.start()
                sampler_started = True
                recording_active = True
            LOGGER.info(
                "执行日志条目 #%d（mode=%s），目标角度=%s",
                idx,
                speed_mode,
                np.array2string(goal_deg, precision=2),
            )
            with self._bus_lock:
                self.follower.write("Goal_Position", goal_deg)
            reached, err_val = _wait_until_reached(
                self.follower,
                goal_deg,
                tol_deg=self.args.tol_deg,
                timeout_s=self.args.timeout,
                poll_hz=poll_hz,
                bus_lock=self._bus_lock,
            )
            if reached:
                LOGGER.info("条目 #%d 到位（最大误差 %.2f°）。", idx, err_val)
            else:
                LOGGER.warning(
                    "条目 #%d 超时未到位（最大误差 %.2f°，tol=%.2f°, timeout=%.1fs）。",
                    idx,
                    err_val,
                    self.args.tol_deg,
                    self.args.timeout,
                )
                missed.append(idx)
            next_recording = False
            if idx + 1 < len(self.log_entries):
                next_recording = bool(self.log_entries[idx + 1].get("recording", False))
            if recording_active and not next_recording:
                LOGGER.info("触觉采样结束（完成 recording 段 #%d）。", idx)
                sampler.stop()
                recording_active = False
        if recording_active:
            LOGGER.info("触觉采样在循环结束时仍在运行，正在强制停止。")
            sampler.stop()
            recording_active = False
        return missed, sampler_started

    def _post_process(self, buffer: Dict[str, List[float]]) -> Dict[str, List[float]]:
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
        tactile_buffer: Dict[str, List[float]],
        anchor_image: np.ndarray,
        missed_segments: Sequence[int],
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
        with tactile_path.open("wb") as f:
            pickle.dump(tactile_buffer, f, protocol=pickle.HIGHEST_PROTOCOL)
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
            "log_file": str(self.args.log_file),
            "missed_segments": list(missed_segments),
            "joint_names": list(self.motor_names),
        }
        metadata_path = episode_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info("已保存元数据：%s", metadata_path)
        print(f"✅ 已保存 {ctx.episode_id} -> {episode_dir}")
        return episode_dir


# --------------------
# CLI glue
# --------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="自定义多模态数据采集脚本")
    default_log = Path(__file__).parent / "outputs" / "logs" / "position_logs.json"
    parser.add_argument("--log-file", type=Path, default=default_log, help="动作日志 JSON 路径")
    parser.add_argument("--sample-hz", type=float, default=200.0, help="传感器采样频率 (Hz)")
    parser.add_argument("--tol-deg", type=float, default=2.0, help="到位角度公差")
    parser.add_argument("--timeout", type=float, default=3.0, help="单段动作超时时间 (s)")
    parser.add_argument("--speed-percent", type=float, default=20.0, help="速度百分比 (0-100)")
    parser.add_argument("--accel", type=int, default=60, help="Acceleration 寄存器值")
    parser.add_argument("--speed-retries", type=int, default=2, help="速度写入重试次数")
    parser.add_argument("--speed-verify-sleep", type=float, default=0.05, help="写入速度后等待 (s)")
    parser.add_argument("--post-speed-sleep", type=float, default=0.05, help="速度设置完成到动作开始的等待 (s)")
    parser.add_argument("--fast-speed-percent", type=float, default=100.0, help="fast 段速度百分比")
    parser.add_argument("--fast-accel", type=int, default=254, help="fast 段 Acceleration 值")
    parser.add_argument("--calibration-dir", type=Path, default=Path(__file__).parent, help="标定目录")
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
    return parser.parse_args()


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
        if value <= 0:
            print("⚠️ 输入 <= 0，将视为无限。")
            return None
        return value
    except ValueError:
        print("⚠️ 输入无法解析，将视为无限。")
        return None


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


def main() -> int:
    args = parse_args()
    _configure_logging(args.log_level)
    target_count = _prompt_target_count()
    if not args.log_file.exists():
        raise SystemExit(f"日志文件不存在：{args.log_file}")

    log_entries = _load_logs(args.log_file)
    if not log_entries:
        raise SystemExit("日志文件为空，无法执行。")

    robot: Optional[ManipulatorRobot] = None
    camera: Optional[VisualAnchorCamera] = None
    current_label = args.default_label
    current_count = 0
    try:
        robot = ManipulatorRobot(
            So101RobotConfig(
                calibration_dir=str(args.calibration_dir),
                leader_arms={},
                cameras={},
            )
        )
        robot.connect()
        follower_name = next(iter(robot.follower_arms.keys()))
        follower = robot.follower_arms[follower_name]
        motor_names = list(follower.motor_names)

        _set_integral_gains(follower, motor_names, value=2)
        _set_speed_percent(
            follower,
            speed_percent=args.speed_percent,
            accel=args.accel,
            retries=args.speed_retries,
            verify_sleep_s=args.speed_verify_sleep,
        )
        if args.post_speed_sleep > 0:
            time.sleep(args.post_speed_sleep)

        camera = VisualAnchorCamera(
            index=args.camera_index,
            width=args.camera_width,
            height=args.camera_height,
            warmup_frames=args.camera_warmup,
            refresh_frames=args.camera_refresh,
        )
        collector = EpisodeCollector(
            follower=follower,
            motor_names=motor_names,
            log_entries=log_entries,
            args=args,
            camera=camera,
            dataset_root=args.dataset_root,
        )
        goal_label = str(target_count) if target_count is not None else "∞"
        print(f"动作日志已加载，共 {len(log_entries)} 条记录。")
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
