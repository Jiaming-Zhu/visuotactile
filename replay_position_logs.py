"""
顺序复刻 position_logs.json 中记录的关节动作。

特性：
- 默认读取 learn_PyBullet/outputs/logs/position_logs.json。
- 使用与 collect_pressing_dataset.py 相同的 Goal_Speed/Acceleration 写法限制最大速度。
- 提供两种模式：
  * debug  : 每条记录执行前按 Enter 确认，可输入 exit/quit 中止。
  * collect: 自动连续执行所有记录。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Optional
import time
import socket
import subprocess
import sys
import threading

import numpy as np

from lerobot.common.robot_devices.robots.configs import So101RobotConfig
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot


_TELEMETRY_LOCK = threading.Lock()
_TELEMETRY_PACKET: Optional[dict] = None


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
    """复制 collect_pressing_dataset 中的速度写法，保持安全的最大速度。"""
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


def _raw_speed_to_deg(raw_words, follower_bus, motor_names: Sequence[str]) -> np.ndarray:
    raw_arr = np.asarray(raw_words, dtype=np.int32)
    neg = (raw_arr & 0x8000) != 0
    mag = (raw_arr & 0x7FFF).astype(np.int32)
    signed_counts = np.where(neg, -mag, mag).astype(np.float32)
    try:
        res_list = []
        for name in motor_names:
            _, model = follower_bus.motors[name]
            resolution = getattr(follower_bus, "model_resolution", {}).get(model, 4096)
            res_list.append(float(resolution))
        res = np.asarray(res_list, dtype=np.float32)
    except Exception:
        res = np.full_like(signed_counts, 4096.0, dtype=np.float32)
    if res.size < signed_counts.size:
        res = np.pad(res, (0, signed_counts.size - res.size), constant_values=4096.0)
    return signed_counts / res[: signed_counts.size] * 360.0



def _wait_until_reached(
    follower_bus,
    goal_deg: np.ndarray,
    tol_deg: float,
    timeout_s: float,
    poll_hz: float,
    motor_names: Sequence[str],
) -> bool:
    start = time.perf_counter()
    period = 1.0 / poll_hz
    prev_pos_arr: Optional[np.ndarray] = None
    prev_t: Optional[float] = None
    while time.perf_counter() - start <= timeout_s:
        try:
            present = np.asarray(follower_bus.read("Present_Position"), dtype=np.float32)
        except Exception as exc:
            print(f"⚠️ 读取 Present_Position 失败：{exc}")
            return False
        try:
            if hasattr(follower_bus, "read_decoded"):
                load = np.asarray(follower_bus.read_decoded("Present_Load"), dtype=np.float32)
            else:
                load = np.asarray(follower_bus.read("Present_Load"), dtype=np.float32)
        except Exception:
            load = None
        try:
            if hasattr(follower_bus, "read_decoded"):
                current = np.asarray(follower_bus.read_decoded("Present_Current"), dtype=np.float32)
            else:
                current = np.asarray(follower_bus.read("Present_Current"), dtype=np.float32)
        except Exception:
            current = None
        speed = None
        speed_derived = None
        try:
            speed_words = follower_bus.read("Present_Speed")
            speed = _raw_speed_to_deg(speed_words, follower_bus, motor_names)
        except Exception:
            speed = None
        now_t = time.perf_counter()
        if prev_pos_arr is not None and prev_t is not None:
            dt = now_t - prev_t
            if dt > 1e-6:
                speed_derived = (present - prev_pos_arr) / dt
        prev_pos_arr = present.copy()
        prev_t = now_t
        if speed_derived is None and present.size:
            speed_derived = np.zeros_like(present)
        packet = {
            "load": list(map(float, load)) if load is not None else [],
            "current": list(map(float, current)) if current is not None else [],
            "position": list(map(float, present)),
            "speed": list(map(float, speed)) if speed is not None else [],
            "speed_derived": list(map(float, speed_derived)) if speed_derived is not None else [],
        }
        with _TELEMETRY_LOCK:
            global _TELEMETRY_PACKET
            _TELEMETRY_PACKET = packet
        err = np.abs(present - goal_deg[: present.size])
        if np.all(err <= tol_deg):
            return True
        time.sleep(period)
    return False


def _load_logs(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError(f"{path} 内容不是 JSON 数组")
    return data


def _telemetry_server(stop_event: threading.Event, motor_names: Sequence[str], port: int) -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)
    try:
        while not stop_event.is_set():
            srv.settimeout(0.5)
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            with conn:
                print(f"[metrics] 客户端已连接 - 127.0.0.1:{port}")
                f = conn.makefile("w")
                try:
                    f.write(json.dumps({"names": list(motor_names)}) + "\n")
                    f.flush()
                    while not stop_event.is_set():
                        with _TELEMETRY_LOCK:
                            pkt = _TELEMETRY_PACKET or {"load": [], "current": [], "position": [], "speed": []}
                        f.write(json.dumps(pkt) + "\n")
                        f.flush()
                        if stop_event.wait(0.1):
                            break
                except Exception:
                    pass
    finally:
        try:
            srv.close()
        except Exception:
            pass


def _launch_viewer(port: int, interval: float, window: float) -> Optional[subprocess.Popen]:
    viewer_path = Path(__file__).with_name("realtime_metrics_viewer.py")
    cmd = [
        sys.executable,
        str(viewer_path),
        "--mode",
        "tcp",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--interval",
        str(interval),
        "--window",
        str(window),
    ]
    try:
        return subprocess.Popen(cmd)
    except Exception as exc:
        print(f"⚠️ 无法启动实时 viewer：{exc}")
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="根据 position_logs.json 复刻机械臂动作")
    default_log = Path(__file__).parent / "outputs" / "logs" / "position_logs.json"
    parser.add_argument("--log-file", type=Path, default=default_log, help="日志 JSON 路径")
    parser.add_argument(
        "--mode",
        choices={"debug", "collect"},
        default="debug",
        help="debug: 每条动作需按 Enter 确认；collect: 自动执行全部",
    )
    parser.add_argument("--start-index", type=int, default=0, help="从第几个条目开始（0-based）")
    parser.add_argument("--tol-deg", type=float, default=2.0, help="到位允许误差（度）")
    parser.add_argument("--timeout", type=float, default=5.0, help="单段最大等待时间（秒）")
    parser.add_argument("--speed-percent", type=float, default=20.0, help="最大速度百分比 (0-100)")
    parser.add_argument("--accel", type=int, default=60, help="Acceleration 寄存器值")
    parser.add_argument("--speed-retries", type=int, default=2, help="速度写入重试次数")
    parser.add_argument(
        "--speed-verify-sleep",
        type=float,
        default=0.05,
        help="写入 Goal_Speed 后等待的秒数（用于读回校验）",
    )
    parser.add_argument(
        "--post-speed-sleep",
        type=float,
        default=0.05,
        help="速度设置成功后到发位姿之间的等待时间",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path(__file__).parent,
        help="LeRobot 标定目录（默认脚本所在目录）",
    )
    parser.add_argument(
        "--viewer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启动实时 viewer（默认开启，可用 --no-viewer 关闭）",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=8766,
        help="遥测 TCP 服务器端口（与 viewer 通信）",
    )
    parser.add_argument(
        "--metrics-interval",
        type=float,
        default=0.1,
        help="viewer 刷新间隔（秒）",
    )
    parser.add_argument(
        "--metrics-window",
        type=float,
        default=20.0,
        help="viewer 窗口长度（秒）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.log_file.exists():
        raise SystemExit(f"日志文件不存在：{args.log_file}")

    entries = _load_logs(args.log_file)
    if args.start_index >= len(entries):
        raise SystemExit(f"start-index={args.start_index} 超出日志条目数 {len(entries)}")

    print(f"共加载 {len(entries)} 条日志，将从第 {args.start_index} 条开始执行。")
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

    telemetry_thread: Optional[threading.Thread] = None
    telemetry_stop: Optional[threading.Event] = None
    metrics_proc: Optional[subprocess.Popen] = None
    if args.viewer:
        telemetry_stop = threading.Event()
        telemetry_thread = threading.Thread(
            target=_telemetry_server,
            args=(telemetry_stop, motor_names, args.metrics_port),
            daemon=True,
        )
        telemetry_thread.start()
        metrics_proc = _launch_viewer(args.metrics_port, args.metrics_interval, args.metrics_window)

    try:
        orig_accel = np.asarray(follower.read("Acceleration")).copy()
    except Exception:
        orig_accel = None
    try:
        orig_speed = np.asarray(follower.read("Goal_Speed")).copy()
    except Exception:
        orig_speed = None
    try:
        orig_i_coeff = np.asarray(follower.read("I_Coefficient")).copy()
    except Exception:
        orig_i_coeff = None

    def _restore_i_coeff():
        if orig_i_coeff is None:
            return
        try:
            for name, val in zip(motor_names, orig_i_coeff):
                follower.write("I_Coefficient", int(val), name)
        except Exception as exc:
            print(f"⚠️ 恢复 I_Coefficient 失败: {exc}")

    try:
        try:
            count = 0
            for name in motor_names:
                if "gripper" in name.lower():
                    continue
                follower.write("I_Coefficient", 2, name)
                count += 1
            if count:
                print(f"已将 {count} 个关节的 I_Coefficient 设置为 2（排除了 gripper）")
        except Exception as exc:
            print(f"⚠️ 设置 I_Coefficient 失败：{exc}")
        _set_speed_percent(
            follower,
            speed_percent=args.speed_percent,
            accel=args.accel,
            retries=args.speed_retries,
            verify_sleep_s=args.speed_verify_sleep,
        )
        if args.post_speed_sleep > 0:
            time.sleep(args.post_speed_sleep)
        total = len(entries) - args.start_index
        for idx in range(args.start_index, len(entries)):
            entry = entries[idx]
            ts = entry.get("timestamp", "unknown")
            source = entry.get("source", "unknown")
            joint_dict = entry.get("joint_angles")
            if not isinstance(joint_dict, dict):
                print(f"⚠️ 条目 #{idx} joint_angles 缺失或不是对象，跳过")
                continue
            goal_deg = _ensure_joint_vector(joint_dict, motor_names)
            print(f"\n[{idx+1}/{len(entries)}] {ts} (source={source})")
            print(f"  目标角度: {np.array2string(goal_deg, precision=2)}")
            if args.mode == "debug":
                user = input("  按 Enter 执行，输入 exit/quit 结束: ").strip().lower()
                if user in {"exit", "quit"}:
                    print("用户取消，退出。")
                    break
            follower.write("Goal_Position", goal_deg)
            reached = _wait_until_reached(
                follower,
                goal_deg,
                tol_deg=args.tol_deg,
                timeout_s=args.timeout,
                poll_hz=20.0,
                motor_names=motor_names,
            )
            if reached:
                print("  ✓ 到位")
            else:
                print("  ⚠️ 未在超时时间内到位，继续下一条")
            if args.mode == "collect":
                time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n用户中断，准备退出...")
    finally:
        try:
            if orig_accel is not None:
                follower.write("Acceleration", orig_accel)
            if orig_speed is not None:
                follower.write("Goal_Speed", orig_speed)
        except Exception:
            pass
        if telemetry_stop is not None:
            telemetry_stop.set()
        if telemetry_thread and telemetry_thread.is_alive():
            telemetry_thread.join(timeout=2.0)
        if metrics_proc is not None and metrics_proc.poll() is None:
            try:
                metrics_proc.terminate()
            except Exception:
                pass
        _restore_i_coeff()
        try:
            robot.disconnect()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
