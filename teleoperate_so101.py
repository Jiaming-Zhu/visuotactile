"""SO-101 遥操作独立入口脚本（供 visuotactile 工作区直接使用）

本脚本的目标很单一：不复用 visuotactile 目录下已有的控制脚本，直接调用
LeRobot 内部已经实现好的 leader -> follower 遥操作链路，让主臂（leader）
实时驱动从臂（follower）。

===============================================================================
一、实现思路
===============================================================================
1. 本脚本不会自己重写遥操作算法，而是直接复用 LeRobot 中
   `ManipulatorRobot.teleop_step(record_data=False)` 的现成逻辑。
2. `teleop_step()` 在每次循环中会：
   - 读取 leader 当前关节位置（Present_Position）
   - 将 leader 的关节位置作为 follower 的目标位置（Goal_Position）
   - 立即写入 follower，从而实现“主从镜像控制”
3. 由于这里使用的是 `record_data=False`，因此不会额外采集相机、观测或数据集，
   只保留最高频的纯遥操作路径，适合调试、演示和手动控制。

===============================================================================
二、与 LeRobot 的关系
===============================================================================
1. 这个脚本只是一个轻量入口封装，核心硬件控制、校准读取、串口通信、关节读写
   都由 LeRobot 原生类处理。
2. 如果当前 Python 环境里无法直接 `import lerobot`，脚本会尝试把当前仓库中的
   `/home/martina/Y3_Project/lerobot` 自动加入 `sys.path`，从而支持在同一仓库内
   直接运行，而不要求额外设置 `PYTHONPATH`。
3. 机器人实例通过 `So101RobotConfig + ManipulatorRobot` 创建，默认沿用
   LeRobot 对 SO-101 的电机定义和关节命名。

===============================================================================
三、串口与硬件配置
===============================================================================
1. 默认情况下，LeRobot 的 `So101RobotConfig` 约定：
   - leader: `/dev/ttyACM0`
   - follower: `/dev/ttyACM1`
2. 但实际机器上 USB 设备枚举顺序可能变化，因此本脚本允许通过命令行覆盖：
   - `--leader-port`
   - `--follower-port`
3. 如果出现 `There is no status packet!`，通常表示：
   - 指向了错误的串口
   - 该串口设备存在，但舵机总线没有响应
   - 机械臂未上电，或 USB 转串口只连上了板卡、未连通伺服总线

===============================================================================
四、校准文件的加载方式
===============================================================================
1. 本脚本通过 `--calibration-dir` 指定校准目录，默认值为：
   `.cache/calibration/so101`
2. 这个路径会原样传给 `So101RobotConfig(calibration_dir=...)`，随后由
   `ManipulatorRobot` 在连接阶段自动加载对应 JSON。
3. 对于当前脚本使用的单臂命名（`main`），LeRobot 会读取：
   - `main_follower.json`
   - `main_leader.json`
4. 也就是说，本脚本不自己解析校准文件，而是完全交给 LeRobot 的标准流程处理。

===============================================================================
五、主循环行为
===============================================================================
1. `run_teleop()` 中会先调用 `robot.connect()`：
   - 连接 follower
   - 连接 leader
   - 激活 follower 力矩
   - 加载并应用校准参数
2. 之后进入无限循环（或直到达到 `--duration` 指定时长）：
   - 调用一次 `robot.teleop_step(record_data=False)`
   - 统计累计循环次数
   - 按 `--status-interval` 定期打印平均频率
3. 如果用户传入 `--fps`，脚本会在每轮末尾做简单 `sleep`，把控制频率限制在
   指定值附近；不传则为不封顶运行。

===============================================================================
六、退出与安全处理
===============================================================================
1. 按 `Ctrl+C` 会触发 `KeyboardInterrupt`，脚本会优雅退出。
2. 在 `finally` 中，只要机器人仍处于连接状态，就会执行断连清理。
3. 默认会先关闭 follower 力矩（`--disable-torque-on-exit` 默认开启），再调用
   `robot.disconnect()`，避免退出后舵机继续保持刚性锁定。

===============================================================================
七、设计边界
===============================================================================
1. 本脚本只负责“纯遥操作”，不负责：
   - 数据采集
   - 相机读取
   - 训练数据格式整理
   - 自定义控制策略推理
2. 如果后续需要把遥操作和 visuotactile 的视觉/触觉采集流程结合，建议在此脚本
   基础上继续扩展，而不是把已有复杂采集脚本硬塞进来。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _ensure_lerobot_on_path() -> None:
    """Allow running this script directly from the monorepo without extra PYTHONPATH."""
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


def _available_serial_ports() -> list[str]:
    ports: list[str] = []
    for pattern in ("ttyACM*", "ttyUSB*"):
        ports.extend(str(path) for path in sorted(Path("/dev").glob(pattern)))
    return ports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SO-101 leader->follower teleoperation")
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path(".cache/calibration/so101"),
        help="Calibration directory used by LeRobot.",
    )
    parser.add_argument(
        "--leader-port",
        type=str,
        default=None,
        help="Serial port for the leader arm. Default keeps LeRobot's built-in /dev/ttyACM0.",
    )
    parser.add_argument(
        "--follower-port",
        type=str,
        default=None,
        help="Serial port for the follower arm. Default keeps LeRobot's built-in /dev/ttyACM1.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional control rate limit. Omit for uncapped teleoperation.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional runtime limit in seconds. Omit to run until Ctrl+C.",
    )
    parser.add_argument(
        "--max-relative-target",
        type=int,
        default=None,
        help="Optional per-step safety limit passed to So101RobotConfig.",
    )
    parser.add_argument(
        "--disable-torque-on-exit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable follower torque before disconnecting.",
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=2.0,
        help="Seconds between lightweight status prints.",
    )
    return parser.parse_args()


def build_robot(args: argparse.Namespace) -> ManipulatorRobot:
    config = So101RobotConfig(
        calibration_dir=str(args.calibration_dir),
        max_relative_target=args.max_relative_target,
        cameras={},
    )

    if args.leader_port is not None:
        config.leader_arms["main"].port = args.leader_port
    if args.follower_port is not None:
        config.follower_arms["main"].port = args.follower_port

    return ManipulatorRobot(config)


def _disable_follower_torque(robot: ManipulatorRobot) -> None:
    for arm in robot.follower_arms.values():
        arm.write("Torque_Enable", 0)


def run_teleop(args: argparse.Namespace) -> None:
    robot = build_robot(args)
    loop_count = 0
    start_t = time.perf_counter()
    next_status_t = start_t + max(0.1, args.status_interval)

    try:
        robot.connect()
        print("Teleoperation started. Move the leader arm to drive the follower arm.")
        print("Press Ctrl+C to stop.")

        while True:
            loop_start_t = time.perf_counter()
            robot.teleop_step(record_data=False)
            loop_count += 1

            now = time.perf_counter()
            elapsed = now - start_t

            if args.duration is not None and elapsed >= args.duration:
                break

            if now >= next_status_t:
                avg_hz = loop_count / elapsed if elapsed > 0 else 0.0
                print(f"running {elapsed:.1f}s | loops={loop_count} | avg={avg_hz:.1f} Hz")
                next_status_t = now + max(0.1, args.status_interval)

            if args.fps is not None and args.fps > 0:
                remaining = (1.0 / args.fps) - (time.perf_counter() - loop_start_t)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        if robot.is_connected:
            if args.disable_torque_on_exit:
                try:
                    _disable_follower_torque(robot)
                except Exception as exc:
                    print(f"Warning: failed to disable follower torque: {exc}", file=sys.stderr)
            robot.disconnect()


def main() -> int:
    args = parse_args()
    try:
        run_teleop(args)
    except KeyboardInterrupt:
        print("\nTeleoperation interrupted by user.")
    except Exception as exc:
        print(f"Teleoperation failed: {exc}", file=sys.stderr)
        error_text = str(exc)
        if "There is no status packet" in error_text:
            ports = _available_serial_ports()
            ports_text = ", ".join(ports) if ports else "(none found)"
            print(
                "Likely causes: wrong leader serial port, leader arm not powered, or the leader USB adapter is connected but the servo bus is not responding.",
                file=sys.stderr,
            )
            print(f"Detected serial ports: {ports_text}", file=sys.stderr)
            print(
                "Try overriding ports explicitly, for example:",
                file=sys.stderr,
            )
            print(
                "  python /home/martina/Y3_Project/visuotactile/teleoperate_so101.py --leader-port /dev/ttyACM1 --follower-port /dev/ttyACM0",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
