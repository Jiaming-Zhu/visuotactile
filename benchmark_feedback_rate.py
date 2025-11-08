"""
SO101 舵机反馈频率基准测试
==========================

用途
- 在你的实际硬件连接下，测量读取若干反馈寄存器（位置/速度/负载/电流）时，控制循环的可达频率（Hz）。
- 通过选择不同的信号组合，得到最接近实际的数据采集上限。

特性
- 纯 headless（不启用相机/GUI）。
- 使用 LeRobot 的 ManipulatorRobot + Feetech 总线 GroupSyncRead。
- 支持选择 signals、测试时长、打印区间统计。

示例
  python learn_PyBullet/benchmark_feedback_rate.py \
    --signals position,speed,load,current \
    --duration 3

建议
- 四路全开时，通常 80–150 Hz；若需要更高频，请减少信号数量或电机数量。
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

from lerobot.common.robot_devices.robots.configs import So101RobotConfig
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.robot_devices.utils import busy_wait


SUPPORTED = {"position", "speed", "load", "current"}


def read_once(follower, signals: List[str]):
    """读取一次所选的反馈信号。

    返回一个 dict，仅为保持一致性；数据不做进一步处理。
    """
    out = {}
    if "position" in signals:
        out["position"] = np.asarray(follower.read("Present_Position"), dtype=np.float32)
    if "speed" in signals:
        fn = getattr(follower, "read_decoded", follower.read)
        out["speed"] = np.asarray(fn("Present_Speed"), dtype=np.float32)
    if "load" in signals:
        fn = getattr(follower, "read_decoded", follower.read)
        out["load"] = np.asarray(fn("Present_Load"), dtype=np.float32)
    if "current" in signals:
        out["current"] = np.asarray(follower.read("Present_Current"), dtype=np.float32)
    return out


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SO101 反馈频率基准测试")
    parser.add_argument(
        "--signals",
        type=str,
        default="position,speed,load,current",
        help="逗号分隔：position,speed,load,current 中的任意组合",
    )
    parser.add_argument("--duration", type=float, default=2.0, help="测试时长（秒）")
    parser.add_argument("--calibration-dir", type=str, default=str(Path(__file__).parent))
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="可选：目标环频(Hz)。若设置，将按该频率节拍（busy_wait），用于模拟固定采样频率。",
    )
    args = parser.parse_args(argv)

    signals = [s.strip().lower() for s in args.signals.split(",") if s.strip()]
    for s in signals:
        if s not in SUPPORTED:
            raise ValueError(f"不支持的信号 '{s}'，可选：{sorted(SUPPORTED)}")
    if not signals:
        raise ValueError("至少选择一个信号（position/speed/load/current）")

    robot = ManipulatorRobot(
        So101RobotConfig(
            calibration_dir=args.calibration_dir,
            leader_arms={},
            cameras={},
        )
    )

    try:
        robot.connect()
        follower_name = next(iter(robot.follower_arms.keys()))
        follower = robot.follower_arms[follower_name]

        # 预热（初始化 GroupSyncRead 等）
        for _ in range(5):
            read_once(follower, signals)

        t0 = time.perf_counter()
        t_end = t0 + float(args.duration)
        iters = 0
        dts = []

        while True:
            start = time.perf_counter()
            read_once(follower, signals)
            dt = time.perf_counter() - start
            dts.append(dt)
            iters += 1

            # 若设置了目标 fps，则在本轮末尾做节拍，模拟固定采样频率
            if args.fps is not None and args.fps > 0:
                target_period = 1.0 / float(args.fps)
                remain = target_period - dt
                if remain > 0:
                    busy_wait(remain)

            if time.perf_counter() >= t_end:
                break

        total_s = time.perf_counter() - t0
        avg_hz = iters / total_s if total_s > 0 else 0.0
        p50 = statistics.median(dts) if dts else 0.0
        p25 = statistics.quantiles(dts, n=4)[0] if len(dts) >= 4 else p50
        p75 = statistics.quantiles(dts, n=4)[-1] if len(dts) >= 4 else p50

        print("=== 反馈频率测试结果 ===")
        print(f"信号: {signals}")
        print(f"时长: {total_s:.3f} s  迭代: {iters}")
        print(f"平均频率: {avg_hz:.1f} Hz" + (f"（目标 {args.fps} Hz）" if args.fps else ""))
        if dts:
            # 额外提供 p95/p99
            p95, p99 = np.percentile(dts, [95, 99]) if len(dts) > 1 else (p50, p50)
            print(
                "单轮耗时 dt (ms): "
                f"p25={p25*1000:.2f}, p50={p50*1000:.2f}, p75={p75*1000:.2f}, p95={p95*1000:.2f}, p99={p99*1000:.2f}"
            )

    except KeyboardInterrupt:
        print("\n用户中断。")
    finally:
        try:
            if robot.is_connected:
                robot.disconnect()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
