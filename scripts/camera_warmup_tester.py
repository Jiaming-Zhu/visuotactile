#!/usr/bin/env python3
"""
Simple utility to estimate how many frames should be discarded after a camera starts.

The script repeatedly opens the target camera, discards N frames, and evaluates the next
few frames with basic image-quality metrics (sharpness/contrast/brightness). Results are
printed in a table so you can pick the warm-up length that yields the best focus score.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import cv2
import numpy as np


@dataclass
class TrialResult:
    drop_frames: int
    samples: int
    laplacian: float
    contrast: float
    brightness: float
    snapshot: Optional[Path]


def _open_camera(idx: int) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    if cap.isOpened():
        return cap
    cap.release()
    cap = cv2.VideoCapture(idx)
    if cap.isOpened():
        return cap
    cap.release()
    return None


def _list_available_cameras(max_index: int) -> List[int]:
    available: List[int] = []
    for idx in range(max_index + 1):
        cap = _open_camera(idx)
        if cap is None:
            continue
        ret, _ = cap.read()
        if ret:
            available.append(idx)
        cap.release()
    return available


def _ensure_drop_values(max_drop: int, step: int, manual: Sequence[int] | None) -> List[int]:
    if manual:
        values = sorted({v for v in manual if v >= 0})
        if values:
            return values
    return sorted(set(max(0, x) for x in range(0, max_drop + 1, max(1, step))))


def _quality_metrics(frame: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    brightness = float(gray.mean())
    return {"laplacian": lap_var, "contrast": contrast, "brightness": brightness}


def _run_trial(
    camera_id: int,
    drop_frames: int,
    sample_frames: int,
    width: int,
    height: int,
    open_delay: float,
    frame_delay: float,
    snapshot_dir: Optional[Path],
) -> Optional[TrialResult]:
    cap = _open_camera(camera_id)
    if cap is None or not cap.isOpened():
        if cap is not None:
            cap.release()
        raise RuntimeError(f"无法打开摄像头 {camera_id}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if open_delay > 0:
        time.sleep(open_delay)

    for _ in range(max(0, drop_frames)):
        ret, _ = cap.read()
        if not ret:
            cap.release()
            return None
        if frame_delay > 0:
            time.sleep(frame_delay)

    metrics_list = []
    last_frame: Optional[np.ndarray] = None
    for _ in range(max(1, sample_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        metrics_list.append(_quality_metrics(frame))
        last_frame = frame.copy()
        if frame_delay > 0:
            time.sleep(frame_delay)

    cap.release()
    if not metrics_list:
        return None

    laplacian = float(np.mean([m["laplacian"] for m in metrics_list]))
    contrast = float(np.mean([m["contrast"] for m in metrics_list]))
    brightness = float(np.mean([m["brightness"] for m in metrics_list]))

    snapshot_path: Optional[Path] = None
    if snapshot_dir and last_frame is not None:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshot_dir / f"drop_{drop_frames:03d}.jpg"
        cv2.imwrite(str(snapshot_path), last_frame)

    return TrialResult(drop_frames, len(metrics_list), laplacian, contrast, brightness, snapshot_path)


def _print_table(results: Iterable[TrialResult]) -> None:
    header = f"{'Drop':>6} | {'LapVar':>12} | {'Contrast':>9} | {'Brightness':>11} | {'Samples':>7} | Snapshot"
    print("\n测量结果（LapVar 越大越锐利，Contrast 越大越清晰）：")
    print(header)
    print("-" * len(header))
    for r in results:
        snap = r.snapshot.name if r.snapshot else "-"
        print(
            f"{r.drop_frames:6d} | {r.laplacian:12.1f} | {r.contrast:9.1f} | "
            f"{r.brightness:11.1f} | {r.samples:7d} | {snap}"
        )


def parse_drop_list(text: str) -> List[int]:
    values: List[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            values.append(int(chunk))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"无法解析 '{chunk}' 为整数") from exc
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="测试摄像头启动后需要丢弃多少帧才能达到最佳画质，LapVar 越大越锐利。"
    )
    parser.add_argument("--cam-index", type=int, default=None, help="指定摄像头索引（默认自动寻找可用摄像头）")
    parser.add_argument("--max-index", type=int, default=5, help="自动搜索摄像头时的最大索引（包含）")
    parser.add_argument("--width", type=int, default=640, help="采集宽度")
    parser.add_argument("--height", type=int, default=480, help="采集高度")
    parser.add_argument("--max-drop", type=int, default=60, help="最大丢帧数量（仅在未指定 --drop-list 时生效）")
    parser.add_argument("--step", type=int, default=5, help="扫描丢帧数量时的步长")
    parser.add_argument(
        "--drop-list",
        type=parse_drop_list,
        default=None,
        help="逗号分隔的丢帧数量列表（如 0,5,10,20），若提供则覆盖 --max-drop/--step",
    )
    parser.add_argument("--samples", type=int, default=5, help="每个丢帧配置下评估的帧数")
    parser.add_argument("--open-delay", type=float, default=0.2, help="摄像头打开后等待的秒数")
    parser.add_argument("--frame-delay", type=float, default=0.0, help="每次读取帧之间的等待秒数")
    parser.add_argument("--snapshot-dir", type=Path, default=None, help="若提供则保存每个配置下的示例帧")
    args = parser.parse_args()

    drop_values = _ensure_drop_values(args.max_drop, args.step, args.drop_list)
    if not drop_values:
        raise SystemExit("未提供合法的丢帧数量。")

    cam_id = args.cam_index
    if cam_id is None:
        cams = _list_available_cameras(args.max_index)
        if not cams:
            raise SystemExit("无法自动找到可用的摄像头，请通过 --cam-index 指定。")
        cam_id = cams[0]
        print(f"自动选择摄像头 {cam_id}，可用列表：{cams}")
    else:
        print(f"使用指定摄像头 {cam_id}")

    results: List[TrialResult] = []
    for drop in drop_values:
        print(f"\n▶️  丢弃 {drop} 帧后开始采样 ...")
        try:
            trial = _run_trial(
                cam_id,
                drop,
                args.samples,
                args.width,
                args.height,
                args.open_delay,
                args.frame_delay,
                args.snapshot_dir,
            )
        except RuntimeError as exc:
            print(f"❌ {exc}")
            break
        if trial is None:
            print("⚠️ 未能读取到足够的帧，已跳过。")
            continue
        print(
            f"    LapVar={trial.laplacian:.1f}, Contrast={trial.contrast:.1f}, "
            f"Brightness={trial.brightness:.1f}（样本 {trial.samples}）"
        )
        results.append(trial)

    if not results:
        print("没有成功的测试结果，请检查摄像头连接。")
        return

    results.sort(key=lambda r: r.laplacian, reverse=True)
    _print_table(results)

    best = results[0]
    snapshot_msg = f"，示例帧 {best.snapshot}" if best.snapshot else ""
    print(
        f"\n✅ 推荐先丢弃 {best.drop_frames} 帧（LapVar={best.laplacian:.1f}, "
        f"Contrast={best.contrast:.1f}, Brightness={best.brightness:.1f}{snapshot_msg}）。"
    )


if __name__ == "__main__":
    main()
