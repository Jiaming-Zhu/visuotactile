#!/usr/bin/env python3
"""
preview_camera.py
=================

Real-time camera preview utility that follows the same capture path used by
collect_custom_multimodal.py.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

try:
    import tkinter as tk
except ImportError:  # pragma: no cover
    tk = None

try:
    from PIL import Image, ImageTk
except ImportError:  # pragma: no cover
    Image = None
    ImageTk = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="摄像头实时预览与拍照测试脚本")
    default_output_dir = Path(__file__).parent / "outputs" / "camera_preview"
    parser.add_argument("--camera-index", type=int, default=0, help="摄像头索引")
    parser.add_argument("--camera-width", type=int, default=640, help="预览宽度")
    parser.add_argument("--camera-height", type=int, default=480, help="预览高度")
    parser.add_argument("--camera-warmup", type=int, default=5, help="打开后丢弃的预热帧数")
    parser.add_argument("--camera-refresh", type=int, default=3, help="每次显示前额外丢弃的刷新帧数")
    parser.add_argument("--scan-max-index", type=int, default=10, help="扫描摄像头时检查的最大索引数量")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="保存测试照片的目录",
    )
    parser.add_argument("--save-prefix", type=str, default="camera_preview", help="保存图片的文件名前缀")
    parser.add_argument(
        "--skip-selection",
        action="store_true",
        help="跳过交互式摄像头扫描，直接使用 --camera-index",
    )
    return parser.parse_args()


def _ensure_frame_size(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height))


def _make_snapshot_path(
    output_dir: Path,
    prefix: str,
    now: Optional[datetime] = None,
) -> Path:
    timestamp = now or datetime.now()
    stamp = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return output_dir / f"{prefix}_{stamp}.jpg"


def _opencv_gui_available(build_info: Optional[str] = None) -> bool:
    info = build_info if build_info is not None else cv2.getBuildInformation()
    for line in info.splitlines():
        if line.strip().startswith("GUI:"):
            gui_backend = line.split(":", 1)[1].strip().upper()
            return gui_backend not in {"", "NONE", "NO"}
    return False


def _tkinter_gui_available() -> bool:
    if tk is None or Image is None or ImageTk is None:
        return False

    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        root.update_idletasks()
        return True
    except Exception:
        return False
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass


def _select_preview_backend(opencv_gui_available: bool, tkinter_available: bool) -> str:
    if opencv_gui_available:
        return "opencv"
    if tkinter_available:
        return "tk"
    return "terminal"


def _scan_available_cameras(max_index: int = 10) -> List[Dict[str, Any]]:
    available = []
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
                return available_cameras[default_in_list]["index"]
            print(f"⚠️ 默认索引 {default_index} 不可用，将使用第一个可用摄像头。")
            return available_cameras[0]["index"]

        try:
            choice = int(user_input)
        except ValueError:
            print("⚠️ 无效输入，请输入数字编号。")
            continue

        if 1 <= choice <= len(available_cameras):
            return available_cameras[choice - 1]["index"]
        print(f"⚠️ 请输入 1 到 {len(available_cameras)} 之间的数字。")


class PreviewCamera:
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
        return _ensure_frame_size(frame, self.width, self.height)

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def _save_snapshot(frame: np.ndarray, output_dir: Path, prefix: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = _make_snapshot_path(output_dir=output_dir, prefix=prefix)
    if not cv2.imwrite(str(save_path), frame):
        raise RuntimeError(f"保存图片失败：{save_path}")
    return save_path


def _build_overlay_lines(
    camera: PreviewCamera,
    frame: np.ndarray,
    saved_count: int,
    last_message: str,
) -> List[str]:
    return [
        f"Camera Index: {camera.index}",
        f"Resolution: {frame.shape[1]}x{frame.shape[0]}",
        f"Saved Frames: {saved_count}",
        "Press S to save current frame",
        "Press Q or Esc to quit",
        last_message,
    ]


def _draw_overlay(frame: np.ndarray, lines: List[str]) -> np.ndarray:
    overlay = frame.copy()
    y_offset = 30
    for line in lines:
        cv2.putText(
            overlay,
            line,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y_offset += 28
    return overlay


def _run_terminal_snapshot_loop(camera: PreviewCamera, output_dir: Path, prefix: str) -> int:
    print("⚠️ 当前 OpenCV 没有 GUI 支持，无法进行实时窗口预览。")
    print("你仍然可以在终端中测试拍照：直接回车保存一张，输入 q 退出。")

    while True:
        user_input = input("[Enter]=拍照保存, q=退出: ").strip().lower()
        if user_input == "q":
            return 0
        frame = camera.capture()
        save_path = _save_snapshot(frame, output_dir=output_dir, prefix=prefix)
        print(f"📸 已保存当前帧到: {save_path}")


def _run_opencv_preview_loop(camera: PreviewCamera, output_dir: Path, prefix: str) -> int:
    window_name = f"Camera {camera.index} Live Preview"
    saved_count = 0
    last_message = f"Save directory: {output_dir}"

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, camera.width, camera.height)
    except cv2.error:
        return _run_terminal_snapshot_loop(camera, output_dir=output_dir, prefix=prefix)

    print("=" * 60)
    print(f"  摄像头实时预览已打开 (索引: {camera.index})")
    print("  按 S 保存当前帧")
    print("  按 Q 或 Esc 退出")
    print("=" * 60)

    try:
        while True:
            frame = camera.capture()
            overlay = _draw_overlay(
                frame,
                _build_overlay_lines(
                    camera=camera,
                    frame=frame,
                    saved_count=saved_count,
                    last_message=last_message,
                ),
            )
            cv2.imshow(window_name, overlay)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                return 0

            if key in (ord("s"), ord("S")):
                save_path = _save_snapshot(frame, output_dir=output_dir, prefix=prefix)
                saved_count += 1
                last_message = f"Saved: {save_path.name}"
                print(f"📸 已保存当前帧到: {save_path}")
    finally:
        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)


def _run_tk_preview_loop(camera: PreviewCamera, output_dir: Path, prefix: str) -> int:
    if tk is None or Image is None or ImageTk is None:
        return _run_terminal_snapshot_loop(camera, output_dir=output_dir, prefix=prefix)

    output_dir.mkdir(parents=True, exist_ok=True)
    root = tk.Tk()
    root.title(f"Camera {camera.index} Live Preview")

    image_label = tk.Label(root)
    image_label.pack(padx=8, pady=8)

    status_var = tk.StringVar(value=f"Save directory: {output_dir}")
    status_label = tk.Label(root, textvariable=status_var, anchor="w", justify="left")
    status_label.pack(fill="x", padx=8)

    hint_label = tk.Label(root, text="快捷键: S 保存当前帧, Q / Esc 退出")
    hint_label.pack(fill="x", padx=8, pady=(4, 8))

    button_frame = tk.Frame(root)
    button_frame.pack(fill="x", padx=8, pady=(0, 8))

    state: Dict[str, Any] = {
        "saved_count": 0,
        "last_message": f"Save directory: {output_dir}",
        "current_frame": None,
        "photo_image": None,
        "exit_code": 0,
    }

    def save_current_frame(_event: Any = None) -> None:
        frame = state["current_frame"]
        if frame is None:
            status_var.set("Waiting for first frame...")
            return
        save_path = _save_snapshot(frame, output_dir=output_dir, prefix=prefix)
        state["saved_count"] += 1
        state["last_message"] = f"Saved: {save_path.name}"
        status_var.set(state["last_message"])
        print(f"📸 已保存当前帧到: {save_path}")

    def request_close(_event: Any = None) -> None:
        root.quit()

    tk.Button(button_frame, text="保存当前帧 (S)", command=save_current_frame).pack(side="left")
    tk.Button(button_frame, text="退出 (Q)", command=request_close).pack(side="right")

    root.bind("<KeyPress-s>", save_current_frame)
    root.bind("<KeyPress-S>", save_current_frame)
    root.bind("<KeyPress-q>", request_close)
    root.bind("<KeyPress-Q>", request_close)
    root.bind("<Escape>", request_close)
    root.protocol("WM_DELETE_WINDOW", request_close)

    print("=" * 60)
    print(f"  摄像头实时预览已打开 (索引: {camera.index}, backend: Tk)")
    print("  按 S 保存当前帧")
    print("  按 Q 或 Esc 退出")
    print("=" * 60)

    def update_frame() -> None:
        try:
            frame = camera.capture()
            state["current_frame"] = frame.copy()
            overlay = _draw_overlay(
                frame,
                _build_overlay_lines(
                    camera=camera,
                    frame=frame,
                    saved_count=state["saved_count"],
                    last_message=state["last_message"],
                ),
            )
            rgb_frame = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(rgb_frame))
            state["photo_image"] = photo
            image_label.configure(image=photo)
            status_var.set(state["last_message"])
        except Exception as exc:
            state["last_message"] = f"Preview error: {exc}"
            status_var.set(state["last_message"])

        if root.winfo_exists():
            root.after(15, update_frame)

    try:
        root.after(0, update_frame)
        root.mainloop()
        return int(state["exit_code"])
    finally:
        if root.winfo_exists():
            root.destroy()


def _run_preview_loop(camera: PreviewCamera, output_dir: Path, prefix: str) -> int:
    backend = _select_preview_backend(
        opencv_gui_available=_opencv_gui_available(),
        tkinter_available=_tkinter_gui_available(),
    )

    if backend == "opencv":
        try:
            return _run_opencv_preview_loop(camera, output_dir=output_dir, prefix=prefix)
        except cv2.error:
            backend = _select_preview_backend(
                opencv_gui_available=False,
                tkinter_available=_tkinter_gui_available(),
            )

    if backend == "tk":
        return _run_tk_preview_loop(camera, output_dir=output_dir, prefix=prefix)

    return _run_terminal_snapshot_loop(camera, output_dir=output_dir, prefix=prefix)


def main() -> int:
    args = parse_args()

    if args.skip_selection:
        camera_index = args.camera_index
    else:
        available_cameras = _scan_available_cameras(max_index=max(1, args.scan_max_index))
        camera_index = _select_camera_interactive(
            available_cameras=available_cameras,
            default_index=args.camera_index,
        )
        if camera_index is None:
            raise SystemExit("未选择摄像头，退出。")

    camera = PreviewCamera(
        index=camera_index,
        width=args.camera_width,
        height=args.camera_height,
        warmup_frames=args.camera_warmup,
        refresh_frames=args.camera_refresh,
    )
    try:
        return _run_preview_loop(
            camera=camera,
            output_dir=args.output_dir,
            prefix=args.save_prefix,
        )
    finally:
        camera.close()


if __name__ == "__main__":
    raise SystemExit(main())
