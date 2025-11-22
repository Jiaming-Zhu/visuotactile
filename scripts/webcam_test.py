import argparse
import threading
import time
from http import server
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np


def _open_camera(idx: int) -> Optional[cv2.VideoCapture]:
    """尝试按 V4L2 → 默认顺序打开摄像头。"""
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    if cap.isOpened():
        return cap
    cap.release()
    cap = cv2.VideoCapture(idx)
    if cap.isOpened():
        return cap
    cap.release()
    return None


def list_available_cameras(max_index: int) -> List[int]:
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


def _preview_window(camera_id: int, width: int, height: int, warmup: int) -> None:
    cap = _open_camera(camera_id)
    if cap is None or not cap.isOpened():
        if cap is not None:
            cap.release()
        print(f"❌ 无法打开摄像头 {camera_id}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    for _ in range(max(1, warmup)):
        cap.read()

    window = f"Camera {camera_id} (q/ESC 退出, s 保存)"
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    except cv2.error as exc:  # 无 GUI 支持
        cap.release()
        raise RuntimeError(
            "OpenCV GUI 不可用，请安装 GTK/Qt 等依赖或改用 --mode web。"
        ) from exc

    save_dir = Path("snapshots")
    save_dir.mkdir(exist_ok=True)
    counter = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ 无法读取视频帧，结束预览")
                break
            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                path = save_dir / f"snapshot_cam{camera_id}_{counter}.jpg"
                cv2.imwrite(str(path), frame)
                counter += 1
                print(f"💾 已保存 {path}")
    finally:
        cap.release()
        cv2.destroyWindow(window)


class _FrameBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None

    def update(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame.copy()

    def get_jpeg(self) -> Optional[bytes]:
        with self._lock:
            if self._frame is None:
                return None
            ret, buf = cv2.imencode(".jpg", self._frame)
            if not ret:
                return None
            return buf.tobytes()


class _StreamingHandler(server.BaseHTTPRequestHandler):
    def __init__(self, *args, frame_provider: Callable[[], Optional[bytes]], **kwargs):
        self._frame_provider = frame_provider
        super().__init__(*args, **kwargs)

    def do_GET(self):  # noqa: N802
        if self.path not in {"/", "/stream"}:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                frame = self._frame_provider()
                if frame is None:
                    time.sleep(0.05)
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                time.sleep(0.03)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            pass

    def log_message(self, format, *args):  # noqa: A003
        return


def _preview_web(
    camera_id: int,
    width: int,
    height: int,
    warmup: int,
    host: str,
    port: int,
    fps: float,
) -> None:
    cap = _open_camera(camera_id)
    if cap is None or not cap.isOpened():
        if cap is not None:
            cap.release()
        print(f"❌ 无法打开摄像头 {camera_id}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    for _ in range(max(1, warmup)):
        cap.read()

    buffer = _FrameBuffer()

    def provider() -> Optional[bytes]:
        return buffer.get_jpeg()

    handler = lambda *args, **kwargs: _StreamingHandler(*args, frame_provider=provider, **kwargs)  # noqa: E731
    httpd = server.ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    print(f"🌐 打开 http://{host}:{port}/ 查看实时画面，Ctrl+C 退出。")

    delay = 0.0 if fps <= 0 else 1.0 / fps
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ 无法读取视频帧，结束预览")
                break
            buffer.update(frame)
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        print("\n收到中断，正在关闭预览...")
    finally:
        cap.release()
        httpd.shutdown()
        thread.join(timeout=2.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="实时显示摄像头画面，便于调整位置")
    parser.add_argument("--cam-index", type=int, default=None, help="指定摄像头索引；不填则自动选择第一个可用摄像头")
    parser.add_argument("--max-index", type=int, default=10, help="自动搜索时的最大索引 (包含)")
    parser.add_argument("--width", type=int, default=640, help="预览宽度")
    parser.add_argument("--height", type=int, default=640, help="预览高度")
    parser.add_argument("--warmup", type=int, default=10, help="采集前预热帧数")
    parser.add_argument("--mode", choices={"window", "web"}, default="window", help="显示模式 (window/web)")
    parser.add_argument("--web-host", type=str, default="127.0.0.1", help="web 模式监听地址")
    parser.add_argument("--web-port", type=int, default=8767, help="web 模式监听端口")
    parser.add_argument("--fps", type=float, default=30.0, help="web 模式推流帧率 (<=0 表示尽力而为)")
    args = parser.parse_args()

    cam_id = args.cam_index
    if cam_id is None:
        cams = list_available_cameras(args.max_index)
        if not cams:
            print("⚠️ 未检测到任何可用摄像头（请确认设备或调整 --max-index）")
            return
        cam_id = cams[0]
        print(f"自动选择摄像头 {cam_id}（可用摄像头列表: {cams}）")
    else:
        print(f"使用指定摄像头 {cam_id}")

    if args.mode == "window":
        try:
            _preview_window(cam_id, args.width, args.height, args.warmup)
        except RuntimeError as exc:
            print(f"{exc}\n➡️ 正在自动切换到 web 模式 ...")
            _preview_web(cam_id, args.width, args.height, args.warmup, args.web_host, args.web_port, args.fps)
    else:
        _preview_web(cam_id, args.width, args.height, args.warmup, args.web_host, args.web_port, args.fps)


if __name__ == "__main__":
    main()
