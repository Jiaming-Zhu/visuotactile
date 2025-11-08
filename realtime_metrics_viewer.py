"""
实时显示真实机械臂各关节的负载、电流、位置、速度为波形曲线（GUI）。

依赖：matplotlib
读取键：
- Present_Load      -> 负载（原始单位，来自驱动器）
- Present_Current   -> 电流（mA）
- Present_Position  -> 位置（度，已校准）
- Present_Speed     -> 速度（驱动器原始单位，通常需换算；此处直接显示）

用法：
  python learn_PyBullet/realtime_metrics_viewer.py \
      --calib-dir /home/martina/Y3_Project/learn_PyBullet \
      --interval 0.1 --window 20

说明：
- calib-dir 指向校准文件所在目录（包含 main_follower.json）。
- interval 为采样周期（秒），window 为显示窗口大小（秒）。
"""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import argparse
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

# 复用已有的真实机械臂封装，避免重复实现
# 兼容作为脚本直接运行或作为包相对导入
try:
    from .interactive_control_oop import RealRobotInterface, LEROBOT_AVAILABLE
except Exception:  # noqa: BLE001 - 容错处理导入路径
    from pathlib import Path as _Path
    import sys as _sys
    _sys.path.insert(0, str(_Path(__file__).parent))
    from interactive_control_oop import RealRobotInterface, LEROBOT_AVAILABLE  # type: ignore


@dataclass
class MetricBuffers:
    load: List[Deque[float]]
    current: List[Deque[float]]
    position: List[Deque[float]]
    speed: List[Deque[float]]
    t_axis: Deque[float]


class RealTimeMetricsViewer:
    def __init__(self, calib_dir: str | None = None, interval: float = 0.1, window_sec: float = 20.0, reader: RealRobotInterface | None = None) -> None:
        """
        calib_dir: 若提供则内部建立连接；
        reader: 若提供则复用外部连接（不会在内部 connect/disconnect）。
        两者至少一个不为 None。
        """
        if not LEROBOT_AVAILABLE:
            raise RuntimeError("需要安装 lerobot 依赖后才能连接真实机械臂。")
        self.interval = float(interval)
        self.window_sec = float(window_sec)
        self._running = False

        if reader is not None:
            self.reader = reader
            self._own_reader = False
        elif calib_dir is not None:
            self.reader = RealRobotInterface(calib_dir)
            self.reader.connect()
            self._own_reader = True
        else:
            raise ValueError("必须提供 calib_dir 或现有 reader 其中之一")

        # 准备电机名称与通道数
        assert self.reader.robot is not None
        follower = next(iter(self.reader.robot.follower_arms.values()))
        self.motor_names: List[str] = list(follower.motor_names)
        self.n = len(self.motor_names)

        # 计算缓存长度
        self.maxlen = max(1, int(self.window_sec / self.interval))

        # 初始化缓冲
        self.buffers = MetricBuffers(
            load=[deque([0.0], maxlen=self.maxlen) for _ in range(self.n)],
            current=[deque([0.0], maxlen=self.maxlen) for _ in range(self.n)],
            position=[deque([0.0], maxlen=self.maxlen) for _ in range(self.n)],
            speed=[deque([0.0], maxlen=self.maxlen) for _ in range(self.n)],
            t_axis=deque([0.0], maxlen=self.maxlen),
        )

        # 创建图形与轴：4 行 1 列
        self.fig, self.axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        self.fig.canvas.manager.set_window_title("Real Robot Metrics")

        # 每条曲线一条 line
        self.lines: Dict[str, List] = {
            "load": [],
            "current": [],
            "position": [],
            "speed": [],
        }

        colors = plt.cm.get_cmap("tab10", max(10, self.n))
        x0 = list(self.buffers.t_axis)
        for i, name in enumerate(self.motor_names):
            color = colors(i % 10)
            # 负载
            (l1,) = self.axes[0].plot(x0, list(self.buffers.load[i]), color=color, label=name)
            # 电流
            (l2,) = self.axes[1].plot(x0, list(self.buffers.current[i]), color=color, label=name)
            # 位置
            (l3,) = self.axes[2].plot(x0, list(self.buffers.position[i]), color=color, label=name)
            # 速度
            (l4,) = self.axes[3].plot(x0, list(self.buffers.speed[i]), color=color, label=name)
            self.lines["load"].append(l1)
            self.lines["current"].append(l2)
            self.lines["position"].append(l3)
            self.lines["speed"].append(l4)

        self.axes[0].set_ylabel("Load (%) (signed)")
        self.axes[1].set_ylabel("Current (mA)")
        self.axes[2].set_ylabel("Position (deg)")
        self.axes[3].set_ylabel("Speed (deg/s) (signed)")
        self.axes[3].set_xlabel("Time (s)")

        for ax in self.axes:
            ax.grid(True, linestyle=":", alpha=0.4)
            ax.legend(loc="upper left", ncol=min(self.n, 4), fontsize=8)

        self._t0 = time.time()

    def _read_all_metrics(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert self.reader.robot is not None
        follower = next(iter(self.reader.robot.follower_arms.values()))
        # 注意：位置/电流直接返回物理单位；负载/速度使用解码后的带符号值
        if hasattr(follower, "read_decoded"):
            load = np.array(follower.read_decoded("Present_Load"), dtype=float)
            speed = np.array(follower.read_decoded("Present_Speed"), dtype=float)
        else:
            load = np.array(follower.read("Present_Load"), dtype=float)
            speed = np.array(follower.read("Present_Speed"), dtype=float)
        current = np.array(follower.read("Present_Current"), dtype=float)
        position = np.array(follower.read("Present_Position"), dtype=float)
        return load, current, position, speed

    def _append_samples(self, t: float, load: np.ndarray, current: np.ndarray, position: np.ndarray, speed: np.ndarray) -> None:
        self.buffers.t_axis.append(t)
        for i in range(self.n):
            self.buffers.load[i].append(float(load[i]))
            self.buffers.current[i].append(float(current[i]))
            self.buffers.position[i].append(float(position[i]))
            self.buffers.speed[i].append(float(speed[i]))

    def _refresh_plot(self) -> None:
        xs = list(self.buffers.t_axis)
        for i in range(self.n):
            self.lines["load"][i].set_data(xs, list(self.buffers.load[i]))
            self.lines["current"][i].set_data(xs, list(self.buffers.current[i]))
            self.lines["position"][i].set_data(xs, list(self.buffers.position[i]))
            self.lines["speed"][i].set_data(xs, list(self.buffers.speed[i]))
        # 自适应 y 轴
        for ax in self.axes:
            ax.relim()
            ax.autoscale_view()
        # x 轴仅显示窗口范围
        if xs:
            self.axes[-1].set_xlim(max(0, xs[0]), xs[-1])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def loop(self) -> None:
        """阻塞式循环，直到 stop() 被调用或窗口关闭/中断。"""
        plt.ion()
        self._running = True
        try:
            while self._running:
                now = time.time() - self._t0
                try:
                    load, current, position, speed = self._read_all_metrics()
                    self._append_samples(now, load, current, position, speed)
                except Exception as e:
                    # 读取失败时保留上一帧
                    print(f"读取失败: {e}")
                self._refresh_plot()
                plt.pause(self.interval)
        except KeyboardInterrupt:
            pass
        finally:
            plt.ioff()
            if self._own_reader:
                try:
                    self.reader.disconnect()
                except Exception:
                    pass

    def stop(self) -> None:
        self._running = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time metrics viewer for the physical arm")
    parser.add_argument("--calib-dir", type=str, default=str(Path(__file__).parent), help="Calibration dir containing main_follower.json")
    parser.add_argument("--interval", type=float, default=0.1, help="Sampling interval (s)")
    parser.add_argument("--window", type=float, default=20.0, help="Window size (s)")
    parser.add_argument("--mode", type=str, choices=["serial", "tcp"], default="serial", help="Data source mode: serial or tcp")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="TCP host (when mode=tcp)")
    parser.add_argument("--port", type=int, default=8765, help="TCP port (when mode=tcp)")
    args = parser.parse_args()

    if args.mode == "serial":
        viewer = RealTimeMetricsViewer(args.calib_dir, args.interval, args.window)
        viewer.loop()
    else:
        # TCP 客户端模式：从 socket 读取 JSON 行并绘制
        import socket
        import json
        plt.ion()
        # 等待服务器就绪（最多 10 秒）
        start_wait = time.time()
        sock = None
        while time.time() - start_wait < 10.0:
            try:
                sock = socket.create_connection((args.host, args.port))
                break
            except OSError:
                time.sleep(0.2)
        if sock is None:
            print(f"无法连接到遥测服务器 {args.host}:{args.port}，请先启动提供端（metrics 线程）。")
            return

        # 简化：直接创建绘图结构（不依赖 RealRobotInterface）
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        axes[0].set_ylabel("Load (%) (signed)")
        axes[1].set_ylabel("Current (mA)")
        axes[2].set_ylabel("Position (deg)")
        axes[3].set_ylabel("Speed (deg/s) (signed)")
        axes[3].set_xlabel("Time (s)")
        for ax in axes:
            ax.grid(True, linestyle=":", alpha=0.4)

        # 延迟获取电机数量与名称：由第一帧决定
        motor_names = None
        lines = {"load": [], "current": [], "position": [], "speed": []}
        t_axis = []
        buffers = {"load": [], "current": [], "position": [], "speed": []}
        start_t = time.time()

        def ensure_lines(n, names):
            if lines["load"]:
                return
            # 使用新版接口避免弃用警告
            try:
                import matplotlib
                cmap = matplotlib.colormaps.get_cmap("tab10")
            except Exception:
                cmap = plt.get_cmap("tab10")
            for i in range(n):
                color = cmap(i % 10) if callable(cmap) else cmap(i % 10)
                (l1,) = axes[0].plot([], [], color=color, label=names[i])
                (l2,) = axes[1].plot([], [], color=color, label=names[i])
                (l3,) = axes[2].plot([], [], color=color, label=names[i])
                (l4,) = axes[3].plot([], [], color=color, label=names[i])
                lines["load"].append(l1)
                lines["current"].append(l2)
                lines["position"].append(l3)
                lines["speed"].append(l4)
            for ax in axes:
                ax.legend(loc="upper left", ncol=min(n, 4), fontsize=8)

        try:
            with sock:
                sock_file = sock.makefile("r")
                while True:
                    line = sock_file.readline()
                    if not line:
                        break
                    try:
                        packet = json.loads(line)
                    except Exception:
                        continue
                    if motor_names is None:
                        motor_names = packet.get("names", [])
                        n = len(motor_names)
                        win_len = max(1, int(args.window/args.interval))
                        buffers = {k: [deque(maxlen=win_len) for _ in range(n)] for k in ("load","current","position","speed")}
                        t_axis = deque(maxlen=win_len)
                        ensure_lines(n, motor_names)
                    t = time.time() - start_t
                    for key in ("load","current","position","speed"):
                        vals = packet.get(key, [])
                        for i, v in enumerate(vals):
                            buffers[key][i].append(float(v))
                    t_axis.append(t)
                    xs_full = list(t_axis)
                    for i in range(len(motor_names)):
                        # 对齐每条曲线的 x/y 长度（以 y 为基准截取 x）
                        for key in ("load","current","position","speed"):
                            y = list(buffers[key][i])
                            if len(y) == 0:
                                x = []
                            else:
                                x = xs_full[-len(y):]
                            lines[key][i].set_data(x, y)
                    for ax in axes:
                        ax.relim(); ax.autoscale_view()
                    if xs_full:
                        axes[-1].set_xlim(max(0, xs_full[0]), xs_full[-1])
                    plt.pause(args.interval)
        except KeyboardInterrupt:
            pass
        finally:
            plt.ioff()


if __name__ == "__main__":
    main()
