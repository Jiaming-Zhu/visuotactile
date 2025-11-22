#!/usr/bin/env python3
"""
Quick visualization utility for Plaintextdataset episodes.

Features
--------
- Lists available episode directories under the dataset root.
- Loads metadata.json + tactile_data.pkl for a chosen episode.
- Shows time-series plots (position/velocity/load/current) and the visual anchor.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def _list_episode_dirs(dataset_root: Path) -> List[Path]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"数据集目录不存在：{dataset_root}")
    dirs = sorted(
        (p for p in dataset_root.iterdir() if p.is_dir() and p.name.startswith("episode_")),
        key=lambda p: p.name,
    )
    if not dirs:
        raise FileNotFoundError(f"{dataset_root} 下没有 episode_* 子目录。")
    return dirs


def _choose_episode(dataset_root: Path, episode_name: str | None, index: int | None) -> Path:
    episodes = _list_episode_dirs(dataset_root)
    if episode_name:
        target = dataset_root / episode_name
        if not target.exists():
            raise FileNotFoundError(f"指定的 episode 目录不存在：{target}")
        return target
    if index is not None:
        if not (0 <= index < len(episodes)):
            raise IndexError(f"索引 {index} 超出范围 (0~{len(episodes)-1})")
        return episodes[index]
    # 默认取最新（按名称排序后的最后一个）
    return episodes[-1]


def _load_metadata(episode_dir: Path) -> Dict[str, object]:
    meta_path = episode_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{episode_dir} 缺少 metadata.json")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _load_episode(episode_dir: Path) -> Dict[str, object]:
    meta_path = episode_dir / "metadata.json"
    tactile_path = episode_dir / "tactile_data.pkl"
    anchor_path = episode_dir / "visual_anchor.jpg"
    if not meta_path.exists() or not tactile_path.exists() or not anchor_path.exists():
        raise FileNotFoundError(f"{episode_dir} 缺少 metadata/tactile/anchor 文件。")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    with tactile_path.open("rb") as f:
        tactile = pickle.load(f)
    image = mpimg.imread(anchor_path)
    return {"metadata": metadata, "tactile": tactile, "anchor": image}


def _plot_episode(data: Dict[str, object], save_path: Path | None) -> None:
    tactile = data["tactile"]
    timestamps = np.asarray(tactile["timestamps"], dtype=np.float64)

    joint_pos = np.asarray(tactile.get("joint_position_profile", []), dtype=np.float64)
    joint_vel = np.asarray(tactile.get("joint_velocity_profile", []), dtype=np.float64)
    joint_load = np.asarray(tactile.get("joint_load_profile", []), dtype=np.float64)
    joint_curr = np.asarray(tactile.get("joint_current_profile", []), dtype=np.float64)

    cmap = cm.get_cmap("tab10")
    joint_names = data["metadata"].get("joint_names") or [
        f"J{i}" for i in range(joint_pos.shape[1] if joint_pos.ndim == 2 else 0)
    ]

    panels = []
    titles = []
    if joint_pos.size:
        panels.append(joint_pos)
        titles.append("Joint Position (deg)")
    if joint_vel.size:
        panels.append(joint_vel)
        titles.append("Joint Velocity (deg/s)")
    if joint_load.size:
        panels.append(joint_load)
        titles.append("Joint Load (decoded)")
    if joint_curr.size:
        panels.append(joint_curr)
        titles.append("Joint Current (decoded)")

    if not panels:
        print("⚠️ 本 episode 缺少 joint_* 数据，回退到 gripper 通道。")
        panels = [
            np.asarray(tactile["gripper_width_profile"], dtype=np.float64).reshape(-1, 1),
            np.asarray(tactile["gripper_velocity_profile"], dtype=np.float64).reshape(-1, 1),
            np.asarray(tactile["load_profile"], dtype=np.float64).reshape(-1, 1),
            np.asarray(tactile["lift_current_profile"], dtype=np.float64).reshape(-1, 1),
        ]
        titles = ["Gripper Width", "Gripper Velocity", "Gripper Load", "Lift Current"]
        joint_names = ["gripper"]

    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 3 * len(panels)), sharex=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, panel, title in zip(axes, panels, titles):
        panel = np.asarray(panel, dtype=np.float64)
        if panel.ndim == 1:
            panel = panel.reshape(-1, 1)
        for j in range(panel.shape[1]):
            label = joint_names[j] if j < len(joint_names) else f"J{j}"
            ax.plot(timestamps[: panel.shape[0]], panel[:, j], label=label, color=cmap(j % 10))
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Episode {data['metadata']['episode_id']} ({data['metadata']['label']})")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig_img, ax_img = plt.subplots(figsize=(6, 4))
    ax_img.imshow(data["anchor"])
    ax_img.set_title("Visual Anchor")
    ax_img.axis("off")

    if save_path:
        save_path = save_path.resolve()
        base = save_path if save_path.suffix else save_path.with_suffix(".png")
        signals_path = base.with_name(base.stem + "_signals.png")
        anchor_path = base.with_name(base.stem + "_anchor.png")
        fig.savefig(signals_path, dpi=150)
        fig_img.savefig(anchor_path, dpi=150)
        print(f"已保存图像：{signals_path}, {anchor_path}")
        plt.close(fig)
        plt.close(fig_img)
    else:
        plt.show()


def _print_episode_table(episodes: List[Path]) -> None:
    print("\n可用 Episode 列表：")
    header = f"{'Idx':>4} | {'Episode ID':<28} | {'Label':<12} | {'Duration(s)':>11} | {'Samples':>8}"
    print(header)
    print("-" * len(header))
    for idx, ep_dir in enumerate(episodes):
        try:
            meta = _load_metadata(ep_dir)
            label = meta.get("label", "unknown")
            dur = f"{meta.get('duration_s', 0):.1f}"
            samples = meta.get("num_samples", "-")
        except Exception:
            label = "n/a"
            dur = "n/a"
            samples = "-"
        print(f"{idx:4d} | {ep_dir.name:<28} | {label:<12} | {dur:>11} | {samples:>8}")
    print()


def _interactive_loop(dataset_root: Path, default_prefix: Optional[Path]) -> None:
    episodes = _list_episode_dirs(dataset_root)
    while True:
        _print_episode_table(episodes)
        choice = input("输入索引/目录名（Enter=最新，r=刷新，q=退出）：").strip()
        if choice.lower() in {"q", "quit", "exit"}:
            break
        if choice.lower() == "r":
            episodes = _list_episode_dirs(dataset_root)
            continue
        try:
            if not choice:
                target = episodes[-1]
            elif choice.isdigit():
                idx = int(choice)
                target = episodes[idx]
            else:
                target = dataset_root / choice
                if not target.exists():
                    print(f"❌ 找不到 {target}，请重试。")
                    continue
            data = _load_episode(target)
        except Exception as exc:
            print(f"❌ 加载失败：{exc}")
            continue
        meta = data["metadata"]
        print(json.dumps(meta, indent=2, ensure_ascii=False))

        save_prefix = default_prefix
        if save_prefix is None:
            resp = input("若需保存图像请输入路径前缀（回车跳过）：").strip()
            if resp:
                save_prefix = Path(resp)
        _plot_episode(data, save_prefix)
        cont = input("按 Enter 继续查看，输入 q 退出：").strip().lower()
        if cont in {"q", "quit", "exit"}:
            break
    print("结束交互式浏览。")


def main() -> int:
    parser = argparse.ArgumentParser(description="可视化 Plaintextdataset Episode 数据。")
    parser.add_argument("--dataset-root", type=Path, default=Path("Plaintextdataset"), help="数据集根目录")
    parser.add_argument("--episode", type=str, default=None, help="指定 episode 名称（目录名）")
    parser.add_argument("--index", type=int, default=None, help="按索引选择 episode（与 --episode 互斥）")
    parser.add_argument("--save-prefix", type=Path, default=None, help="保存图像的前缀（无则直接显示）")
    parser.add_argument(
        "--interactive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启用交互式浏览（默认开启，可用 --no-interactive 关闭）。",
    )
    args = parser.parse_args()

    if args.interactive and args.episode is None and args.index is None:
        _interactive_loop(args.dataset_root, args.save_prefix)
        return 0

    if args.episode and args.index is not None:
        raise SystemExit("请在 --episode 与 --index 之间选择其一。")

    episode_dir = _choose_episode(args.dataset_root, args.episode, args.index)
    print(f"使用 episode 目录：{episode_dir}")
    data = _load_episode(episode_dir)
    meta = data["metadata"]
    print(json.dumps(meta, indent=2, ensure_ascii=False))

    _plot_episode(data, args.save_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
