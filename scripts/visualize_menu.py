#!/usr/bin/env python

"""
Interactive terminal menu to visualize collected LeRobot datasets without command-line flags.

Self-contained version placed under learn_PyBullet/ (no package imports required).

Workflow:
1) Pick a local dataset root (auto-detected under ./data or custom path)
2) Pick an episode index
3) Pick which signals to plot
4) Pick joints (by name or index)
5) Set frame range and downsample
6) Choose to show and/or save the figure
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


# ---- plotting core (standalone) -------------------------------------------------

def _load_meta(root: Path) -> dict:
    meta_path = root / "meta" / "info.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta file: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _episode_parquet_path(root: Path, info: dict, episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    data_path_fmt = info["data_path"]  # e.g. data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet
    episode_chunk = episode_index // chunks_size
    rel = data_path_fmt.format(episode_chunk=episode_chunk, episode_index=episode_index)
    p = root / rel
    if not p.exists():
        raise FileNotFoundError(f"Episode parquet not found: {p}")
    return p


def _to_numpy_matrix(col) -> np.ndarray:
    return np.asarray(col.to_pylist(), dtype=float)


def _resolve_joint_indices(info: dict, signal_key: str, joints: list[str] | list[int] | None) -> tuple[list[int], list[str]]:
    features = info.get("features", {})
    entry = features.get(signal_key)
    # Fallback for virtual/custom signals (e.g., derived.velocity)
    if entry is None:
        # Prefer position names if available, else try any feature with names
        fallback_keys = [
            "observation.position",
            "action",
        ] + list(features.keys())
        names = None
        for k in fallback_keys:
            ent = features.get(k)
            if not ent:
                continue
            n = ent.get("names")
            if n:
                names = list(n)
                break
            # Build numeric names from shape
            shape = ent.get("shape") or []
            if shape:
                names = [str(i) for i in range(int(shape[0]))]
                break
        if names is None:
            names = []
    else:
        names = entry.get("names")
        if names is None:
            shape = entry.get("shape") or [0]
            dim = int(shape[0]) if shape else 0
            names = [str(i) for i in range(dim)]

    idxs: list[int]
    if joints is None or len(joints) == 0:
        idxs = list(range(len(names)))
    else:
        if isinstance(joints[0], int):
            idxs = [int(j) for j in joints]
        else:
            name_to_idx = {str(n): i for i, n in enumerate(names)}
            idxs = []
            for j in joints:  # type: ignore[assignment]
                key = str(j)
                if key in name_to_idx:
                    idxs.append(name_to_idx[key])
                else:
                    print(f"警告：关节 '{key}' 在该信号中不可用，已忽略。")
            if not idxs:
                idxs = list(range(len(names)))
    selected_names = [names[i] for i in idxs]
    return idxs, selected_names


def visualize_timeseries(
    root: Path,
    episode_index: int,
    signals: list[str],
    joints: list[str] | list[int] | None = None,
    start: int | None = None,
    end: int | None = None,
    downsample: int = 1,
    save: Path | None = None,
    show: bool = True,
):
    info = _load_meta(root)
    parquet_path = _episode_parquet_path(root, info, episode_index)
    table = pq.read_table(parquet_path)
    cols = {name: table[name] for name in table.column_names if name in signals or name in {"timestamp"}}

    if "timestamp" in cols:
        t = np.asarray(cols["timestamp"].to_pylist(), dtype=float)
        t = t - t[0]
    else:
        fps = float(info.get("fps", 30))
        n = table.num_rows
        t = np.arange(n, dtype=float) / fps

    n = t.shape[0]
    s = start or 0
    e = end if end is not None else n
    s = max(0, min(s, n))
    e = max(s + 1, min(e, n))
    sel = slice(s, e, downsample if max(1, downsample) > 1 else 1)
    t_sel = t[sel]

    num_plots = len(signals)
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 3 * num_plots), sharex=True)
    if num_plots == 1:
        axes = [axes]

    def _ylabel(sig_name: str) -> str:
        units = {
            "observation.position": "Position (deg)",
            "observation.velocity": "Velocity (deg/s)",
            "observation.raw_present_speed": "Velocity (deg/s)",
            "observation.load": "Load (%)",
            "observation.current": "Current (mA)",
            "action": "Action (deg)",
        }
        for key, label in units.items():
            if sig_name == key or sig_name.startswith(key):
                return label
        return "Value"

    for ax, sig in zip(axes, signals):
        if sig not in cols:
            if sig in table.column_names:
                cols[sig] = table[sig]
            else:
                raise KeyError(f"Signal '{sig}' not found in parquet columns: {table.column_names}")
        mat = _to_numpy_matrix(cols[sig])[sel]
        idxs, names = _resolve_joint_indices(info, sig, joints)
        for i, name in zip(idxs, names):
            if i < 0 or i >= mat.shape[1]:
                raise IndexError(f"Joint index {i} out of bounds for signal '{sig}' with D={mat.shape[1]}")
            ax.plot(t_sel, mat[:, i], label=name)
        ax.set_title(sig)
        ax.set_ylabel(_ylabel(sig))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=min(3, len(names)))

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def _scan_local_datasets(base: Path) -> list[Path]:
    datasets: list[Path] = []
    if base.exists() and base.is_dir():
        for p in sorted(base.iterdir()):
            if (p / "meta" / "info.json").exists():
                datasets.append(p)
    return datasets


def _episode_length(root: Path, episode_index: int) -> Optional[int]:
    episodes_jsonl = root / "meta" / "episodes.jsonl"
    if not episodes_jsonl.exists():
        return None
    try:
        with episodes_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if int(obj.get("episode_index", -1)) == int(episode_index):
                    return int(obj.get("length", 0))
    except Exception:
        return None
    return None


def _list_available_signals(root: Path, info: dict, episode_index: int) -> list[str]:
    parquet_path = _episode_parquet_path(root, info, episode_index)
    table = pq.read_table(parquet_path)
    cols = table.column_names
    preferred = [
        "observation.position",
        "observation.velocity",
        "observation.load",
        "observation.current",
        "action",
    ]
    ordered = [c for c in preferred if c in cols]
    ignored = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    rest = [c for c in cols if c not in ignored and c not in ordered]
    signals = ordered + rest
    return signals


def _get_joint_names(info: dict, signal_key: str) -> list[str]:
    features = info.get("features", {})
    entry = features.get(signal_key)
    if entry and entry.get("names"):
        return list(entry["names"])  # type: ignore[index]
    # Fallback to position/action names for virtual signals
    for k in ("observation.position", "action"):
        ent = features.get(k)
        if ent and ent.get("names"):
            return list(ent["names"])  # type: ignore[index]
        if ent and ent.get("shape"):
            dim = int(ent["shape"][0])
            return [str(i) for i in range(dim)]
    return []


def _input_int(prompt: str, min_v: Optional[int] = None, max_v: Optional[int] = None) -> int:
    while True:
        s = input(prompt).strip()
        try:
            v = int(s)
            if min_v is not None and v < min_v:
                print(f"值必须大于等于 {min_v}")
                continue
            if max_v is not None and v > max_v:
                print(f"值必须小于等于 {max_v}")
                continue
            return v
        except ValueError:
            print("请输入整数。")


def _input_choice(prompt: str, choices: list[str]) -> int:
    while True:
        s = input(prompt).strip()
        try:
            idx = int(s)
            if 0 <= idx < len(choices):
                return idx
        except ValueError:
            pass
        print("无效选择，请输入前面显示的数字编号。")


def _input_multi_items(prompt: str, available: list[str]) -> list[str]:
    print(prompt)
    print("请输入逗号分隔的名称（或编号），或输入 'all' / '全部' 选择全部。")
    s = input("> ").strip()
    # 兼容中文分隔符与分号/空格
    for sep in ["，", "、", ";", "；", " "]:
        s = s.replace(sep, ",")
    if s.lower() in {"all", "a", ""} or s in {"全部"}:
        return available
    tokens = [x.strip() for x in s.split(",") if x.strip()]
    out: list[str] = []
    for tok in tokens:
        try:
            i = int(tok)
            if 0 <= i < len(available):
                out.append(available[i])
                continue
        except ValueError:
            pass
        if tok in available:
            out.append(tok)
        else:
            print(f"警告：未识别 '{tok}'，已忽略。")
    seen = set()
    uniq: list[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq if uniq else available


def _confirm(prompt: str, default_yes: bool = True) -> bool:
    suffix = "[Y/n]" if default_yes else "[y/N]"
    while True:
        s = input(f"{prompt} {suffix} ").strip().lower()
        if s == "" and default_yes:
            return True
        if s in {"y", "yes", "是", "s"}:
            return True
        if s in {"n", "no", "否", "f"}:
            return False
        if s == "":
            return False
        print("请输入 y 或 n（或 是/否）。")


def run_menu() -> None:
    print("=== LeRobot 数据集时序可视化（learn_PyBullet）===")

    base = Path("data")
    candidates = _scan_local_datasets(base)
    while True:
        print("\n请选择一个数据集根目录：")
        options = [str(p) for p in candidates] + ["输入自定义路径", "退出"]
        for i, opt in enumerate(options):
            print(f"  {i}. {opt}")
        choice = _input_choice("请输入编号：", options)
        if choice == len(options) - 1:
            return
        if choice == len(options) - 2:
            custom = Path(input("请输入数据集根目录路径：").strip())
            try:
                info = _load_meta(custom)
                root = custom
                break
            except Exception as e:
                print(f"数据集无效：{e}")
                continue
        else:
            root = candidates[choice]
            info = _load_meta(root)
            break

    total_episodes = int(info.get("total_episodes", 0))
    fps = float(info.get("fps", 0))
    print(f"\n已选择数据集：{root}")
    print(f"总 Episode 数：{total_episodes} | FPS：{fps}")

    if total_episodes <= 0:
        print("未在 meta 中找到任何 Episode。")
        return
    ep = _input_int(f"请输入 Episode 索引 [0..{total_episodes-1}]：", 0, total_episodes - 1)
    ep_len = _episode_length(root, ep)
    if ep_len is not None:
        print(f"该 Episode 帧数：{ep_len}")

    signals_available = _list_available_signals(root, info, ep)
    print("\n可用信号：")
    for i, s in enumerate(signals_available):
        print(f"  {i}. {s}")
    signals = _input_multi_items("请选择需要绘制的信号", signals_available)
    print("已选择信号：", ", ".join(signals))

    joint_names = _get_joint_names(info, signals[0])
    print("\n可用关节（按名称）：")
    for i, name in enumerate(joint_names):
        print(f"  {i}. {name}")
    selected_joint_names = _input_multi_items("请选择关节（名称或编号）", joint_names)
    print("已选择关节：", ", ".join(selected_joint_names))

    start = _input_int("起始帧（默认 0）：", 0)
    if ep_len is None:
        end = _input_int("结束帧（不含，0 表示到末尾）：", 0)
        end = None if end == 0 else end
    else:
        end = _input_int(f"结束帧（不含，<= {ep_len}，0 表示到末尾）：", 0, ep_len)
        end = None if end == 0 else end
    downsample = _input_int("下采样步长（>=1）：", 1)

    do_save = _confirm("是否保存图片到文件？", default_yes=False)
    save_path: Optional[Path] = None
    if do_save:
        default_name = f"outputs/episode_{ep}.png"
        sp = input(f"保存路径（默认 {default_name}）：").strip()
        save_path = Path(sp) if sp else Path(default_name)
    do_show = _confirm("是否打开交互式窗口显示？", default_yes=True)

    print("\n正在绘图……")
    try:
        visualize_timeseries(
            root=root,
            episode_index=ep,
            signals=signals,
            joints=selected_joint_names,
            start=start,
            end=end,
            downsample=downsample,
            save=save_path,
            show=do_show,
        )
        print("完成。")
    except Exception as e:
        print(f"绘图失败：{e}")

    if _confirm("返回主菜单？", default_yes=False):
        run_menu()


if __name__ == "__main__":
    run_menu()
