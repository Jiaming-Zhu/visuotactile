import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class GateScoreStats:
    seed_label: str
    split: str
    num_samples: int
    mean: float
    std: float
    min: float
    p10: float
    p50: float
    p90: float
    max: float


def parse_seed_label(run_name: str) -> str:
    match = re.search(r"seed(\d+)", run_name)
    if match:
        return match.group(1)
    return run_name


def sort_seed_labels(labels: Sequence[str]) -> List[str]:
    numeric: List[Tuple[int, str]] = []
    other: List[str] = []
    for label in labels:
        try:
            numeric.append((int(label), label))
        except ValueError:
            other.append(label)
    numeric_sorted = [label for _, label in sorted(numeric, key=lambda x: x[0])]
    other_sorted = sorted(other)
    return numeric_sorted + other_sorted


def find_eval_result(run_dir: Path, split: str) -> Optional[Path]:
    direct = run_dir / f"eval_{split}" / "evaluation_results.json"
    if direct.exists():
        return direct
    return None


def load_gate_scores(result_path: Path) -> np.ndarray:
    obj = json.loads(result_path.read_text(encoding="utf-8"))
    gate_scores = obj.get("gate_scores", [])
    if not isinstance(gate_scores, list) or not gate_scores:
        return np.asarray([], dtype=np.float64)
    return np.asarray(gate_scores, dtype=np.float64)


def compute_stats(seed_label: str, split: str, g: np.ndarray) -> GateScoreStats:
    if g.size == 0:
        return GateScoreStats(
            seed_label=seed_label,
            split=split,
            num_samples=0,
            mean=float("nan"),
            std=float("nan"),
            min=float("nan"),
            p10=float("nan"),
            p50=float("nan"),
            p90=float("nan"),
            max=float("nan"),
        )
    q10, q50, q90 = np.quantile(g, [0.1, 0.5, 0.9]).tolist()
    return GateScoreStats(
        seed_label=seed_label,
        split=split,
        num_samples=int(g.size),
        mean=float(g.mean()),
        std=float(g.std()),
        min=float(g.min()),
        p10=float(q10),
        p50=float(q50),
        p90=float(q90),
        max=float(g.max()),
    )


def plot_boxplot(
    split: str,
    seed_to_scores: Dict[str, np.ndarray],
    out_path: Path,
) -> None:
    seed_labels = sort_seed_labels(list(seed_to_scores.keys()))
    data = [seed_to_scores[s] for s in seed_labels]

    fig, ax = plt.subplots(figsize=(max(10, 0.8 * len(seed_labels) + 4), 5.5), constrained_layout=True)
    ax.boxplot(
        data,
        tick_labels=seed_labels,
        showmeans=True,
        meanline=False,
        whis=(5, 95),
        showfliers=False,
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Gate score distribution by seed ({split})")
    ax.set_xlabel("Seed")
    ax.set_ylabel("gate score (g)")
    ax.grid(alpha=0.2, axis="y", linestyle="--")
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_hist_grid(
    split: str,
    seed_to_scores: Dict[str, np.ndarray],
    out_path: Path,
    bins: int,
) -> None:
    seed_labels = sort_seed_labels(list(seed_to_scores.keys()))
    n = len(seed_labels)
    if n == 0:
        return
    ncols = 3 if n >= 3 else n
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.2 * nrows), constrained_layout=True)
    if nrows == 1 and ncols == 1:
        axes = np.asarray([axes])
    axes = np.asarray(axes).reshape(-1)

    for idx, seed in enumerate(seed_labels):
        ax = axes[idx]
        g = seed_to_scores[seed]
        ax.hist(g, bins=bins, range=(0.0, 1.0), color="#4C72B0", alpha=0.85)
        ax.set_title(f"seed {seed} (n={g.size})")
        ax.set_xlim(0.0, 1.0)
        ax.grid(alpha=0.2, axis="y", linestyle="--")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Gate score histograms ({split})", fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_mean_std(
    split_to_seed_to_scores: Dict[str, Dict[str, np.ndarray]],
    out_path: Path,
) -> None:
    splits = list(split_to_seed_to_scores.keys())
    all_seeds = set()
    for split in splits:
        all_seeds.update(split_to_seed_to_scores[split].keys())
    seed_labels = sort_seed_labels(list(all_seeds))
    if not seed_labels:
        return

    x = np.arange(len(seed_labels))
    width = 0.36 if len(splits) == 2 else 0.8 / max(1, len(splits))

    fig, ax = plt.subplots(figsize=(max(10, 0.7 * len(seed_labels) + 4), 5.5), constrained_layout=True)
    for idx, split in enumerate(sorted(splits)):
        means = []
        stds = []
        for seed in seed_labels:
            g = split_to_seed_to_scores[split].get(seed, np.asarray([], dtype=np.float64))
            means.append(float(np.mean(g)) if g.size else float("nan"))
            stds.append(float(np.std(g)) if g.size else float("nan"))
        offset = (idx - (len(splits) - 1) / 2) * width
        ax.bar(x + offset, means, width=width, label=split, alpha=0.85)
        ax.errorbar(x + offset, means, yerr=stds, fmt="none", ecolor="black", elinewidth=1.0, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(seed_labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Gate score mean ± std by seed")
    ax.set_xlabel("Seed")
    ax.set_ylabel("gate score (g)")
    ax.grid(alpha=0.2, axis="y", linestyle="--")
    ax.legend()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_stats_csv(stats: Sequence[GateScoreStats], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "seed",
                "num_samples",
                "mean",
                "std",
                "min",
                "p10",
                "p50",
                "p90",
                "max",
            ],
        )
        writer.writeheader()
        for s in stats:
            writer.writerow(
                {
                    "split": s.split,
                    "seed": s.seed_label,
                    "num_samples": s.num_samples,
                    "mean": s.mean,
                    "std": s.std,
                    "min": s.min,
                    "p10": s.p10,
                    "p50": s.p50,
                    "p90": s.p90,
                    "max": s.max,
                }
            )


def run(args: argparse.Namespace) -> None:
    runs_root = Path(args.runs_root)
    if not runs_root.is_dir():
        raise FileNotFoundError(f"--runs_root is not a directory: {runs_root}")

    split_names = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not split_names:
        raise ValueError("--splits cannot be empty")

    output_dir = Path(args.output_dir) if args.output_dir else (runs_root / "gate_score_viz")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = [p for p in runs_root.glob(args.run_glob) if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run dirs matched: {runs_root}/{args.run_glob}")

    split_to_seed_to_scores: Dict[str, Dict[str, np.ndarray]] = {split: {} for split in split_names}
    stats: List[GateScoreStats] = []
    missing: List[str] = []

    for run_dir in sorted(run_dirs):
        seed_label = parse_seed_label(run_dir.name)
        for split in split_names:
            result_path = find_eval_result(run_dir, split)
            if result_path is None:
                missing.append(f"{run_dir.name}/eval_{split}/evaluation_results.json")
                continue
            g = load_gate_scores(result_path)
            if g.size == 0:
                missing.append(str(result_path))
                continue
            split_to_seed_to_scores[split][seed_label] = g
            stats.append(compute_stats(seed_label, split, g))

    # Drop splits with no data.
    split_to_seed_to_scores = {k: v for k, v in split_to_seed_to_scores.items() if v}
    if not split_to_seed_to_scores:
        raise RuntimeError("No gate_scores found in matched runs.")

    stats = sorted(stats, key=lambda s: (s.split, int(s.seed_label) if s.seed_label.isdigit() else 1_000_000, s.seed_label))

    write_stats_csv(stats, output_dir / "gate_score_stats.csv")
    (output_dir / "gate_score_stats.json").write_text(
        json.dumps([s.__dict__ for s in stats], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for split, seed_to_scores in split_to_seed_to_scores.items():
        plot_boxplot(split, seed_to_scores, output_dir / f"gate_score_boxplot_{split}.png")
        plot_hist_grid(split, seed_to_scores, output_dir / f"gate_score_hist_{split}.png", bins=args.bins)

    plot_mean_std(split_to_seed_to_scores, output_dir / "gate_score_mean_std.png")

    if missing:
        (output_dir / "missing_gate_scores.txt").write_text("\n".join(missing), encoding="utf-8")

    print(f"Done. Saved to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize and visualize per-sample gating scores (g) from train_fusion_gating eval outputs."
    )
    parser.add_argument(
        "--runs_root",
        type=str,
        default="/home/martina/Y3_Project/visuotactile/outputs/fusion/gating",
        help="Root directory that contains per-seed run folders.",
    )
    parser.add_argument(
        "--run_glob",
        type=str,
        default="fusion_gating_entropy_seed*",
        help="Glob under runs_root to pick runs (e.g. fusion_gating_seed*).",
    )
    parser.add_argument("--splits", type=str, default="test,ood_test")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--bins", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
