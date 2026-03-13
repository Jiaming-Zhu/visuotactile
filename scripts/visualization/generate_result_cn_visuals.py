import argparse
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise ImportError("matplotlib is required to generate result_cn visuals") from exc


ROOT = Path("/home/martina/Y3_Project/visuotactile")
DEFAULT_MARKDOWN = ROOT / "docs/result_cn.md"
DEFAULT_FIGURE_DIR = ROOT / "docs/figures/result_cn"
AUTO_MARKER_PREFIX = "AUTO-GENERATED RESULT_CN FIGURE"
PALETTE = (
    "#2563eb",
    "#ef4444",
    "#10b981",
    "#f59e0b",
    "#8b5cf6",
    "#14b8a6",
)

PUBLICATION_MODEL_STYLES = {
    "B": {"label": "B  Vision", "facecolor": "#d4d4d4", "edgecolor": "#4b5563", "hatch": ""},
    "C": {"label": "C  Tactile", "facecolor": "#a3a3a3", "edgecolor": "#374151", "hatch": ""},
    "A": {"label": "A  Fusion", "facecolor": "#737373", "edgecolor": "#1f2937", "hatch": ""},
    "G1": {"label": "G1  Entropy", "facecolor": "#8ecae6", "edgecolor": "#1d4ed8", "hatch": ""},
    "G2": {"label": "G2  Aux + Gate", "facecolor": "#b91c1c", "edgecolor": "#7f1d1d", "hatch": "///"},
}
PUBLICATION_TASK_LABELS = {
    "Mass": "Mass",
    "Stiffness": "Stiffness",
    "Material": "Material",
    "Avg Acc": "Average",
}


TABLE_CONFIG: Dict[int, Dict[str, str]] = {
    3: {
        "filename": "table_03_baseline_ood_comparison.png",
        "title": "Baseline OOD Comparison",
        "alt": "基础消融 OOD 对比可视化",
        "caption": "基础消融 OOD 对比。展示纯视觉、纯触觉与基础融合在三个任务及平均准确率上的差异。",
    },
    4: {
        "filename": "table_04_gating_core_comparison.png",
        "title": "Gating Improvement on OOD",
        "alt": "门控与辅助监督核心对比可视化",
        "caption": "基础融合、G1 与 G2 的 OOD 对比。左侧展示任务准确率与平均准确率，右侧展示方差和平均门控值。",
    },
    5: {
        "filename": "table_05_g1_seed_breakdown.png",
        "title": "G1 Seed Breakdown",
        "alt": "G1 多随机种子结果可视化",
        "caption": "G1 在不同随机种子上的 OOD 结果变化，突出其较大的性能波动。",
    },
    6: {
        "filename": "table_06_g2_seed_breakdown.png",
        "title": "G2 Seed Breakdown",
        "alt": "G2 多随机种子结果可视化",
        "caption": "G2 在不同随机种子上的 OOD 结果变化，展示辅助监督带来的稳定性提升。",
    },
    7: {
        "filename": "table_07_vision_seed_breakdown.png",
        "title": "Vision-Only Seed Breakdown",
        "alt": "纯视觉模型多随机种子结果可视化",
        "caption": "纯视觉模型在 OOD 上的多随机种子结果，展示视觉 shortcut 导致的整体失效。",
    },
    8: {
        "filename": "table_08_tactile_seed_breakdown.png",
        "title": "Tactile-Only Seed Breakdown",
        "alt": "纯触觉模型多随机种子结果可视化",
        "caption": "纯触觉模型在 OOD 上的多随机种子结果，展示其稳定但仍有限的泛化能力。",
    },
    9: {
        "filename": "table_09_fusion_seed_breakdown.png",
        "title": "Fusion Seed Breakdown",
        "alt": "基础融合模型多随机种子结果可视化",
        "caption": "基础融合模型在 OOD 上的多随机种子结果，展示其优于单模态但仍存在明显方差。",
    },
    10: {
        "filename": "table_10_g3_single_run_comparison.png",
        "title": "G3 Single-Run Test vs OOD",
        "alt": "G3 单次最佳运行结果可视化",
        "caption": "G3 单次最佳运行在 test 与 ood_test 上的结果对比，包含平均门控值。",
    },
    11: {
        "filename": "table_11_g3_prefix_curve_seed42.png",
        "title": "G3 Prefix Curve (Seed 42)",
        "alt": "G3 单种子前缀曲线可视化",
        "caption": "G3 在 seed 42 下的前缀曲线。左侧展示任务准确率与平均准确率，右侧展示门控值随前缀变化。",
    },
    12: {
        "filename": "table_12_g3_multiseed_summary.png",
        "title": "G3 Multi-Seed Summary",
        "alt": "G3 多随机种子汇总可视化",
        "caption": "G3 在 test 与 ood_test 上的多随机种子汇总结果，包含平均门控值。",
    },
}


@dataclass
class NumericValue:
    mean: Optional[float]
    std: Optional[float]
    is_percent: bool
    raw: str


@dataclass
class MarkdownTable:
    index: int
    headings: List[str]
    start_line: int
    end_line: int
    headers: List[str]
    rows: List[List[str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate visual summaries for result_cn.md tables")
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--figure_dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def strip_markdown(text: str) -> str:
    cleaned = text.strip()
    for token in ("**", "__", "`", "*"):
        cleaned = cleaned.replace(token, "")
    cleaned = cleaned.replace("\\|", "|")
    return cleaned.strip()


def slugify(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", strip_markdown(text))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = ascii_text.lower()
    ascii_text = re.sub(r"[^a-z0-9]+", "_", ascii_text).strip("_")
    return ascii_text or "figure"


def clean_auto_blocks(text: str) -> str:
    lines = text.splitlines()
    cleaned: List[str] = []
    skip = False
    for line in lines:
        if line.startswith(f"<!-- {AUTO_MARKER_PREFIX}:") and line.endswith("START -->"):
            skip = True
            continue
        if line.startswith(f"<!-- {AUTO_MARKER_PREFIX}:") and line.endswith("END -->"):
            skip = False
            continue
        if not skip:
            cleaned.append(line)
    return "\n".join(cleaned).rstrip() + "\n"


def parse_table_row(line: str) -> List[str]:
    parts = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return parts


def extract_tables(text: str) -> Tuple[List[str], List[MarkdownTable]]:
    lines = text.splitlines()
    headings: List[str] = []
    tables: List[MarkdownTable] = []
    index = 0
    line_no = 0
    while line_no < len(lines):
        line = lines[line_no]
        heading_match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            headings = headings[: level - 1] + [title]

        is_table = (
            line.startswith("|")
            and line_no + 1 < len(lines)
            and re.match(r"^\|?\s*[-:| ]+\|?\s*$", lines[line_no + 1])
        )
        if is_table:
            index += 1
            start = line_no
            raw_rows = [line, lines[line_no + 1]]
            line_no += 2
            while line_no < len(lines) and lines[line_no].startswith("|"):
                raw_rows.append(lines[line_no])
                line_no += 1
            headers = parse_table_row(raw_rows[0])
            rows = [parse_table_row(raw_row) for raw_row in raw_rows[2:]]
            tables.append(
                MarkdownTable(
                    index=index,
                    headings=headings.copy(),
                    start_line=start,
                    end_line=line_no - 1,
                    headers=headers,
                    rows=rows,
                )
            )
            continue
        line_no += 1
    return lines, tables


def parse_numeric_value(text: str) -> NumericValue:
    raw = strip_markdown(text)
    if raw in {"", "-", "—"}:
        return NumericValue(mean=None, std=None, is_percent=False, raw=raw)

    is_percent = "%" in raw
    normalized = raw.replace("%", "").strip()
    if "±" in normalized:
        left, right = [part.strip() for part in normalized.split("±", 1)]
        if left == "":
            try:
                return NumericValue(mean=None, std=float(right), is_percent=is_percent, raw=raw)
            except ValueError:
                return NumericValue(mean=None, std=None, is_percent=is_percent, raw=raw)
        try:
            return NumericValue(mean=float(left), std=float(right), is_percent=is_percent, raw=raw)
        except ValueError:
            return NumericValue(mean=None, std=None, is_percent=is_percent, raw=raw)
    try:
        return NumericValue(mean=float(normalized), std=None, is_percent=is_percent, raw=raw)
    except ValueError:
        return NumericValue(mean=None, std=None, is_percent=is_percent, raw=raw)


def translate_label(text: str) -> str:
    cleaned = strip_markdown(text)
    overrides = {
        "B. 纯视觉": "B",
        "C. 纯触觉": "C",
        "A. 融合(基础)": "A",
        "A. 融合 (基础)": "A",
        "G1. 门控 (仅 Entropy)": "G1",
        "G2. 门控 + 辅助约束": "G2",
        "质量 Acc (3类)": "Mass",
        "质量 Acc": "Mass",
        "刚度 Acc": "Stiffness",
        "材质 Acc": "Material",
        "平均 Acc": "Avg Acc",
        "平均 Gate": "Avg Gate",
        "方差 (Std)": "Std",
        "Mass (3类)": "Mass",
        "Split": "Split",
        "前缀比例": "Prefix",
        "test": "Test",
        "ood_test": "OOD",
    }
    if cleaned in overrides:
        return overrides[cleaned]

    lowered = cleaned.lower()
    if "average accuracy" in lowered or "平均 acc" in lowered or "avg acc" in lowered:
        return "Avg Acc"
    if "average gate" in lowered or "平均 gate" in lowered or "avg gate" in lowered:
        return "Avg Gate"
    if "mass" in lowered or "质量" in cleaned:
        return "Mass"
    if "stiffness" in lowered or "刚度" in cleaned:
        return "Stiffness"
    if "material" in lowered or "材质" in cleaned:
        return "Material"
    if "std" in lowered or "方差" in cleaned:
        return "Std"
    if lowered.startswith("seed "):
        return cleaned

    ascii_label = unicodedata.normalize("NFKD", cleaned).encode("ascii", "ignore").decode("ascii").strip()
    return ascii_label or cleaned


def is_target_table(table: MarkdownTable) -> bool:
    return table.index in TABLE_CONFIG


def group_columns_by_scale(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> Tuple[List[int], List[int]]:
    percent_like: List[int] = []
    unit_like: List[int] = []
    for col_idx in range(1, len(headers)):
        values = [parse_numeric_value(row[col_idx]).mean for row in rows if col_idx < len(row)]
        numeric_values = [value for value in values if value is not None]
        if not numeric_values:
            continue
        if all(0.0 <= value <= 1.2 for value in numeric_values):
            unit_like.append(col_idx)
        else:
            percent_like.append(col_idx)
    return percent_like, unit_like


def group_rows_by_scale(rows: Sequence[Sequence[str]], start_col: int, end_col: int) -> Tuple[List[int], List[int]]:
    percent_like: List[int] = []
    unit_like: List[int] = []
    for row_idx, row in enumerate(rows):
        numeric_values = []
        for col_idx in range(start_col, end_col):
            if col_idx >= len(row):
                continue
            value = parse_numeric_value(row[col_idx]).mean
            if value is not None:
                numeric_values.append(value)
        if not numeric_values:
            continue
        if all(0.0 <= value <= 1.2 for value in numeric_values):
            unit_like.append(row_idx)
        else:
            percent_like.append(row_idx)
    return percent_like, unit_like


def format_legend_suffix(text: str) -> str:
    value = parse_numeric_value(text)
    if value.mean is None:
        return ""
    if value.std is None:
        return f" ({value.mean:.2f})"
    return f" ({value.mean:.2f} ± {value.std:.2f})"


def configure_axis(ax: plt.Axes, unit_scale: bool) -> None:
    if unit_scale:
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
    else:
        ax.set_ylim(0, 105)
        ax.set_ylabel("Value (%)")
    ax.grid(axis="y", alpha=0.25)


def create_axes(num_groups: int, figure_width: float = 12.0) -> Tuple[plt.Figure, List[plt.Axes]]:
    if num_groups == 1:
        fig, ax = plt.subplots(figsize=(figure_width, 5.4))
        return fig, [ax]
    fig, axes = plt.subplots(1, num_groups, figsize=(figure_width, 5.4), width_ratios=[3, 1.8])
    return fig, list(axes)


def plot_grouped_bars(table: MarkdownTable, output_path: Path, title: str, dpi: int) -> None:
    row_labels = [translate_label(row[0]) for row in table.rows]
    percent_cols, unit_cols = group_columns_by_scale(table.headers, table.rows)
    col_groups = [percent_cols]
    if unit_cols:
        col_groups.append(unit_cols)

    fig, axes = create_axes(len(col_groups))
    for ax, col_group in zip(axes, col_groups):
        if not col_group:
            ax.axis("off")
            continue
        x = np.arange(len(row_labels))
        width = min(0.75 / max(len(col_group), 1), 0.22)
        unit_scale = col_group == unit_cols and bool(unit_cols)
        for offset_idx, col_idx in enumerate(col_group):
            header = translate_label(table.headers[col_idx])
            means = []
            stds = []
            for row in table.rows:
                numeric = parse_numeric_value(row[col_idx])
                means.append(np.nan if numeric.mean is None else numeric.mean)
                stds.append(0.0 if numeric.std is None else numeric.std)
            offset = (offset_idx - (len(col_group) - 1) / 2.0) * width
            ax.bar(
                x + offset,
                means,
                width=width,
                yerr=stds if any(value > 0 for value in stds) else None,
                capsize=4,
                label=header,
                color=PALETTE[offset_idx % len(PALETTE)],
                edgecolor="white",
                linewidth=0.8,
                alpha=0.92,
            )
        configure_axis(ax, unit_scale=unit_scale)
        ax.set_xticks(x)
        ax.set_xticklabels(row_labels)
        ax.legend(frameon=False, fontsize=9, ncol=2 if len(col_group) > 2 else 1)
        ax.set_title("Accuracy / Std" if not unit_scale else "Gate")

    fig.suptitle(title)
    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.09, right=0.98, hspace=0.05)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _broken_axis_limits(values: Sequence[float]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    min_value = float(np.nanmin(values))
    max_value = float(np.nanmax(values))
    if min_value < 30.0 and max_value > 80.0:
        return (70.0, 101.0), (0.0, 30.0)
    return (98.0, 100.6), (88.0, 96.0)


def _draw_axis_break(ax_top: plt.Axes, ax_bottom: plt.Axes) -> None:
    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(labeltop=False, bottom=False)
    ax_bottom.xaxis.tick_bottom()

    kwargs = dict(color="#111827", clip_on=False, linewidth=1.1)
    d = 0.012
    ax_top.plot((-d, +d), (-d, +d), transform=ax_top.transAxes, **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), transform=ax_top.transAxes, **kwargs)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), transform=ax_bottom.transAxes, **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_bottom.transAxes, **kwargs)


def _publication_table_payload(
    table: MarkdownTable,
    std_overrides: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[List[str], List[str], Dict[str, List[float]], Dict[str, List[float]]]:
    metric_cols = []
    for col_idx, header in enumerate(table.headers[1:], start=1):
        translated = translate_label(header)
        if translated in PUBLICATION_TASK_LABELS:
            metric_cols.append((col_idx, PUBLICATION_TASK_LABELS[translated]))

    if not metric_cols:
        raise ValueError(f"No publication-style metric columns found in table {table.index}")

    std_col = None
    for col_idx, header in enumerate(table.headers):
        if translate_label(header) == "Std":
            std_col = col_idx
            break

    task_labels = [label for _, label in metric_cols]
    model_labels: List[str] = []
    means: Dict[str, List[float]] = {}
    stds: Dict[str, List[float]] = {}

    for row in table.rows:
        model_key = translate_label(row[0])
        model_labels.append(model_key)
        means[model_key] = []
        stds[model_key] = []

        fallback_std = None
        if std_col is not None and std_col < len(row):
            std_value = parse_numeric_value(row[std_col])
            fallback_std = std_value.std if std_value.std is not None else std_value.mean

        for col_idx, _ in metric_cols:
            task_label = PUBLICATION_TASK_LABELS[translate_label(table.headers[col_idx])]
            numeric = parse_numeric_value(row[col_idx])
            means[model_key].append(np.nan if numeric.mean is None else numeric.mean)
            std_value = numeric.std
            if std_value is None and std_overrides is not None:
                std_value = std_overrides.get(model_key, {}).get(task_label)
            if std_value is None and fallback_std is not None:
                std_value = fallback_std
            stds[model_key].append(0.0 if std_value is None else std_value)

    return task_labels, model_labels, means, stds


def build_publication_std_overrides(tables: Sequence[MarkdownTable]) -> Dict[str, Dict[str, float]]:
    overrides: Dict[str, Dict[str, float]] = {}

    for table in tables:
        if table.index == 3:
            for row in table.rows:
                model_key = translate_label(row[0])
                overrides.setdefault(model_key, {})
                for col_idx, header in enumerate(table.headers[1:], start=1):
                    metric_label = translate_label(header)
                    if metric_label not in PUBLICATION_TASK_LABELS:
                        continue
                    numeric = parse_numeric_value(row[col_idx])
                    if numeric.std is not None:
                        overrides[model_key][PUBLICATION_TASK_LABELS[metric_label]] = float(numeric.std)
        elif table.index in {5, 6, 9}:
            model_key = {5: "G1", 6: "G2", 9: "A"}[table.index]
            overrides.setdefault(model_key, {})
            summary_col = len(table.headers) - 1
            for row in table.rows:
                metric_label = translate_label(row[0])
                if metric_label not in PUBLICATION_TASK_LABELS:
                    continue
                numeric = parse_numeric_value(row[summary_col])
                std_value = numeric.std if numeric.std is not None else numeric.mean
                if std_value is not None:
                    overrides[model_key][PUBLICATION_TASK_LABELS[metric_label]] = float(std_value)

    return overrides


def plot_publication_broken_bars(
    table: MarkdownTable,
    output_path: Path,
    title: str,
    dpi: int,
    std_overrides: Optional[Dict[str, Dict[str, float]]] = None,
) -> None:
    task_labels, model_labels, means, stds = _publication_table_payload(table, std_overrides=std_overrides)

    all_values: List[float] = []
    for model_key in model_labels:
        all_values.extend([value for value in means[model_key] if not np.isnan(value)])
    top_ylim, bottom_ylim = _broken_axis_limits(all_values)

    x = np.arange(len(task_labels))
    width = min(0.78 / max(len(model_labels), 1), 0.22)

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(11.5, 6.8),
        gridspec_kw={"height_ratios": [2.8, 1.2], "hspace": 0.05},
    )

    for idx, model_key in enumerate(model_labels):
        style = PUBLICATION_MODEL_STYLES.get(
            model_key,
            {"label": model_key, "facecolor": PALETTE[idx % len(PALETTE)], "edgecolor": "#1f2937", "hatch": ""},
        )
        offset = (idx - (len(model_labels) - 1) / 2.0) * width
        y = np.asarray(means[model_key], dtype=np.float64)
        yerr = np.asarray(stds[model_key], dtype=np.float64)

        bar_kwargs = {
            "width": width,
            "color": style["facecolor"],
            "edgecolor": style["edgecolor"],
            "linewidth": 1.0,
            "alpha": 0.96,
            "hatch": style["hatch"],
            "label": style["label"],
            "zorder": 3,
        }
        ax_top.bar(x + offset, y, **bar_kwargs)
        ax_bottom.bar(x + offset, y, **bar_kwargs)
        ax_top.errorbar(
            x + offset,
            y,
            yerr=yerr,
            fmt="none",
            ecolor="black",
            elinewidth=1.0,
            capsize=3,
            capthick=1.0,
            zorder=4,
        )
        ax_bottom.errorbar(
            x + offset,
            y,
            yerr=yerr,
            fmt="none",
            ecolor="black",
            elinewidth=1.0,
            capsize=3,
            capthick=1.0,
            zorder=4,
        )

    ax_top.set_ylim(*top_ylim)
    ax_bottom.set_ylim(*bottom_ylim)
    ax_top.grid(axis="y", alpha=0.22, linestyle="--", linewidth=0.8)
    ax_bottom.grid(axis="y", alpha=0.22, linestyle="--", linewidth=0.8)
    ax_top.set_title(title, pad=10)
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(task_labels)
    ax_bottom.set_xlabel("Evaluation Task")
    ax_top.set_ylabel("Accuracy (%)")
    ax_bottom.set_ylabel("Accuracy (%)")

    _draw_axis_break(ax_top, ax_bottom)
    ax_top.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=min(len(model_labels), 5),
        frameon=False,
        fontsize=9,
        columnspacing=1.2,
        handlelength=1.8,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_seed_breakdown(table: MarkdownTable, output_path: Path, title: str, dpi: int) -> None:
    seed_headers = [strip_markdown(header) for header in table.headers[1:-1]]
    percent_rows, unit_rows = group_rows_by_scale(table.rows, 1, len(table.headers) - 1)
    row_groups = [percent_rows]
    if unit_rows:
        row_groups.append(unit_rows)

    fig, axes = create_axes(len(row_groups))
    x = np.arange(len(seed_headers))
    for ax, row_group in zip(axes, row_groups):
        if not row_group:
            ax.axis("off")
            continue
        unit_scale = row_group == unit_rows and bool(unit_rows)
        for idx, row_idx in enumerate(row_group):
            row = table.rows[row_idx]
            values = []
            for col_idx in range(1, len(table.headers) - 1):
                numeric = parse_numeric_value(row[col_idx])
                values.append(np.nan if numeric.mean is None else numeric.mean)
            legend = translate_label(row[0]) + format_legend_suffix(row[-1])
            ax.plot(
                x,
                values,
                marker="o",
                linewidth=2.0,
                markersize=5,
                label=legend,
                color=PALETTE[idx % len(PALETTE)],
            )
        configure_axis(ax, unit_scale=unit_scale)
        ax.set_xticks(x)
        ax.set_xticklabels(seed_headers)
        ax.legend(frameon=False, fontsize=8)
        ax.set_title("Accuracy" if not unit_scale else "Gate")

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_prefix_curve(table: MarkdownTable, output_path: Path, title: str, dpi: int) -> None:
    x_values = []
    for row in table.rows:
        numeric = parse_numeric_value(row[0])
        x_values.append(np.nan if numeric.mean is None else numeric.mean)

    percent_cols, unit_cols = group_columns_by_scale(table.headers, table.rows)
    col_groups = [percent_cols]
    if unit_cols:
        col_groups.append(unit_cols)

    fig, axes = create_axes(len(col_groups))
    for ax, col_group in zip(axes, col_groups):
        if not col_group:
            ax.axis("off")
            continue
        unit_scale = col_group == unit_cols and bool(unit_cols)
        for idx, col_idx in enumerate(col_group):
            series_name = translate_label(table.headers[col_idx])
            y_values = []
            for row in table.rows:
                numeric = parse_numeric_value(row[col_idx])
                y_values.append(np.nan if numeric.mean is None else numeric.mean)
            ax.plot(
                x_values,
                y_values,
                marker="o",
                linewidth=2.0,
                markersize=5,
                label=series_name,
                color=PALETTE[idx % len(PALETTE)],
            )
        configure_axis(ax, unit_scale=unit_scale)
        ax.set_xlabel("Prefix Ratio")
        ax.set_xticks(x_values)
        ax.legend(frameon=False, fontsize=9)
        ax.set_title("Accuracy" if not unit_scale else "Gate")

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_table(
    table: MarkdownTable,
    output_path: Path,
    title: str,
    dpi: int,
    std_overrides: Optional[Dict[str, Dict[str, float]]] = None,
) -> None:
    if table.index in {3, 4}:
        plot_publication_broken_bars(table, output_path, title, dpi, std_overrides=std_overrides)
        return
    if any(header.startswith("Seed ") for header in table.headers[1:]):
        plot_seed_breakdown(table, output_path, title, dpi)
        return
    if translate_label(table.headers[0]) == "Prefix":
        plot_prefix_curve(table, output_path, title, dpi)
        return
    plot_grouped_bars(table, output_path, title, dpi)


def inject_visuals(lines: List[str], tables: Sequence[MarkdownTable], markdown_path: Path, figure_dir: Path) -> str:
    by_start_line = {table.start_line: table for table in tables if is_target_table(table)}
    result_lines: List[str] = []
    line_no = 0
    while line_no < len(lines):
        table = by_start_line.get(line_no)
        if table is None:
            result_lines.append(lines[line_no])
            line_no += 1
            continue

        for idx in range(table.start_line, table.end_line + 1):
            result_lines.append(lines[idx])

        config = TABLE_CONFIG[table.index]
        rel_path = figure_dir.relative_to(markdown_path.parent)
        image_path = (rel_path / config["filename"]).as_posix()
        marker_slug = slugify(config["filename"])
        result_lines.append("")
        result_lines.append(f"<!-- {AUTO_MARKER_PREFIX}: {marker_slug} START -->")
        result_lines.append(f"![{config['alt']}]({image_path})")
        result_lines.append("")
        result_lines.append(f"*图：{config['caption']}*")
        result_lines.append(f"<!-- {AUTO_MARKER_PREFIX}: {marker_slug} END -->")
        result_lines.append("")
        line_no = table.end_line + 1
    return "\n".join(result_lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    markdown_path = args.markdown.resolve()
    figure_dir = args.figure_dir.resolve()
    figure_dir.mkdir(parents=True, exist_ok=True)

    original_text = markdown_path.read_text(encoding="utf-8")
    cleaned_text = clean_auto_blocks(original_text)
    lines, tables = extract_tables(cleaned_text)
    std_overrides = build_publication_std_overrides(tables)

    for table in tables:
        if not is_target_table(table):
            continue
        config = TABLE_CONFIG[table.index]
        render_table(
            table=table,
            output_path=figure_dir / config["filename"],
            title=config["title"],
            dpi=args.dpi,
            std_overrides=std_overrides,
        )

    updated_markdown = inject_visuals(lines, tables, markdown_path, figure_dir)
    markdown_path.write_text(updated_markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
