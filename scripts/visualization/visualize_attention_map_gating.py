import argparse
import json
import re
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from train_fusion_gating import FusionModel, RoboticGraspDataset, resolve_device, set_seed

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


class AnalysisRoboticGraspDataset(RoboticGraspDataset):
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = super().__getitem__(idx)
        sample = self.samples[idx]
        item["obj_class"] = sample.img_path.parent.parent.name
        item["episode_name"] = sample.img_path.parent.name
        return item


def build_loader(split_dir: Path, batch_size: int, max_tactile_len: int, num_workers: int) -> DataLoader:
    dataset = AnalysisRoboticGraspDataset(split_dir=split_dir, max_tactile_len=max_tactile_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


@contextmanager
def safe_sdp_context():
    if hasattr(torch.nn, "attention") and hasattr(torch.nn.attention, "sdpa_kernel"):
        backends = [torch.nn.attention.SDPBackend.MATH]
        with torch.nn.attention.sdpa_kernel(backends=backends):
            yield
        return

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            yield
        return

    with nullcontext():
        yield


class AttentionExtractor:
    def __init__(self, model: FusionModel) -> None:
        self.model = model
        self.attn_weights_store: List[torch.Tensor] = []
        self.handles = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def attn_pre_hook(_module, args, kwargs):
            kw = dict(kwargs)
            kw["need_weights"] = True
            kw["average_attn_weights"] = False
            return args, kw

        def attn_post_hook(_module, _input, output):
            if output[1] is None:
                raise RuntimeError("Attention weights are None. Try forcing math SDPA backend.")
            self.attn_weights_store.append(output[1].detach().cpu())

        for layer in self.model.transformer_encoder.layers:
            self.handles.append(layer.self_attn.register_forward_pre_hook(attn_pre_hook, with_kwargs=True))
            self.handles.append(layer.self_attn.register_forward_hook(attn_post_hook))

    def clear(self) -> None:
        self.attn_weights_store.clear()

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def load_model(checkpoint_path: Path, args: argparse.Namespace, device: torch.device) -> FusionModel:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})
    model = FusionModel(
        fusion_dim=cfg.get("fusion_dim", args.fusion_dim),
        num_heads=cfg.get("num_heads", args.num_heads),
        dropout=cfg.get("dropout", args.dropout),
        num_layers=cfg.get("num_layers", args.num_layers),
        freeze_visual=True,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def resolve_layer_index(layer_index: int, num_layers: int) -> int:
    idx = layer_index if layer_index >= 0 else num_layers + layer_index
    if idx < 0 or idx >= num_layers:
        raise ValueError(f"Invalid layer_index={layer_index}, num_layers={num_layers}")
    return idx


def sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return safe or "unnamed"


def _accumulate_variable_length(
    sum_vector: torch.Tensor | None,
    count_vector: torch.Tensor | None,
    value_vector: torch.Tensor,
    weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    vec = value_vector.double()
    vec_len = vec.size(0)
    if sum_vector is None or count_vector is None:
        sum_vector = torch.zeros_like(vec)
        count_vector = torch.zeros_like(vec)
    else:
        cur_len = sum_vector.size(0)
        if vec_len > cur_len:
            extra = vec_len - cur_len
            sum_vector = torch.cat([sum_vector, torch.zeros(extra, dtype=sum_vector.dtype)], dim=0)
            count_vector = torch.cat([count_vector, torch.zeros(extra, dtype=count_vector.dtype)], dim=0)
        elif vec_len < cur_len:
            vec = torch.cat([vec, torch.zeros(cur_len - vec_len, dtype=vec.dtype)], dim=0)
            vec_len = cur_len

    sum_vector += vec
    count_vector[:vec_len] += float(weight)
    return sum_vector, count_vector


@torch.no_grad()
def collect_split_attention(
    model: FusionModel,
    loader: DataLoader,
    device: torch.device,
    layer_index: int,
    max_samples: int = 0,
) -> Dict[str, object]:
    model.eval()
    num_layers = len(model.transformer_encoder.layers)
    selected_layer = resolve_layer_index(layer_index, num_layers)

    extractor = AttentionExtractor(model)
    total_samples = 0
    sum_vector = None
    count_vector = None
    layer_share_sum = torch.zeros((num_layers, 3), dtype=torch.float64)
    class_vector_sum: Dict[str, torch.Tensor] = {}
    class_vector_count: Dict[str, torch.Tensor] = {}
    class_layer_share_sum: Dict[str, torch.Tensor] = {}
    class_sample_count: Dict[str, int] = defaultdict(int)
    sample_entries: List[Dict[str, object]] = []

    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc="collect attention", leave=False)

    try:
        for batch in iterator:
            images = batch["image"].to(device)
            tactile = batch["tactile"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            obj_classes = list(batch["obj_class"])
            episode_names = list(batch["episode_name"])

            if max_samples > 0 and total_samples >= max_samples:
                break
            orig_bsz = images.size(0)
            if max_samples > 0 and total_samples + orig_bsz > max_samples:
                keep = max_samples - total_samples
                images = images[:keep]
                tactile = tactile[:keep]
                padding_mask = padding_mask[:keep]
                obj_classes = obj_classes[:keep]
                episode_names = episode_names[:keep]
            bsz = images.size(0)
            if bsz == 0:
                break

            extractor.clear()
            with safe_sdp_context():
                _ = model(images, tactile, padding_mask=padding_mask)

            if len(extractor.attn_weights_store) != num_layers:
                raise RuntimeError(
                    f"Expected {num_layers} attention maps, got {len(extractor.attn_weights_store)}."
                )

            per_layer_per_sample: List[torch.Tensor] = []
            for layer_idx, attn in enumerate(extractor.attn_weights_store):
                cls_attn = attn[:, :, 0, :]
                cls_share = cls_attn[:, :, 0:1].sum(dim=-1)
                vis_share = cls_attn[:, :, 1:50].sum(dim=-1)
                tac_share = cls_attn[:, :, 50:].sum(dim=-1)
                total = (cls_share + vis_share + tac_share).clamp(min=1e-12)
                ratios = torch.stack([cls_share / total, vis_share / total, tac_share / total], dim=-1)
                per_sample_ratios = ratios.mean(dim=1).double()
                per_layer_per_sample.append(per_sample_ratios)
                layer_share_sum[layer_idx] += per_sample_ratios.sum(dim=0)

            target = extractor.attn_weights_store[selected_layer]
            cls_to_all = target[:, :, 0, :].mean(dim=1).double()
            for sample_idx in range(bsz):
                vec = cls_to_all[sample_idx]
                sum_vector, count_vector = _accumulate_variable_length(
                    sum_vector=sum_vector,
                    count_vector=count_vector,
                    value_vector=vec,
                    weight=1.0,
                )

                obj_class = str(obj_classes[sample_idx])
                episode_name = str(episode_names[sample_idx])
                class_vector_sum[obj_class], class_vector_count[obj_class] = _accumulate_variable_length(
                    sum_vector=class_vector_sum.get(obj_class),
                    count_vector=class_vector_count.get(obj_class),
                    value_vector=vec,
                    weight=1.0,
                )

                if obj_class not in class_layer_share_sum:
                    class_layer_share_sum[obj_class] = torch.zeros((num_layers, 3), dtype=torch.float64)
                for layer_idx in range(num_layers):
                    class_layer_share_sum[obj_class][layer_idx] += per_layer_per_sample[layer_idx][sample_idx]

                class_sample_count[obj_class] += 1
                sample_entries.append(
                    {
                        "global_index": total_samples + sample_idx,
                        "obj_class": obj_class,
                        "episode_name": episode_name,
                        "vector": vec.float().cpu().numpy(),
                    }
                )

            total_samples += bsz

            if tqdm is not None and hasattr(iterator, "set_postfix"):
                iterator.set_postfix({"samples": total_samples})
    finally:
        extractor.remove()

    if total_samples == 0 or sum_vector is None or count_vector is None:
        raise RuntimeError("No samples processed while collecting attention.")

    valid = count_vector > 0
    mean_vector = torch.zeros_like(sum_vector)
    mean_vector[valid] = sum_vector[valid] / count_vector[valid]
    mean_vector = mean_vector.float().cpu().numpy()
    mean_layer_shares = (layer_share_sum / total_samples).float().cpu().numpy()

    class_results: Dict[str, Dict[str, np.ndarray | int]] = {}
    for obj_class in sorted(class_sample_count.keys()):
        class_valid = class_vector_count[obj_class] > 0
        class_mean_vector = torch.zeros_like(class_vector_sum[obj_class])
        class_mean_vector[class_valid] = class_vector_sum[obj_class][class_valid] / class_vector_count[obj_class][
            class_valid
        ]
        class_results[obj_class] = {
            "mean_vector": class_mean_vector.float().cpu().numpy(),
            "layer_shares": (class_layer_share_sum[obj_class] / class_sample_count[obj_class]).float().cpu().numpy(),
            "num_samples": int(class_sample_count[obj_class]),
        }

    return {
        "mean_vector": mean_vector,
        "layer_shares": mean_layer_shares,
        "num_samples": np.asarray([total_samples], dtype=np.int64),
        "selected_layer": np.asarray([selected_layer], dtype=np.int64),
        "class_results": class_results,
        "sample_entries": sample_entries,
    }


def _token_ticks(max_len: int) -> Dict[str, List[int]]:
    tick_candidates = [0, 25, 49, 50, max_len // 2, max_len - 1]
    tick_positions = sorted(set([p for p in tick_candidates if 0 <= p < max_len]))
    tick_labels = []
    for p in tick_positions:
        if p == 0:
            tick_labels.append("CLS")
        elif p == 49:
            tick_labels.append("V-end")
        elif p == 50:
            tick_labels.append("T-start")
        elif p == max_len - 1:
            tick_labels.append("T-end")
        else:
            tick_labels.append(str(p))
    return {"positions": tick_positions, "labels": tick_labels}


def plot_single_split_attention(vector: np.ndarray, split_name: str, out_path: Path) -> None:
    heat = vector.reshape(1, -1)
    max_len = int(vector.shape[0])
    ticks = _token_ticks(max_len)

    fig, ax = plt.subplots(figsize=(14, 2.6), constrained_layout=True)
    im = ax.imshow(heat, aspect="auto", cmap="magma")
    ax.axvline(49.5, color="cyan", linestyle="--", linewidth=1.2)
    ax.set_yticks([0])
    ax.set_yticklabels([split_name])
    ax.set_xticks(ticks["positions"])
    ax.set_xticklabels(ticks["labels"])
    ax.set_title(f"{split_name} CLS -> All Tokens Attention")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Attention")
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_multi_row_attention(vectors: Sequence[np.ndarray], row_labels: Sequence[str], title: str, out_path: Path) -> None:
    if not vectors:
        raise ValueError("vectors is empty")

    max_len = int(max(v.shape[0] for v in vectors))
    heat = np.full((len(vectors), max_len), np.nan, dtype=np.float32)
    for row_idx, vec in enumerate(vectors):
        heat[row_idx, : vec.shape[0]] = vec
    ticks = _token_ticks(max_len)

    fig_height = max(3.0, 0.42 * len(vectors) + 1.8)
    fig, ax = plt.subplots(figsize=(14, fig_height), constrained_layout=True)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="#f0f0f0")
    im = ax.imshow(heat, aspect="auto", cmap=cmap)
    ax.axvline(49.5, color="cyan", linestyle="--", linewidth=1.2)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xticks(ticks["positions"])
    ax.set_xticklabels(ticks["labels"])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Attention")
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _slice_entries(entries: Sequence[Dict[str, object]], max_plot_samples: int) -> List[Dict[str, object]]:
    if max_plot_samples is None or max_plot_samples <= 0:
        return list(entries)
    return list(entries[:max_plot_samples])


def plot_split_sample_comparison(
    split: str,
    sample_entries: Sequence[Dict[str, object]],
    out_path: Path,
    max_plot_samples: int,
) -> int:
    selected = _slice_entries(sample_entries, max_plot_samples)
    if not selected:
        return 0
    vectors = [e["vector"] for e in selected]
    labels = [f"{e['obj_class']} | {e['episode_name']}" for e in selected]
    plot_multi_row_attention(
        vectors=vectors,
        row_labels=labels,
        title=f"{split} | per-sample attention comparison",
        out_path=out_path,
    )
    return len(selected)


def plot_test_ood_sample_comparison(
    test_entries: Sequence[Dict[str, object]],
    ood_entries: Sequence[Dict[str, object]],
    out_path: Path,
    max_plot_samples_per_split: int,
) -> Dict[str, int]:
    test_selected = _slice_entries(test_entries, max_plot_samples_per_split)
    ood_selected = _slice_entries(ood_entries, max_plot_samples_per_split)
    combined = test_selected + ood_selected
    if not combined:
        return {"test": 0, "ood_test": 0}

    vectors = [e["vector"] for e in combined]
    row_labels = [
        f"{'test' if idx < len(test_selected) else 'ood_test'} | {e['obj_class']} | {e['episode_name']}"
        for idx, e in enumerate(combined)
    ]
    max_len = int(max(v.shape[0] for v in vectors))
    heat = np.full((len(vectors), max_len), np.nan, dtype=np.float32)
    for row_idx, vec in enumerate(vectors):
        heat[row_idx, : vec.shape[0]] = vec
    ticks = _token_ticks(max_len)

    fig_height = max(4.0, 0.35 * len(vectors) + 2.0)
    fig, ax = plt.subplots(figsize=(14, fig_height), constrained_layout=True)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="#f0f0f0")
    im = ax.imshow(heat, aspect="auto", cmap=cmap)
    ax.axvline(49.5, color="cyan", linestyle="--", linewidth=1.2)
    if len(test_selected) > 0 and len(ood_selected) > 0:
        ax.axhline(len(test_selected) - 0.5, color="white", linestyle="--", linewidth=1.2)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xticks(ticks["positions"])
    ax.set_xticklabels(ticks["labels"])
    ax.set_title("test vs ood_test | per-sample attention comparison")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Attention")
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return {"test": len(test_selected), "ood_test": len(ood_selected)}


def plot_test_ood_class_comparison(
    test_class_results: Dict[str, Dict[str, np.ndarray | int]],
    ood_class_results: Dict[str, Dict[str, np.ndarray | int]],
    out_path: Path,
) -> None:
    vectors: List[np.ndarray] = []
    labels: List[str] = []
    for obj_class in sorted(test_class_results.keys()):
        vectors.append(test_class_results[obj_class]["mean_vector"])
        labels.append(f"test | {obj_class} (n={test_class_results[obj_class]['num_samples']})")
    for obj_class in sorted(ood_class_results.keys()):
        vectors.append(ood_class_results[obj_class]["mean_vector"])
        labels.append(f"ood_test | {obj_class} (n={ood_class_results[obj_class]['num_samples']})")

    if not vectors:
        return
    plot_multi_row_attention(
        vectors=vectors,
        row_labels=labels,
        title="test vs ood_test | class-level mean attention comparison",
        out_path=out_path,
    )


def export_split_detailed_maps(
    split: str,
    split_result: Dict[str, object],
    output_dir: Path,
    save_per_sample: bool,
    save_class_internal_maps: bool,
) -> None:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    by_class_dir = split_dir / "by_class"
    by_class_dir.mkdir(parents=True, exist_ok=True)

    class_results = split_result["class_results"]
    sample_entries = split_result["sample_entries"]

    class_vectors = []
    class_labels = []
    for obj_class in sorted(class_results.keys()):
        class_res = class_results[obj_class]
        class_vector = class_res["mean_vector"]
        class_vectors.append(class_vector)
        class_labels.append(f"{obj_class} (n={class_res['num_samples']})")

        safe_class = sanitize_filename(obj_class)
        np.save(by_class_dir / f"{safe_class}_mean.npy", class_vector)
        np.save(by_class_dir / f"{safe_class}_layer_shares.npy", class_res["layer_shares"])
        plot_single_split_attention(
            vector=class_vector,
            split_name=f"{split}:{obj_class}",
            out_path=by_class_dir / f"{safe_class}_mean.png",
        )

        class_samples = [e for e in sample_entries if e["obj_class"] == obj_class]
        if save_class_internal_maps and class_samples:
            class_sample_vectors = [e["vector"] for e in class_samples]
            class_sample_labels = [f"{e['episode_name']}#{e['global_index']}" for e in class_samples]
            plot_multi_row_attention(
                vectors=class_sample_vectors,
                row_labels=class_sample_labels,
                title=f"{split} | {obj_class} | per-sample attention",
                out_path=by_class_dir / f"{safe_class}_samples.png",
            )

    if class_vectors:
        plot_multi_row_attention(
            vectors=class_vectors,
            row_labels=class_labels,
            title=f"{split} | class-level mean attention",
            out_path=split_dir / f"attention_map_{split}_by_class.png",
        )

    if save_per_sample and sample_entries:
        sample_dir = split_dir / "per_sample"
        sample_dir.mkdir(parents=True, exist_ok=True)
        for e in sample_entries:
            safe_class = sanitize_filename(e["obj_class"])
            safe_episode = sanitize_filename(e["episode_name"])
            idx = int(e["global_index"])
            plot_single_split_attention(
                vector=e["vector"],
                split_name=f"{split}:{e['obj_class']}:{e['episode_name']}",
                out_path=sample_dir / f"{idx:04d}_{safe_class}_{safe_episode}.png",
            )


def plot_test_ood_comparison(test_vec: np.ndarray, ood_vec: np.ndarray, out_path: Path) -> None:
    max_len = int(max(test_vec.shape[0], ood_vec.shape[0]))
    heat = np.full((2, max_len), np.nan, dtype=np.float32)
    heat[0, : test_vec.shape[0]] = test_vec
    heat[1, : ood_vec.shape[0]] = ood_vec
    ticks = _token_ticks(max_len)

    fig, ax = plt.subplots(figsize=(14, 3.2), constrained_layout=True)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="#f0f0f0")
    im = ax.imshow(heat, aspect="auto", cmap=cmap)
    ax.axvline(49.5, color="cyan", linestyle="--", linewidth=1.2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["test", "ood_test"])
    ax.set_xticks(ticks["positions"])
    ax.set_xticklabels(ticks["labels"])
    ax.set_title("Last-layer CLS -> All Tokens Attention (Head-averaged)")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Attention")
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_layer_share_comparison(test_shares: np.ndarray, ood_shares: np.ndarray, out_path: Path) -> None:
    num_layers = int(test_shares.shape[0])
    x = np.arange(num_layers)
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    test_cls, test_vis, test_tac = test_shares[:, 0], test_shares[:, 1], test_shares[:, 2]
    ood_cls, ood_vis, ood_tac = ood_shares[:, 0], ood_shares[:, 1], ood_shares[:, 2]

    ax.bar(x - width / 2, test_vis, width=width, label="Test Vision", color="#4C72B0")
    ax.bar(x - width / 2, test_tac, width=width, bottom=test_vis, label="Test Tactile", color="#55A868")
    ax.bar(x - width / 2, test_cls, width=width, bottom=test_vis + test_tac, label="Test CLS", color="#C44E52")

    ax.bar(x + width / 2, ood_vis, width=width, label="OOD Vision", color="#8172B2")
    ax.bar(x + width / 2, ood_tac, width=width, bottom=ood_vis, label="OOD Tactile", color="#64B5CD")
    ax.bar(x + width / 2, ood_cls, width=width, bottom=ood_vis + ood_tac, label="OOD CLS", color="#CCB974")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {i + 1}" for i in range(num_layers)])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Attention share (sum-normalized)")
    ax.set_title("CLS Attention Share by Layer (test vs ood_test)")
    ax.grid(alpha=0.2, axis="y", linestyle="--")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))

    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    data_root = Path(args.data_root)
    checkpoint = Path(args.checkpoint)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = checkpoint.parent / "attention_maps_gating"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(checkpoint, args, device)
    split_results: Dict[str, Dict[str, object]] = {}

    for split in args.splits:
        split_dir = data_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split not found: {split_dir}")

        loader = build_loader(
            split_dir=split_dir,
            batch_size=args.batch_size,
            max_tactile_len=args.max_tactile_len,
            num_workers=args.num_workers,
        )
        print(f"Collecting attention for split={split}, samples={len(loader.dataset)}")
        result = collect_split_attention(
            model=model,
            loader=loader,
            device=device,
            layer_index=args.layer_index,
            max_samples=args.max_samples,
        )
        split_results[split] = result
        np.save(output_dir / f"attention_vector_{split}.npy", result["mean_vector"])
        np.save(output_dir / f"attention_shares_{split}.npy", result["layer_shares"])
        plot_single_split_attention(
            vector=result["mean_vector"],
            split_name=split,
            out_path=output_dir / f"attention_map_{split}.png",
        )
        export_split_detailed_maps(
            split=split,
            split_result=result,
            output_dir=output_dir,
            save_per_sample=args.save_per_sample_maps,
            save_class_internal_maps=args.save_class_internal_maps,
        )
        if args.save_sample_comparison_maps:
            plotted = plot_split_sample_comparison(
                split=split,
                sample_entries=result["sample_entries"],
                out_path=output_dir / split / f"attention_map_{split}_all_samples.png",
                max_plot_samples=args.max_plot_samples_per_split,
            )
            print(f"Saved per-sample comparison for {split}: {plotted} rows")

    if "test" in split_results and "ood_test" in split_results:
        plot_test_ood_comparison(
            test_vec=split_results["test"]["mean_vector"],
            ood_vec=split_results["ood_test"]["mean_vector"],
            out_path=output_dir / "attention_map_test_vs_ood.png",
        )
        plot_test_ood_class_comparison(
            test_class_results=split_results["test"]["class_results"],
            ood_class_results=split_results["ood_test"]["class_results"],
            out_path=output_dir / "attention_map_test_vs_ood_by_class.png",
        )
        plot_layer_share_comparison(
            test_shares=split_results["test"]["layer_shares"],
            ood_shares=split_results["ood_test"]["layer_shares"],
            out_path=output_dir / "attention_share_layers_test_vs_ood.png",
        )
        if args.save_sample_comparison_maps:
            plotted_counts = plot_test_ood_sample_comparison(
                test_entries=split_results["test"]["sample_entries"],
                ood_entries=split_results["ood_test"]["sample_entries"],
                out_path=output_dir / "attention_map_test_vs_ood_all_samples.png",
                max_plot_samples_per_split=args.max_plot_samples_per_split,
            )
            print(
                "Saved test-vs-ood per-sample comparison: "
                f"test={plotted_counts['test']}, ood_test={plotted_counts['ood_test']}"
            )

    summary = {
        "checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
        "splits": {},
    }
    for split, result in split_results.items():
        class_results = result["class_results"]
        summary["splits"][split] = {
            "num_samples_used": int(result["num_samples"][0]),
            "selected_layer_index": int(result["selected_layer"][0]),
            "sample_comparison_rows": int(
                (
                    min(len(result["sample_entries"]), args.max_plot_samples_per_split)
                    if args.max_plot_samples_per_split > 0
                    else len(result["sample_entries"])
                )
                if args.save_sample_comparison_maps
                else 0
            ),
            "avg_layer_share": {
                "cls": float(result["layer_shares"][:, 0].mean()),
                "vision": float(result["layer_shares"][:, 1].mean()),
                "tactile": float(result["layer_shares"][:, 2].mean()),
            },
            "classes": {
                obj_class: {
                    "num_samples": int(class_results[obj_class]["num_samples"]),
                    "avg_layer_share": {
                        "cls": float(class_results[obj_class]["layer_shares"][:, 0].mean()),
                        "vision": float(class_results[obj_class]["layer_shares"][:, 1].mean()),
                        "tactile": float(class_results[obj_class]["layer_shares"][:, 2].mean()),
                    },
                }
                for obj_class in sorted(class_results.keys())
            },
        }

    (output_dir / "attention_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Done. Results saved to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize attention maps for FusionModel (train_fusion_gating.py) on test and ood_test splits."
    )
    parser.add_argument("--data_root", type=str, default="/home/martina/Y3_Project/Plaintextdataset")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--splits", type=str, default="test,ood_test")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_tactile_len", type=int, default=3000)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means using all samples.")
    parser.add_argument(
        "--max_plot_samples_per_split",
        type=int,
        default=0,
        help="Number of samples shown in per-sample comparison maps per split (0 means all).",
    )
    parser.set_defaults(save_sample_comparison_maps=False)
    parser.add_argument(
        "--save_sample_comparison_maps",
        dest="save_sample_comparison_maps",
        action="store_true",
        help="Save split-level and test-vs-ood per-sample comparison maps.",
    )
    parser.add_argument(
        "--no_save_sample_comparison_maps",
        dest="save_sample_comparison_maps",
        action="store_false",
        help="Disable per-sample comparison maps (default).",
    )
    parser.set_defaults(save_per_sample_maps=False)
    parser.add_argument(
        "--save_per_sample_maps",
        dest="save_per_sample_maps",
        action="store_true",
        help="Also export one PNG per sample under <output_dir>/<split>/per_sample.",
    )
    parser.add_argument(
        "--no_save_per_sample_maps",
        dest="save_per_sample_maps",
        action="store_false",
        help="Disable per-sample PNG export (class-level outputs are still saved).",
    )
    parser.set_defaults(save_class_internal_maps=False)
    parser.add_argument(
        "--save_class_internal_maps",
        dest="save_class_internal_maps",
        action="store_true",
        help="Save class-internal per-sample comparison maps under <output_dir>/<split>/by_class/*_samples.png.",
    )
    parser.add_argument(
        "--no_save_class_internal_maps",
        dest="save_class_internal_maps",
        action="store_false",
        help="Disable class-internal per-sample maps (default).",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--layer_index", type=int, default=-1, help="-1 means the last transformer layer.")

    parser.add_argument("--fusion_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_layers", type=int, default=4)

    args = parser.parse_args()
    args.splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not args.splits:
        raise ValueError("--splits cannot be empty")
    return args


if __name__ == "__main__":
    run(parse_args())
