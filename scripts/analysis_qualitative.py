import argparse
import csv
import json
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader, TensorDataset

from train_fusion import FusionModel as StandardFusionModel, RoboticGraspDataset, resolve_device, set_seed
from train_tactile import TactileOnlyModel

try:
    from train_fusion_gating import FusionModel as GatedFusionModel
except Exception:  # pragma: no cover
    GatedFusionModel = None

# Keep the existing type hints valid for both standard and gating fusion models.
FusionModel = StandardFusionModel


TASK_NUM_CLASSES = {"mass": 4, "stiffness": 4, "material": 5}
TASKS = list(TASK_NUM_CLASSES.keys())


@dataclass
class AnalysisSample:
    img_path: Path
    tactile_path: Path
    labels: Dict[str, int]
    obj_class: str


class AnalysisRoboticGraspDataset(RoboticGraspDataset):
    """Adds object-class metadata and valid tactile length for analysis."""

    def _collect_samples(self) -> List[AnalysisSample]:
        samples: List[AnalysisSample] = []
        for obj_class_dir in sorted(self.split_dir.iterdir()):
            if not obj_class_dir.is_dir():
                continue
            if obj_class_dir.name.startswith("analysis"):
                continue

            labels = self._parse_labels(obj_class_dir.name)
            if labels is None:
                continue

            for episode_dir in sorted(obj_class_dir.iterdir()):
                if not episode_dir.is_dir():
                    continue
                img_path = episode_dir / "visual_anchor.jpg"
                tactile_path = episode_dir / "tactile_data.pkl"
                if img_path.exists() and tactile_path.exists():
                    samples.append(
                        AnalysisSample(
                            img_path=img_path,
                            tactile_path=tactile_path,
                            labels=labels,
                            obj_class=obj_class_dir.name,
                        )
                    )

        if not samples:
            raise RuntimeError(f"No valid samples found in {self.split_dir}")
        return samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = super().__getitem__(idx)
        sample = self.samples[idx]
        valid_tactile_len = int((~item["padding_mask"]).sum().item())
        item["obj_class"] = sample.obj_class
        item["valid_tactile_len"] = torch.tensor(valid_tactile_len, dtype=torch.long)
        return item


def build_analysis_loader(
    split_dir: Path,
    batch_size: int,
    max_tactile_len: int,
    num_workers: int,
    shuffle: bool = False,
) -> DataLoader:
    dataset = AnalysisRoboticGraspDataset(split_dir=split_dir, max_tactile_len=max_tactile_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def _downsample_tactile_padding_mask(padding_mask: torch.Tensor, num_tac_tokens: int) -> torch.Tensor:
    tac_mask = padding_mask.float().unsqueeze(1)
    tac_mask = F.max_pool1d(tac_mask, kernel_size=2, stride=2)
    tac_mask = F.max_pool1d(tac_mask, kernel_size=2, stride=2)
    tac_mask = F.max_pool1d(tac_mask, kernel_size=2, stride=2)
    tac_mask = tac_mask.squeeze(1) > 0.5
    return tac_mask[:, :num_tac_tokens]


def forward_fusion_with_modality_mask(
    model: FusionModel,
    images: torch.Tensor,
    tactile: torch.Tensor,
    padding_mask: torch.Tensor,
    mode: str,
) -> Dict[str, torch.Tensor]:
    if mode not in {"full", "tactile_only", "vision_only"}:
        raise ValueError(f"Unsupported mode: {mode}")

    bsz = images.shape[0]
    device = images.device

    v = model.vis_backbone(images)
    v = model.vis_proj(v)
    v_tokens = v.flatten(2).transpose(1, 2)
    num_vis_tokens = v_tokens.shape[1]

    t = model.tac_encoder(tactile)
    t_tokens = t.transpose(1, 2)
    num_tac_tokens = t_tokens.shape[1]

    cls_token = model.cls_token.expand(bsz, -1, -1)
    x = torch.cat([cls_token, v_tokens, t_tokens], dim=1)
    x = x + model.pos_emb[:, : x.shape[1], :]

    cls_mask = torch.zeros(bsz, 1, dtype=torch.bool, device=device)
    vis_mask = torch.zeros(bsz, num_vis_tokens, dtype=torch.bool, device=device)
    tac_mask = _downsample_tactile_padding_mask(padding_mask, num_tac_tokens)

    if mode == "tactile_only":
        vis_mask[:] = True
    elif mode == "vision_only":
        tac_mask[:] = True

    full_mask = torch.cat([cls_mask, vis_mask, tac_mask], dim=1)
    x = model.transformer_encoder(x, src_key_padding_mask=full_mask)
    cls_out = x[:, 0, :]

    return {
        "mass": model.head_mass(cls_out),
        "stiffness": model.head_stiffness(cls_out),
        "material": model.head_material(cls_out),
    }


def load_label_maps(split_dir: Path) -> Dict[str, Dict[int, str]]:
    props_path = split_dir / "physical_properties.json"
    config = json.loads(props_path.read_text(encoding="utf-8"))
    return {
        "mass": {int(k): v for k, v in config["idx_to_mass"].items()},
        "stiffness": {int(k): v for k, v in config["idx_to_stiffness"].items()},
        "material": {int(k): v for k, v in config["idx_to_material"].items()},
    }


def load_fusion_model(checkpoint_path: Path, device: torch.device, args: argparse.Namespace) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})
    state_dict = checkpoint["model_state_dict"]
    is_gated_checkpoint = any(k == "t_null" or k.startswith("gate_mlp.") for k in state_dict.keys())

    if is_gated_checkpoint:
        if GatedFusionModel is None:
            raise ImportError(
                "Detected gating checkpoint, but train_fusion_gating.FusionModel is unavailable."
            )
        fusion_cls = GatedFusionModel
    else:
        fusion_cls = StandardFusionModel

    model = fusion_cls(
        fusion_dim=cfg.get("fusion_dim", args.fusion_dim),
        num_heads=cfg.get("num_heads", args.num_heads),
        dropout=cfg.get("dropout", args.dropout),
        num_layers=cfg.get("num_layers", args.num_layers),
        freeze_visual=True,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_tactile_model(checkpoint_path: Path, device: torch.device, args: argparse.Namespace) -> TactileOnlyModel:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})
    model = TactileOnlyModel(
        fusion_dim=cfg.get("fusion_dim", args.fusion_dim),
        num_heads=cfg.get("num_heads", args.num_heads),
        dropout=cfg.get("dropout", args.dropout),
        num_layers=cfg.get("num_layers", args.num_layers),
        max_tactile_len=cfg.get("max_tactile_len", args.max_tactile_len),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def collect_cls_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_kind: str,
) -> Dict[str, object]:
    embeddings: List[torch.Tensor] = []
    labels = {task: [] for task in TASKS}
    obj_classes: List[str] = []
    valid_lens: List[int] = []

    hook_cache: List[torch.Tensor] = []

    def hook_fn(_module, _inputs, output):
        # Transformer output shape is (B, SeqLen, D) because batch_first=True in all models used.
        hook_cache.append(output[:, 0, :].detach().cpu())

    handle = model.transformer_encoder.register_forward_hook(hook_fn)
    model.eval()

    for batch in loader:
        hook_cache.clear()
        tactile = batch["tactile"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        if model_kind == "tactile_only":
            _ = model(tactile, padding_mask=padding_mask)
        elif model_kind == "fusion_tactile_only":
            images = batch["image"].to(device)
            _ = forward_fusion_with_modality_mask(
                model=model,
                images=images,
                tactile=tactile,
                padding_mask=padding_mask,
                mode="tactile_only",
            )
        elif model_kind == "fusion_full":
            images = batch["image"].to(device)
            _ = model(images, tactile, padding_mask=padding_mask)
        else:
            raise ValueError(f"Unknown model_kind: {model_kind}")

        if not hook_cache:
            raise RuntimeError("Transformer hook did not capture embeddings.")

        embeddings.append(hook_cache[-1])
        for task in TASKS:
            labels[task].extend(batch[task].cpu().numpy().tolist())
        obj_classes.extend(list(batch["obj_class"]))
        valid_lens.extend(batch["valid_tactile_len"].cpu().numpy().tolist())

    handle.remove()

    return {
        "embeddings": torch.cat(embeddings, dim=0).numpy(),
        "labels": {task: np.asarray(vals, dtype=np.int64) for task, vals in labels.items()},
        "obj_class": np.asarray(obj_classes),
        "valid_tactile_len": np.asarray(valid_lens, dtype=np.int64),
    }


def run_reducer_tsne(embeddings: np.ndarray, perplexity: int, seed: int) -> np.ndarray:
    n_samples = embeddings.shape[0]
    safe_perplexity = max(2, min(perplexity, n_samples - 1))
    reducer = TSNE(
        n_components=2,
        perplexity=safe_perplexity,
        learning_rate="auto",
        init="pca",
        random_state=seed,
        max_iter=1000,
    )
    return reducer.fit_transform(embeddings)


def run_reducer_umap(embeddings: np.ndarray, n_neighbors: int, min_dist: float, seed: int) -> np.ndarray:
    try:
        import umap  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise ImportError("UMAP 依赖缺失，请先安装: pip install umap-learn") from exc

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=max(2, min(n_neighbors, embeddings.shape[0] - 1)),
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
    )
    return reducer.fit_transform(embeddings)


def safe_silhouette(coords: np.ndarray, labels: np.ndarray) -> Optional[float]:
    unique = np.unique(labels)
    if len(unique) < 2:
        return None
    counts = [np.sum(labels == u) for u in unique]
    if min(counts) < 2:
        return None
    try:
        return float(silhouette_score(coords, labels))
    except Exception:
        return None


def plot_two_model_scatter(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    label_names: Dict[int, str],
    out_path: Path,
    method_name: str,
    legend_title: str,
    title_a: str,
    title_b: str,
) -> None:
    all_ids = sorted(set(np.unique(labels_a).tolist()) | set(np.unique(labels_b).tolist()))
    cmap = plt.get_cmap("tab10", max(len(all_ids), 1))
    color_of = {label_id: cmap(i) for i, label_id in enumerate(all_ids)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    panel_data = [
        (axes[0], coords_a, labels_a, title_a),
        (axes[1], coords_b, labels_b, title_b),
    ]

    for ax, coords, labels, title in panel_data:
        for label_id in all_ids:
            mask = labels == label_id
            if not np.any(mask):
                continue
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=36,
                alpha=0.85,
                color=color_of[label_id],
                label=label_names.get(label_id, str(label_id)),
                edgecolors="none",
            )
        ax.set_title(title)
        ax.set_xlabel(f"{method_name}-1")
        ax.set_ylabel(f"{method_name}-2")
        ax.grid(alpha=0.2, linestyle="--")

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", title=legend_title)
    fig.suptitle(f"{method_name} OOD Feature Clusters", fontsize=14, fontweight="bold")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_tsne_umap_analysis(
    model_a: nn.Module,
    model_b: nn.Module,
    model_a_kind: str,
    model_b_kind: str,
    model_a_name: str,
    model_b_name: str,
    ood_loader: DataLoader,
    label_maps: Dict[str, Dict[int, str]],
    output_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, object]:
    model_a_data = collect_cls_embeddings(model_a, ood_loader, args.device, model_kind=model_a_kind)
    model_b_data = collect_cls_embeddings(model_b, ood_loader, args.device, model_kind=model_b_kind)

    if model_a_data["embeddings"].shape[0] != model_b_data["embeddings"].shape[0]:
        raise RuntimeError("Mismatch in sample count between the two compared models.")

    obj_to_idx = {name: i for i, name in enumerate(sorted(np.unique(model_a_data["obj_class"]).tolist()))}
    object_labels = np.asarray([obj_to_idx[name] for name in model_a_data["obj_class"]], dtype=np.int64)
    object_label_names = {v: k for k, v in obj_to_idx.items()}

    tsne_a = run_reducer_tsne(model_a_data["embeddings"], perplexity=args.tsne_perplexity, seed=args.seed)
    tsne_b = run_reducer_tsne(model_b_data["embeddings"], perplexity=args.tsne_perplexity, seed=args.seed)

    umap_a = run_reducer_umap(
        model_a_data["embeddings"],
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        seed=args.seed,
    )
    umap_b = run_reducer_umap(
        model_b_data["embeddings"],
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        seed=args.seed,
    )

    plot_two_model_scatter(
        coords_a=umap_a,
        coords_b=umap_b,
        labels_a=model_a_data["labels"]["material"],
        labels_b=model_b_data["labels"]["material"],
        label_names=label_maps["material"],
        out_path=output_dir / "umap_by_material.png",
        method_name="UMAP",
        legend_title="Material",
        title_a=model_a_name,
        title_b=model_b_name,
    )
    plot_two_model_scatter(
        coords_a=umap_a,
        coords_b=umap_b,
        labels_a=object_labels,
        labels_b=object_labels,
        label_names=object_label_names,
        out_path=output_dir / "umap_by_object.png",
        method_name="UMAP",
        legend_title="Object class",
        title_a=model_a_name,
        title_b=model_b_name,
    )
    plot_two_model_scatter(
        coords_a=tsne_a,
        coords_b=tsne_b,
        labels_a=model_a_data["labels"]["material"],
        labels_b=model_b_data["labels"]["material"],
        label_names=label_maps["material"],
        out_path=output_dir / "tsne_by_material.png",
        method_name="t-SNE",
        legend_title="Material",
        title_a=model_a_name,
        title_b=model_b_name,
    )
    plot_two_model_scatter(
        coords_a=tsne_a,
        coords_b=tsne_b,
        labels_a=object_labels,
        labels_b=object_labels,
        label_names=object_label_names,
        out_path=output_dir / "tsne_by_object.png",
        method_name="t-SNE",
        legend_title="Object class",
        title_a=model_a_name,
        title_b=model_b_name,
    )

    return {
        "num_samples": int(model_a_data["embeddings"].shape[0]),
        "comparison": {
            "mode": args.tsne_compare_mode,
            "model_a": {"name": model_a_name, "kind": model_a_kind},
            "model_b": {"name": model_b_name, "kind": model_b_kind},
        },
        "methods": {
            "umap": {
                "material_silhouette": {
                    "model_a": safe_silhouette(umap_a, model_a_data["labels"]["material"]),
                    "model_b": safe_silhouette(umap_b, model_b_data["labels"]["material"]),
                },
                "object_silhouette": {
                    "model_a": safe_silhouette(umap_a, object_labels),
                    "model_b": safe_silhouette(umap_b, object_labels),
                },
            },
            "tsne": {
                "material_silhouette": {
                    "model_a": safe_silhouette(tsne_a, model_a_data["labels"]["material"]),
                    "model_b": safe_silhouette(tsne_b, model_b_data["labels"]["material"]),
                },
                "object_silhouette": {
                    "model_a": safe_silhouette(tsne_a, object_labels),
                    "model_b": safe_silhouette(tsne_b, object_labels),
                },
            },
        },
        "object_classes": sorted(obj_to_idx.keys()),
    }


def find_sample_by_obj_class(dataset: AnalysisRoboticGraspDataset, obj_class: str) -> Dict[str, object]:
    for idx, sample in enumerate(dataset.samples):
        if sample.obj_class == obj_class:
            item = dataset[idx]
            item["sample_idx"] = idx
            return item
    raise ValueError(f"Could not find sample with obj_class={obj_class} in {dataset.split_dir}")


def make_single_batch(item: Dict[str, object], device: torch.device) -> Dict[str, object]:
    return {
        "image": item["image"].unsqueeze(0).to(device),
        "tactile": item["tactile"].unsqueeze(0).to(device),
        "padding_mask": item["padding_mask"].unsqueeze(0).to(device),
        "obj_class": item["obj_class"],
        "sample_idx": item["sample_idx"],
    }


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


def extract_attention_layers(
    model: FusionModel,
    image: torch.Tensor,
    tactile: torch.Tensor,
    padding_mask: torch.Tensor,
) -> List[torch.Tensor]:
    model.eval()
    attn_weights_store: List[torch.Tensor] = []

    def attn_pre_hook(_module, args, kwargs):
        kw = dict(kwargs)
        kw["need_weights"] = True
        kw["average_attn_weights"] = False
        return args, kw

    def attn_post_hook(_module, _input, output):
        if output[1] is None:
            raise RuntimeError("Attention weights are None. SDPA fallback likely not active.")
        attn_weights_store.append(output[1].detach().cpu())

    handles = []
    for layer in model.transformer_encoder.layers:
        handles.append(layer.self_attn.register_forward_pre_hook(attn_pre_hook, with_kwargs=True))
        handles.append(layer.self_attn.register_forward_hook(attn_post_hook))

    with torch.no_grad():
        with safe_sdp_context():
            _ = model(image, tactile, padding_mask=padding_mask)

    for handle in handles:
        handle.remove()

    if len(attn_weights_store) != len(model.transformer_encoder.layers):
        raise RuntimeError(
            f"Expected {len(model.transformer_encoder.layers)} attention maps, got {len(attn_weights_store)}."
        )
    return attn_weights_store


def compute_layer_shares(attn_layers: List[torch.Tensor]) -> np.ndarray:
    shares = []
    for attn in attn_layers:
        cls_attn = attn[:, :, 0, :]
        cls_share = cls_attn[:, :, 0:1].sum(dim=-1)
        vis_share = cls_attn[:, :, 1:50].sum(dim=-1)
        tac_share = cls_attn[:, :, 50:].sum(dim=-1)
        total = (cls_share + vis_share + tac_share).clamp(min=1e-12)

        cls_ratio = (cls_share / total).mean().item()
        vis_ratio = (vis_share / total).mean().item()
        tac_ratio = (tac_share / total).mean().item()
        shares.append([cls_ratio, vis_ratio, tac_ratio])
    return np.asarray(shares, dtype=np.float32)


def plot_attention_stacked_bars(id_shares: np.ndarray, ood_shares: np.ndarray, out_path: Path) -> None:
    num_layers = id_shares.shape[0]
    x = np.arange(num_layers)
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    id_cls, id_vis, id_tac = id_shares[:, 0], id_shares[:, 1], id_shares[:, 2]
    ood_cls, ood_vis, ood_tac = ood_shares[:, 0], ood_shares[:, 1], ood_shares[:, 2]

    ax.bar(x - width / 2, id_vis, width=width, label="ID Vision", color="#4C72B0")
    ax.bar(x - width / 2, id_tac, width=width, bottom=id_vis, label="ID Tactile", color="#55A868")
    ax.bar(
        x - width / 2,
        id_cls,
        width=width,
        bottom=id_vis + id_tac,
        label="ID CLS",
        color="#C44E52",
    )

    ax.bar(x + width / 2, ood_vis, width=width, label="OOD Vision", color="#8172B2")
    ax.bar(x + width / 2, ood_tac, width=width, bottom=ood_vis, label="OOD Tactile", color="#64B5CD")
    ax.bar(
        x + width / 2,
        ood_cls,
        width=width,
        bottom=ood_vis + ood_tac,
        label="OOD CLS",
        color="#CCB974",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {i+1}" for i in range(num_layers)])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Attention share (sum-normalized)")
    ax.set_title("CLS Attention Capacity Allocation (ID vs OOD)")
    ax.grid(alpha=0.2, axis="y", linestyle="--")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))

    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_attention_heatmap(
    id_layers: List[torch.Tensor],
    ood_layers: List[torch.Tensor],
    out_path: Path,
    id_name: str,
    ood_name: str,
) -> None:
    id_vec = id_layers[-1][0, :, 0, :].mean(dim=0).numpy()
    ood_vec = ood_layers[-1][0, :, 0, :].mean(dim=0).numpy()
    max_len = int(max(id_vec.shape[0], ood_vec.shape[0]))
    heat = np.full((2, max_len), np.nan, dtype=np.float32)
    heat[0, : id_vec.shape[0]] = id_vec
    heat[1, : ood_vec.shape[0]] = ood_vec

    fig, ax = plt.subplots(figsize=(14, 3.4), constrained_layout=True)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="#f0f0f0")
    im = ax.imshow(heat, aspect="auto", cmap=cmap)
    ax.axvline(49.5, color="cyan", linestyle="--", linewidth=1.2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([f"ID: {id_name}", f"OOD: {ood_name}"])
    tick_candidates = [0, 25, 49, 50, max_len // 2, max_len - 1]
    tick_positions = sorted(set([t for t in tick_candidates if 0 <= t < max_len]))
    tick_labels = []
    for t in tick_positions:
        if t == 0:
            tick_labels.append("CLS")
        elif t == 49:
            tick_labels.append("V-end")
        elif t == 50:
            tick_labels.append("T-start")
        elif t == max_len - 1:
            tick_labels.append("T-end")
        else:
            tick_labels.append(str(t))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_title("Last-Layer CLS -> All Tokens Attention (Head-averaged)")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Attention")
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_attention_heatmap_multi_models(
    model_to_vectors: Dict[str, List[np.ndarray]],
    row_labels: List[str],
    out_path: Path,
) -> None:
    model_names = list(model_to_vectors.keys())
    if not model_names:
        raise ValueError("model_to_vectors is empty.")

    max_len = max(vec.shape[0] for vectors in model_to_vectors.values() for vec in vectors)
    n_rows = len(row_labels)
    n_cols = len(model_names)

    all_values = np.concatenate([np.concatenate(vectors) for vectors in model_to_vectors.values()])
    vmin = float(np.min(all_values))
    vmax = float(np.max(all_values))

    fig_width = max(8, 6 * n_cols)
    fig_height = max(4, 0.48 * n_rows + 2.6)
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_width, fig_height), constrained_layout=True)
    if n_cols == 1:
        axes = [axes]

    im = None
    for col, model_name in enumerate(model_names):
        heat = np.full((n_rows, max_len), np.nan, dtype=np.float32)
        vectors = model_to_vectors[model_name]
        for row_idx, vec in enumerate(vectors):
            heat[row_idx, : vec.shape[0]] = vec

        ax = axes[col]
        cmap = plt.get_cmap("magma").copy()
        cmap.set_bad(color="#f0f0f0")
        im = ax.imshow(heat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axvline(49.5, color="cyan", linestyle="--", linewidth=1.2)
        ax.set_title(model_name)

        if col == 0:
            ax.set_yticks(np.arange(n_rows))
            ax.set_yticklabels(row_labels)
        else:
            ax.set_yticks(np.arange(n_rows))
            ax.set_yticklabels([])

        tick_candidates = [0, 25, 49, 50, max_len // 2, max_len - 1]
        tick_positions = sorted(set([t for t in tick_candidates if 0 <= t < max_len]))
        tick_labels = []
        for t in tick_positions:
            if t == 0:
                tick_labels.append("CLS")
            elif t == 49:
                tick_labels.append("V-end")
            elif t == 50:
                tick_labels.append("T-start")
            elif t == max_len - 1:
                tick_labels.append("T-end")
            else:
                tick_labels.append(str(t))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)

    fig.suptitle("Last-Layer CLS -> All Tokens Attention (Multi-object, Multi-model)", fontsize=13, fontweight="bold")
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="Attention")
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_attention_stacked_bars_multi_models(
    model_to_shares: Dict[str, Dict[str, np.ndarray]],
    out_path: Path,
) -> None:
    model_names = list(model_to_shares.keys())
    if not model_names:
        raise ValueError("model_to_shares is empty.")

    num_models = len(model_names)
    num_layers = int(model_to_shares[model_names[0]]["id"].shape[0])
    x = np.arange(num_layers)
    width = 0.36

    fig, axes = plt.subplots(1, num_models, figsize=(7 * num_models, 5.4), constrained_layout=True, sharey=True)
    if num_models == 1:
        axes = [axes]

    for idx, model_name in enumerate(model_names):
        ax = axes[idx]
        id_shares = model_to_shares[model_name]["id"]
        ood_shares = model_to_shares[model_name]["ood"]
        id_cls, id_vis, id_tac = id_shares[:, 0], id_shares[:, 1], id_shares[:, 2]
        ood_cls, ood_vis, ood_tac = ood_shares[:, 0], ood_shares[:, 1], ood_shares[:, 2]

        ax.bar(x - width / 2, id_vis, width=width, label="ID Vision", color="#4C72B0")
        ax.bar(x - width / 2, id_tac, width=width, bottom=id_vis, label="ID Tactile", color="#55A868")
        ax.bar(
            x - width / 2,
            id_cls,
            width=width,
            bottom=id_vis + id_tac,
            label="ID CLS",
            color="#C44E52",
        )

        ax.bar(x + width / 2, ood_vis, width=width, label="OOD Vision", color="#8172B2")
        ax.bar(x + width / 2, ood_tac, width=width, bottom=ood_vis, label="OOD Tactile", color="#64B5CD")
        ax.bar(
            x + width / 2,
            ood_cls,
            width=width,
            bottom=ood_vis + ood_tac,
            label="OOD CLS",
            color="#CCB974",
        )

        ax.set_xticks(x)
        ax.set_xticklabels([f"L{i+1}" for i in range(num_layers)])
        ax.set_ylim(0.0, 1.0)
        ax.set_title(model_name)
        ax.grid(alpha=0.2, axis="y", linestyle="--")
        if idx == 0:
            ax.set_ylabel("Attention share (sum-normalized)")

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.01, 0.5))
    fig.suptitle("CLS Attention Capacity Allocation (ID vs OOD, averaged over selected objects)", fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def run_attention_analysis(
    models: Dict[str, nn.Module],
    data_root: Path,
    output_dir: Path,
    device: torch.device,
    id_obj_classes: List[str],
    ood_obj_classes: List[str],
    max_tactile_len: int,
) -> Dict[str, object]:
    id_dataset = AnalysisRoboticGraspDataset(split_dir=data_root / "test", max_tactile_len=max_tactile_len)
    ood_dataset = AnalysisRoboticGraspDataset(split_dir=data_root / "ood_test", max_tactile_len=max_tactile_len)

    item_entries = []
    for obj_class in id_obj_classes:
        item = find_sample_by_obj_class(id_dataset, obj_class)
        item_entries.append(
            {
                "split": "ID",
                "obj_class": obj_class,
                "sample_idx": int(item["sample_idx"]),
                "batch": make_single_batch(item, device),
            }
        )
    for obj_class in ood_obj_classes:
        item = find_sample_by_obj_class(ood_dataset, obj_class)
        item_entries.append(
            {
                "split": "OOD",
                "obj_class": obj_class,
                "sample_idx": int(item["sample_idx"]),
                "batch": make_single_batch(item, device),
            }
        )

    row_labels = [f"{e['split']}: {e['obj_class']}" for e in item_entries]
    model_to_vectors: Dict[str, List[np.ndarray]] = {}
    model_to_mean_shares: Dict[str, Dict[str, np.ndarray]] = {}
    model_summary: Dict[str, Dict[str, object]] = {}

    for model_name, fusion_model in models.items():
        vectors: List[np.ndarray] = []
        id_shares_all: List[np.ndarray] = []
        ood_shares_all: List[np.ndarray] = []
        per_item_summary: List[Dict[str, object]] = []

        for entry in item_entries:
            batch = entry["batch"]
            layers = extract_attention_layers(
                fusion_model,
                batch["image"],
                batch["tactile"],
                batch["padding_mask"],
            )
            shares = compute_layer_shares(layers)
            last_vec = layers[-1][0, :, 0, :].mean(dim=0).numpy()
            vectors.append(last_vec)

            if entry["split"] == "ID":
                id_shares_all.append(shares)
            else:
                ood_shares_all.append(shares)

            per_item_summary.append(
                {
                    "split": entry["split"],
                    "obj_class": entry["obj_class"],
                    "sample_idx": int(entry["sample_idx"]),
                    "avg_share": {
                        "cls": float(shares[:, 0].mean()),
                        "vision": float(shares[:, 1].mean()),
                        "tactile": float(shares[:, 2].mean()),
                    },
                }
            )

        model_to_vectors[model_name] = vectors

        if id_shares_all and ood_shares_all:
            id_mean = np.mean(np.stack(id_shares_all, axis=0), axis=0)
            ood_mean = np.mean(np.stack(ood_shares_all, axis=0), axis=0)
            model_to_mean_shares[model_name] = {"id": id_mean, "ood": ood_mean}
            mean_share_summary = {
                "id": {
                    "cls": float(id_mean[:, 0].mean()),
                    "vision": float(id_mean[:, 1].mean()),
                    "tactile": float(id_mean[:, 2].mean()),
                },
                "ood": {
                    "cls": float(ood_mean[:, 0].mean()),
                    "vision": float(ood_mean[:, 1].mean()),
                    "tactile": float(ood_mean[:, 2].mean()),
                },
            }
        else:
            mean_share_summary = {}

        model_summary[model_name] = {
            "per_item": per_item_summary,
            "mean_share": mean_share_summary,
        }

    plot_attention_heatmap_multi_models(
        model_to_vectors=model_to_vectors,
        row_labels=row_labels,
        out_path=output_dir / "attention_heatmap_comparison.png",
    )
    if model_to_mean_shares:
        plot_attention_stacked_bars_multi_models(
            model_to_shares=model_to_mean_shares,
            out_path=output_dir / "attention_stacked_bar.png",
        )

    return {
        "items": [
            {
                "split": str(e["split"]),
                "obj_class": str(e["obj_class"]),
                "sample_idx": int(e["sample_idx"]),
            }
            for e in item_entries
        ],
        "models": model_summary,
    }


@torch.no_grad()
def extract_prefusion_features(
    model: FusionModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    cache: Dict[str, torch.Tensor] = {}

    def vis_hook(_module, _inp, out):
        cache["vis"] = out.detach()

    def tac_hook(_module, _inp, out):
        cache["tac"] = out.detach()

    h_vis = model.vis_proj.register_forward_hook(vis_hook)
    h_tac = model.tac_encoder.register_forward_hook(tac_hook)

    vis_feats, tac_feats = [], []
    fusion_logits = {task: [] for task in TASKS}
    labels = {task: [] for task in TASKS}
    obj_classes: List[str] = []
    sample_idx: List[int] = []
    valid_lens: List[int] = []

    cursor = 0
    for batch in loader:
        cache.clear()
        image = batch["image"].to(device)
        tactile = batch["tactile"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        outputs = model(image, tactile, padding_mask=padding_mask)
        if "vis" not in cache or "tac" not in cache:
            raise RuntimeError("Hook failed to capture vis/tac pre-fusion features.")

        vis_map = cache["vis"]
        tac_map = cache["tac"]

        vis_tokens = vis_map.flatten(2).transpose(1, 2)
        vis_feat = vis_tokens.mean(dim=1)

        tac_tokens = tac_map.transpose(1, 2)
        down_mask = _downsample_tactile_padding_mask(padding_mask, tac_tokens.shape[1])
        valid_mask = (~down_mask).float()
        valid_counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        tac_feat = (tac_tokens * valid_mask.unsqueeze(-1)).sum(dim=1) / valid_counts

        vis_feats.append(vis_feat.cpu())
        tac_feats.append(tac_feat.cpu())

        for task in TASKS:
            fusion_logits[task].append(outputs[task].detach().cpu())
            labels[task].extend(batch[task].cpu().numpy().tolist())

        bsz = image.size(0)
        obj_classes.extend(list(batch["obj_class"]))
        valid_lens.extend(batch["valid_tactile_len"].cpu().numpy().tolist())
        sample_idx.extend(range(cursor, cursor + bsz))
        cursor += bsz

    h_vis.remove()
    h_tac.remove()

    return {
        "vision_feat": torch.cat(vis_feats, dim=0).numpy(),
        "tactile_feat": torch.cat(tac_feats, dim=0).numpy(),
        "labels": {task: np.asarray(vals, dtype=np.int64) for task, vals in labels.items()},
        "fusion_logits": {task: torch.cat(v, dim=0).numpy() for task, v in fusion_logits.items()},
        "obj_class": np.asarray(obj_classes),
        "sample_idx": np.asarray(sample_idx, dtype=np.int64),
        "valid_tactile_len": np.asarray(valid_lens, dtype=np.int64),
    }


def train_linear_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    num_classes: int,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
) -> Tuple[nn.Module, float]:
    probe = nn.Linear(train_x.shape[1], num_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(
        torch.from_numpy(train_x).float(),
        torch.from_numpy(train_y).long(),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    val_x_t = torch.from_numpy(val_x).float().to(device)
    val_y_t = torch.from_numpy(val_y).long().to(device)

    best_state = None
    best_val_acc = -1.0
    for _ in range(epochs):
        probe.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = probe(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        probe.eval()
        with torch.no_grad():
            val_logits = probe(val_x_t)
            val_pred = val_logits.argmax(dim=1)
            val_acc = (val_pred == val_y_t).float().mean().item()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}

    if best_state is not None:
        probe.load_state_dict(best_state)
    probe.eval()
    return probe, float(best_val_acc)


@torch.no_grad()
def predict_probe(probe: nn.Module, features: np.ndarray, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = torch.from_numpy(features).float().to(device)
    logits = probe(x)
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    return logits.cpu().numpy(), pred.cpu().numpy(), conf.cpu().numpy()


def save_conflict_table(rows: List[Dict[str, object]], out_path: Path) -> None:
    fieldnames = [
        "sample_idx",
        "obj_class",
        "task",
        "gt_label",
        "vision_pred",
        "vision_conf",
        "tactile_pred",
        "tactile_conf",
        "fusion_pred",
        "fusion_conf",
        "confidence_delta",
        "fusion_matches_vision",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_conflict_stats(stats: Dict[str, Dict[str, float]], out_path: Path) -> None:
    tasks = list(stats.keys())
    categories = [
        "visual_dominant_error",
        "tactile_dominant_error",
        "both_correct",
        "both_wrong",
    ]
    colors = {
        "visual_dominant_error": "#C44E52",
        "tactile_dominant_error": "#8172B2",
        "both_correct": "#55A868",
        "both_wrong": "#4C72B0",
    }

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    x = np.arange(len(tasks))
    bottom = np.zeros(len(tasks))
    for cat in categories:
        values = np.asarray([stats[t][cat] for t in tasks], dtype=np.float32)
        ax.bar(x, values, bottom=bottom, label=cat, color=colors[cat])
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Proportion")
    ax.set_title("OOD Conflict Category Distribution by Task")
    ax.grid(alpha=0.2, axis="y", linestyle="--")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_confidence_delta(
    points: List[Dict[str, object]],
    conflict_rows: List[Dict[str, object]],
    out_path: Path,
) -> None:
    x = np.asarray([p["vision_conf"] for p in points], dtype=np.float32)
    y = np.asarray([p["tactile_conf"] for p in points], dtype=np.float32)
    fusion_correct = np.asarray([p["fusion_correct"] for p in points], dtype=bool)

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    ax.scatter(
        x[fusion_correct],
        y[fusion_correct],
        s=28,
        c="#55A868",
        alpha=0.7,
        label="Fusion correct",
        edgecolors="none",
    )
    ax.scatter(
        x[~fusion_correct],
        y[~fusion_correct],
        s=28,
        c="#C44E52",
        alpha=0.7,
        label="Fusion wrong",
        edgecolors="none",
    )

    if conflict_rows:
        conflict_x = np.asarray([r["vision_conf"] for r in conflict_rows], dtype=np.float32)
        conflict_y = np.asarray([r["tactile_conf"] for r in conflict_rows], dtype=np.float32)
        ax.scatter(
            conflict_x,
            conflict_y,
            s=56,
            facecolors="none",
            edgecolors="black",
            linewidths=1.1,
            label="Conflict cases",
        )

        top_row = max(conflict_rows, key=lambda r: r["confidence_delta"])
        ax.annotate(
            f"max delta: s{top_row['sample_idx']} {top_row['task']}",
            xy=(top_row["vision_conf"], top_row["tactile_conf"]),
            xytext=(top_row["vision_conf"] + 0.02, max(0.02, top_row["tactile_conf"] - 0.08)),
            arrowprops=dict(arrowstyle="->", lw=1.0, color="black"),
            fontsize=9,
        )

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="gray", alpha=0.6)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Vision probe confidence")
    ax.set_ylabel("Tactile probe confidence")
    ax.set_title("Confidence Delta Scatter (OOD)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2, linestyle="--")
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def run_logits_conflict_analysis(
    fusion_model: FusionModel,
    data_root: Path,
    output_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
    label_maps: Dict[str, Dict[int, str]],
) -> Dict[str, object]:
    train_loader = build_analysis_loader(
        split_dir=data_root / "train",
        batch_size=args.batch_size,
        max_tactile_len=args.max_tactile_len,
        num_workers=args.num_workers,
        shuffle=False,
    )
    val_loader = build_analysis_loader(
        split_dir=data_root / "val",
        batch_size=args.batch_size,
        max_tactile_len=args.max_tactile_len,
        num_workers=args.num_workers,
        shuffle=False,
    )
    ood_loader = build_analysis_loader(
        split_dir=data_root / "ood_test",
        batch_size=args.batch_size,
        max_tactile_len=args.max_tactile_len,
        num_workers=args.num_workers,
        shuffle=False,
    )

    train_data = extract_prefusion_features(fusion_model, train_loader, device=device)
    val_data = extract_prefusion_features(fusion_model, val_loader, device=device)
    ood_data = extract_prefusion_features(fusion_model, ood_loader, device=device)

    probes: Dict[str, Dict[str, nn.Module]] = {"vision": {}, "tactile": {}}
    probe_val_acc: Dict[str, Dict[str, float]] = {"vision": {}, "tactile": {}}

    for modality, feat_key in [("vision", "vision_feat"), ("tactile", "tactile_feat")]:
        for task in TASKS:
            probe, best_val_acc = train_linear_probe(
                train_x=train_data[feat_key],
                train_y=train_data["labels"][task],
                val_x=val_data[feat_key],
                val_y=val_data["labels"][task],
                num_classes=TASK_NUM_CLASSES[task],
                device=device,
                epochs=args.probe_epochs,
                lr=args.probe_lr,
                weight_decay=args.probe_weight_decay,
                batch_size=args.probe_batch_size,
            )
            probes[modality][task] = probe
            probe_val_acc[modality][task] = best_val_acc

    ood_pred = {"vision": {}, "tactile": {}, "fusion": {}}
    ood_conf = {"vision": {}, "tactile": {}, "fusion": {}}
    ood_logits = {"vision": {}, "tactile": {}, "fusion": {}}

    for task in TASKS:
        v_logits, v_pred, v_conf = predict_probe(probes["vision"][task], ood_data["vision_feat"], device=device)
        t_logits, t_pred, t_conf = predict_probe(
            probes["tactile"][task], ood_data["tactile_feat"], device=device
        )

        fusion_logits = ood_data["fusion_logits"][task]
        fusion_probs = torch.softmax(torch.from_numpy(fusion_logits), dim=1).numpy()
        fusion_pred = fusion_probs.argmax(axis=1)
        fusion_conf = fusion_probs.max(axis=1)

        ood_logits["vision"][task] = v_logits
        ood_logits["tactile"][task] = t_logits
        ood_logits["fusion"][task] = fusion_logits

        ood_pred["vision"][task] = v_pred
        ood_pred["tactile"][task] = t_pred
        ood_pred["fusion"][task] = fusion_pred

        ood_conf["vision"][task] = v_conf
        ood_conf["tactile"][task] = t_conf
        ood_conf["fusion"][task] = fusion_conf

    conflict_rows: List[Dict[str, object]] = []
    stats_by_task: Dict[str, Dict[str, float]] = {}
    scatter_points: List[Dict[str, object]] = []

    n_samples = len(ood_data["sample_idx"])
    for task in TASKS:
        gt = ood_data["labels"][task]
        vp, tp, fp = ood_pred["vision"][task], ood_pred["tactile"][task], ood_pred["fusion"][task]
        vc, tc, fc = ood_conf["vision"][task], ood_conf["tactile"][task], ood_conf["fusion"][task]

        visual_dom = 0
        tactile_dom = 0
        both_correct = 0
        both_wrong = 0

        for i in range(n_samples):
            v_right = bool(vp[i] == gt[i])
            t_right = bool(tp[i] == gt[i])
            f_right = bool(fp[i] == gt[i])

            if not v_right and t_right and (not f_right):
                visual_dom += 1
                conflict_rows.append(
                    {
                        "sample_idx": int(ood_data["sample_idx"][i]),
                        "obj_class": str(ood_data["obj_class"][i]),
                        "task": task,
                        "gt_label": label_maps[task][int(gt[i])],
                        "vision_pred": label_maps[task][int(vp[i])],
                        "vision_conf": float(vc[i]),
                        "tactile_pred": label_maps[task][int(tp[i])],
                        "tactile_conf": float(tc[i]),
                        "fusion_pred": label_maps[task][int(fp[i])],
                        "fusion_conf": float(fc[i]),
                        "confidence_delta": float(vc[i] - tc[i]),
                        "fusion_matches_vision": bool(fp[i] == vp[i]),
                    }
                )

            if v_right and (not t_right) and (not f_right):
                tactile_dom += 1
            if v_right and t_right:
                both_correct += 1
            if (not v_right) and (not t_right):
                both_wrong += 1

            scatter_points.append(
                {
                    "task": task,
                    "sample_idx": int(ood_data["sample_idx"][i]),
                    "vision_conf": float(vc[i]),
                    "tactile_conf": float(tc[i]),
                    "fusion_correct": f_right,
                }
            )

        stats_by_task[task] = {
            "visual_dominant_error": visual_dom / n_samples,
            "tactile_dominant_error": tactile_dom / n_samples,
            "both_correct": both_correct / n_samples,
            "both_wrong": both_wrong / n_samples,
        }

    conflict_rows.sort(key=lambda row: row["confidence_delta"], reverse=True)
    save_conflict_table(conflict_rows, output_dir / "logits_conflict_table.csv")
    plot_conflict_stats(stats_by_task, output_dir / "logits_conflict_stats.png")
    plot_confidence_delta(
        points=scatter_points,
        conflict_rows=conflict_rows,
        out_path=output_dir / "logits_confidence_delta.png",
    )

    return {
        "num_ood_samples": n_samples,
        "num_conflict_rows": len(conflict_rows),
        "probe_val_accuracy": probe_val_acc,
        "stats_by_task": stats_by_task,
        "top_conflicts": conflict_rows[:10],
    }


def save_summary(summary: Dict[str, object], output_dir: Path) -> None:
    out_path = output_dir / "analysis_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_obj_class_list(multi_classes: str, fallback_single: str) -> List[str]:
    if multi_classes.strip():
        result: List[str] = []
        seen = set()
        for token in multi_classes.split(","):
            cls_name = token.strip()
            if not cls_name:
                continue
            if cls_name in seen:
                continue
            seen.add(cls_name)
            result.append(cls_name)
        if result:
            return result
    return [fallback_single]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Qualitative analysis toolkit: t-SNE/UMAP clusters, "
            "attention maps, and single-modality logits conflict probing."
        )
    )
    parser.add_argument("--data_root", type=str, default="/home/martina/Y3_Project/Plaintextdataset")
    parser.add_argument("--fusion_ckpt", type=str, default="outputs/fusion/gating/fusion_model_gating/best_model.pth")
    parser.add_argument("--fusion_ref_ckpt", type=str, default="outputs/fusion/standard/fusion_model_clean/best_model.pth")
    parser.add_argument("--tactile_ckpt", type=str, default="outputs/tactile/tactile_model_clean/best_model.pth")
    parser.add_argument("--output_dir", type=str, default="outputs/analysis/qualitative_analysis")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_tactile_len", type=int, default=3000)

    parser.add_argument("--all", action="store_true")
    parser.add_argument("--tsne", action="store_true")
    parser.add_argument("--attention", action="store_true")
    parser.add_argument("--logits-conflict", action="store_true")

    parser.add_argument("--tsne_perplexity", type=int, default=30)
    parser.add_argument("--umap_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument(
        "--tsne_compare_mode",
        type=str,
        choices=["fusion_full_vs_fusion_gated", "tactile_vs_fusion_masked"],
        default="fusion_full_vs_fusion_gated",
        help=(
            "Comparison pair for t-SNE/UMAP: "
            "fusion_full_vs_fusion_gated (recommended) or tactile_vs_fusion_masked (legacy)."
        ),
    )

    parser.add_argument("--id_obj_class", type=str, default="WoodBlock_Native")
    parser.add_argument("--ood_obj_class", type=str, default="YogaBrick_Foil_ANCHOR")
    parser.add_argument(
        "--id_obj_classes",
        type=str,
        default="",
        help="Comma-separated ID object classes used in attention-map comparison.",
    )
    parser.add_argument(
        "--ood_obj_classes",
        type=str,
        default="",
        help="Comma-separated OOD object classes used in attention-map comparison.",
    )
    parser.add_argument(
        "--attention_compare_mode",
        type=str,
        choices=["fusion_full_vs_fusion_gated", "single"],
        default="fusion_full_vs_fusion_gated",
        help="Attention-map comparison mode.",
    )

    parser.add_argument("--probe_epochs", type=int, default=15)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--probe_weight_decay", type=float, default=0.0)
    parser.add_argument("--probe_batch_size", type=int, default=128)

    # Fallback constructor args for old checkpoints with incomplete config.
    parser.add_argument("--fusion_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_layers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (args.all or args.tsne or args.attention or args.logits_conflict):
        args.all = True

    set_seed(args.seed)
    device = resolve_device(args.device)
    args.device = device

    data_root = Path(args.data_root)
    fusion_ckpt = Path(args.fusion_ckpt)
    fusion_ref_ckpt = Path(args.fusion_ref_ckpt) if args.fusion_ref_ckpt else None
    tactile_ckpt = Path(args.tactile_ckpt)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, object] = {
        "config": {
            "data_root": str(data_root),
            "fusion_ckpt": str(fusion_ckpt),
            "fusion_ref_ckpt": str(fusion_ref_ckpt) if fusion_ref_ckpt is not None else "",
            "tactile_ckpt": str(tactile_ckpt),
            "tsne_compare_mode": args.tsne_compare_mode,
            "attention_compare_mode": args.attention_compare_mode,
            "device": str(device),
            "seed": args.seed,
            "id_obj_class": args.id_obj_class,
            "ood_obj_class": args.ood_obj_class,
            "id_obj_classes": args.id_obj_classes,
            "ood_obj_classes": args.ood_obj_classes,
        }
    }

    label_maps = load_label_maps(data_root / "ood_test")

    fusion_model: Optional[FusionModel] = None
    fusion_ref_model: Optional[nn.Module] = None
    if args.all or args.tsne or args.attention or args.logits_conflict:
        fusion_model = load_fusion_model(fusion_ckpt, device, args)

    if args.all or args.tsne:
        ood_loader = build_analysis_loader(
            split_dir=data_root / "ood_test",
            batch_size=args.batch_size,
            max_tactile_len=args.max_tactile_len,
            num_workers=args.num_workers,
            shuffle=False,
        )

        if args.tsne_compare_mode == "fusion_full_vs_fusion_gated":
            if fusion_ref_ckpt is None:
                raise ValueError("--fusion_ref_ckpt is required for fusion_full_vs_fusion_gated mode.")
            if fusion_ref_model is None:
                fusion_ref_model = load_fusion_model(fusion_ref_ckpt, device, args)
            summary["tsne_umap"] = run_tsne_umap_analysis(
                model_a=fusion_ref_model,
                model_b=fusion_model,
                model_a_kind="fusion_full",
                model_b_kind="fusion_full",
                model_a_name="Fusion Full",
                model_b_name="Fusion Gated",
                ood_loader=ood_loader,
                label_maps=label_maps,
                output_dir=output_dir,
                args=args,
            )
        else:
            tactile_model = load_tactile_model(tactile_ckpt, device, args)
            summary["tsne_umap"] = run_tsne_umap_analysis(
                model_a=tactile_model,
                model_b=fusion_model,
                model_a_kind="tactile_only",
                model_b_kind="fusion_tactile_only",
                model_a_name="TactileOnlyModel",
                model_b_name="FusionModel (visual masked)",
                ood_loader=ood_loader,
                label_maps=label_maps,
                output_dir=output_dir,
                args=args,
            )
        save_summary(summary, output_dir)
        print("[done] t-SNE + UMAP analysis")

    if args.all or args.attention:
        id_obj_classes = resolve_obj_class_list(args.id_obj_classes, args.id_obj_class)
        ood_obj_classes = resolve_obj_class_list(args.ood_obj_classes, args.ood_obj_class)

        if args.attention_compare_mode == "fusion_full_vs_fusion_gated":
            if fusion_ref_ckpt is None:
                raise ValueError("--fusion_ref_ckpt is required for fusion_full_vs_fusion_gated mode.")
            if fusion_ref_model is None:
                fusion_ref_model = load_fusion_model(fusion_ref_ckpt, device, args)
            attention_models: Dict[str, nn.Module] = {
                "Fusion Full": fusion_ref_model,
                "Fusion Gated": fusion_model,
            }
        else:
            attention_models = {"Fusion Model": fusion_model}

        summary["attention"] = run_attention_analysis(
            models=attention_models,
            data_root=data_root,
            output_dir=output_dir,
            device=device,
            id_obj_classes=id_obj_classes,
            ood_obj_classes=ood_obj_classes,
            max_tactile_len=args.max_tactile_len,
        )
        save_summary(summary, output_dir)
        print("[done] Attention map analysis")

    if args.all or args.logits_conflict:
        summary["logits_conflict"] = run_logits_conflict_analysis(
            fusion_model=fusion_model,
            data_root=data_root,
            output_dir=output_dir,
            device=device,
            args=args,
            label_maps=label_maps,
        )
        save_summary(summary, output_dir)
        print("[done] Logits conflict analysis")

    save_summary(summary, output_dir)
    print(f"Saved summary: {output_dir / 'analysis_summary.json'}")


if __name__ == "__main__":
    main()
