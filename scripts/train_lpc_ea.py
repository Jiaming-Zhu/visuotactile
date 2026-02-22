"""
LPC-EA: Latent Predictive Coding with Evidential Accumulation (Streaming).

Key ideas implemented in this script:
1) Visual prior encoder: single pre-grasp RGB image -> latent z_vis (static prior).
2) Streaming proprio encoder: causal sliding windows over proprioceptive signals -> z_kin^(t).
3) Latent forward dynamics: conditioned on z_vis.detach() to avoid visual gradient pollution.
4) Soft-temporal residual gating: compute "surprise" between predicted and observed latent trajectories
   using a diagonal-wavefront (vectorized) Soft-DTW implementation in pure PyTorch.
5) Evidential Deep Learning (EDL): accumulate evidence over time and output Dirichlet parameters.

This is designed as a production-ready research baseline compatible with the existing dataset layout:
Plaintextdataset/{train,val,test,ood_test}/<Class>/<Episode>/{visual_anchor.jpg,tactile_data.pkl}.

NOTE on performance:
- Soft-DTW is still O(k^2) DP. We avoid nested Python loops by using diagonal wavefront updates,
  but for large rollout_k this will become a bottleneck. Keep rollout_k small (default 8).
"""

import argparse
import json
import os
import pickle
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


TACTILE_STATS = {
    "joint_position": {"mean": 21.70, "std": 38.13},
    "joint_load": {"mean": 7.21, "std": 14.03},
    "joint_current": {"mean": 52.56, "std": 133.43},
    "joint_velocity": {"mean": 0.13, "std": 9.79},
}


TASKS = ("mass", "stiffness", "material")


@dataclass
class Sample:
    img_path: Path
    tactile_path: Path
    labels: Dict[str, int]


class RoboticGraspDataset(Dataset):
    def __init__(
        self,
        split_dir: Path,
        max_tactile_len: int = 3000,
        tactile_stats: Optional[Dict[str, Dict[str, float]]] = None,
        strict_data: bool = False,
        max_data_warnings: int = 10,
    ) -> None:
        self.split_dir = Path(split_dir)
        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        props_file = self.split_dir / "physical_properties.json"
        if not props_file.exists():
            raise FileNotFoundError(f"Missing properties file: {props_file}")
        props_config = json.loads(props_file.read_text())

        self.properties = props_config["properties"]
        self.mass_to_idx = props_config["mass_to_idx"]
        self.stiffness_to_idx = props_config["stiffness_to_idx"]
        self.material_to_idx = props_config["material_to_idx"]
        self.max_tactile_len = max_tactile_len
        self.tactile_stats = tactile_stats or TACTILE_STATS
        self.strict_data = bool(strict_data)
        self.max_data_warnings = int(max_data_warnings)
        self._data_warnings_emitted = 0
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.samples = self._collect_samples()

    def _parse_labels(self, obj_class: str) -> Optional[Dict[str, int]]:
        if obj_class not in self.properties:
            return None
        props = self.properties[obj_class]
        mass_idx = self.mass_to_idx.get(props["mass"])
        stiffness_idx = self.stiffness_to_idx.get(props["stiffness"])
        material_idx = self.material_to_idx.get(props["material"])
        if mass_idx is None or stiffness_idx is None or material_idx is None:
            return None
        return {"mass": mass_idx, "stiffness": stiffness_idx, "material": material_idx}

    def _collect_samples(self) -> List[Sample]:
        samples: List[Sample] = []
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
                    samples.append(Sample(img_path=img_path, tactile_path=tactile_path, labels=labels))
        if not samples:
            raise RuntimeError(f"No valid samples found in {self.split_dir}")
        return samples

    def _normalize(self, arr: np.ndarray, key: str) -> np.ndarray:
        mean = self.tactile_stats[key]["mean"]
        std = self.tactile_stats[key]["std"]
        return (np.asarray(arr) - mean) / (std + 1e-8)

    def _load_tactile(self, tactile_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        with open(tactile_path, "rb") as f:
            data = pickle.load(f)

        joint_pos = self._normalize(data["joint_position_profile"], "joint_position")
        joint_load = self._normalize(data["joint_load_profile"], "joint_load")
        joint_current = self._normalize(data["joint_current_profile"], "joint_current")
        joint_vel = self._normalize(data["joint_velocity_profile"], "joint_velocity")
        tactile = np.concatenate([joint_pos, joint_load, joint_current, joint_vel], axis=1).T

        t = tactile.shape[1]
        valid_len = min(t, self.max_tactile_len)
        tactile_tensor = torch.zeros((24, self.max_tactile_len), dtype=torch.float32)
        tactile_tensor[:, :valid_len] = torch.tensor(tactile[:, :valid_len], dtype=torch.float32)

        padding_mask = torch.zeros(self.max_tactile_len, dtype=torch.bool)
        padding_mask[valid_len:] = True
        return tactile_tensor, padding_mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        try:
            image = Image.open(sample.img_path).convert("RGB")
            image = self.transform(image)
        except Exception as exc:
            if self.strict_data:
                raise RuntimeError(f"Failed to load image: {sample.img_path}") from exc
            if self._data_warnings_emitted < self.max_data_warnings:
                warnings.warn(
                    f"Failed to load image {sample.img_path}: {exc}. Using zeros.",
                    category=RuntimeWarning,
                    stacklevel=2,
                )
                self._data_warnings_emitted += 1
                if self._data_warnings_emitted == self.max_data_warnings:
                    warnings.warn(
                        "Max data warnings reached; further image-load warnings will be suppressed.",
                        category=RuntimeWarning,
                        stacklevel=2,
                    )
            image = torch.zeros((3, 224, 224), dtype=torch.float32)

        tactile, padding_mask = self._load_tactile(sample.tactile_path)
        return {
            "image": image,
            "tactile": tactile,
            "padding_mask": padding_mask,
            "mass": torch.tensor(sample.labels["mass"], dtype=torch.long),
            "stiffness": torch.tensor(sample.labels["stiffness"], dtype=torch.long),
            "material": torch.tensor(sample.labels["material"], dtype=torch.long),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    requested = (device_arg or "").strip().lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available in current runtime, falling back to CPU.")
        return torch.device("cpu")
    if not requested:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_loader(
    split_dir: Path,
    batch_size: int,
    max_tactile_len: int,
    num_workers: int,
    shuffle: bool,
    strict_data: bool,
    max_data_warnings: int,
) -> DataLoader:
    dataset = RoboticGraspDataset(
        split_dir=split_dir,
        max_tactile_len=max_tactile_len,
        strict_data=strict_data,
        max_data_warnings=max_data_warnings,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
    )


def _plot_confusion_matrix(labels, preds, label_names, task_name: str, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
    except ImportError as exc:
        raise ImportError("eval plotting requires matplotlib, seaborn, scikit-learn") from exc

    all_class_labels = list(range(len(label_names)))
    cm = confusion_matrix(labels, preds, labels=all_class_labels)
    cm_normalized = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)

    annot_labels = []
    for i in range(cm.shape[0]):
        row = []
        for j in range(cm.shape[1]):
            row.append(f"{cm[i, j]}\n({cm_normalized[i, j]:.1%})")
        annot_labels.append(row)
    annot_labels = np.asarray(annot_labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=annot_labels,
        fmt="",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
        cbar=True,
        square=True,
        annot_kws={"fontsize": 10},
    )
    ax.set_title(f"{task_name.upper()} Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(output_dir / f"confusion_matrix_{task_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_summary(results: Dict[str, Dict], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("eval summary plotting requires matplotlib") from exc

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Model Evaluation Summary", fontsize=14, fontweight="bold")

    tasks = list(results.keys())
    colors = ["#2ecc71", "#3498db", "#e74c3c"]

    accs = [results[t]["accuracy"] for t in tasks]
    axes[0].bar(tasks, accs, color=colors[: len(tasks)])
    axes[0].set_title("Accuracy by Task")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Accuracy")
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.02, f"{v:.2%}", ha="center", fontweight="bold")

    f1s_macro = [results[t]["macro"]["f1"] for t in tasks]
    f1s_weighted = [results[t]["weighted"]["f1"] for t in tasks]
    x = np.arange(len(tasks))
    width = 0.35
    axes[1].bar(x - width / 2, f1s_macro, width, label="Macro", color="#3498db")
    axes[1].bar(x + width / 2, f1s_weighted, width, label="Weighted", color="#e74c3c")
    axes[1].set_title("F1-Score by Task")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tasks)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("F1-Score")
    axes[1].legend()

    for i, task in enumerate(tasks):
        axes[2].scatter(
            results[task]["macro"]["recall"],
            results[task]["macro"]["precision"],
            s=200,
            c=colors[i % len(colors)],
            label=task,
            marker="o",
        )
    axes[2].set_title("Precision vs Recall (Macro)")
    axes[2].set_xlabel("Recall")
    axes[2].set_ylabel("Precision")
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "evaluation_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_training_curves(history: List[Dict], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("training curve plotting requires matplotlib") from exc

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train"]["loss"] for h in history]
    val_loss = [h["val"]["loss"] for h in history]
    val_mass = [h["val"]["mass"] for h in history]
    val_stiffness = [h["val"]["stiffness"] for h in history]
    val_material = [h["val"]["material"] for h in history]
    val_avg = [(m + s + mt) / 3.0 for m, s, mt in zip(val_mass, val_stiffness, val_material)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, train_loss, "b-", linewidth=2, label="Train Loss")
    axes[0].plot(epochs, val_loss, "r-", linewidth=2, label="Val Loss")
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training/Validation Loss", fontsize=14, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, val_mass, "r-", linewidth=2, label="Mass")
    axes[1].plot(epochs, val_stiffness, "g-", linewidth=2, label="Stiffness")
    axes[1].plot(epochs, val_material, "b-", linewidth=2, label="Material")
    axes[1].plot(epochs, val_avg, "k--", linewidth=2, label="Average")
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Validation Accuracy", fontsize=12)
    axes[1].set_title("Validation Accuracy by Task", fontsize=14, fontweight="bold")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


class LiveTrainingPlotter:
    """Interactive epoch-level training curves (loss + validation accuracy)."""

    def __init__(self, output_dir: Path):
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("live plotting requires matplotlib") from exc

        self.plt = plt
        self.output_dir = output_dir
        self.plt.ion()
        self.fig, self.axes = self.plt.subplots(1, 2, figsize=(14, 5))
        self.fig.suptitle("LPC-EA Training (Live)", fontsize=14, fontweight="bold")

        self.epochs: List[int] = []
        self.train_loss: List[float] = []
        self.val_loss: List[float] = []
        self.val_mass: List[float] = []
        self.val_stiffness: List[float] = []
        self.val_material: List[float] = []

    def update(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
        self.epochs.append(epoch)
        self.train_loss.append(train_metrics["loss"])
        self.val_loss.append(val_metrics["loss"])
        self.val_mass.append(val_metrics["mass"])
        self.val_stiffness.append(val_metrics["stiffness"])
        self.val_material.append(val_metrics["material"])
        val_avg = [(m + s + t) / 3.0 for m, s, t in zip(self.val_mass, self.val_stiffness, self.val_material)]

        ax0, ax1 = self.axes
        ax0.clear()
        ax0.plot(self.epochs, self.train_loss, "b-", linewidth=2, label="Train Loss")
        ax0.plot(self.epochs, self.val_loss, "r-", linewidth=2, label="Val Loss")
        ax0.set_title("Loss", fontsize=13, fontweight="bold")
        ax0.set_xlabel("Epoch")
        ax0.set_ylabel("Loss")
        ax0.grid(True, alpha=0.3)
        ax0.legend()

        ax1.clear()
        ax1.plot(self.epochs, self.val_mass, "r-", linewidth=2, label="Mass")
        ax1.plot(self.epochs, self.val_stiffness, "g-", linewidth=2, label="Stiffness")
        ax1.plot(self.epochs, self.val_material, "b-", linewidth=2, label="Material")
        ax1.plot(self.epochs, val_avg, "k--", linewidth=2, label="Average")
        ax1.set_title("Validation Accuracy", fontsize=13, fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.plt.pause(0.01)

    def save(self, filename: str = "training_curves_live.png") -> None:
        self.fig.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")

    def close(self) -> None:
        self.plt.ioff()
        self.plt.close(self.fig)


class VisualPriorEncoder(nn.Module):
    def __init__(self, latent_dim: int, freeze_backbone: bool = True) -> None:
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.freeze_backbone = bool(freeze_backbone)
        self.proj = nn.Sequential(
            nn.Linear(512, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        # Freeze BN running stats when using a "frozen visual prior".
        # Without this, BatchNorm running_mean/var will drift in train() even if weights are frozen.
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(images)
        return self.proj(feats)


class TransformerProprioceptiveEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        input_dim: int = 24,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_window_size: int = 256,
    ) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
        )
        downsampled_len = max_window_size // 8 + 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, latent_dim))
        self.pos_emb = nn.Parameter(torch.randn(1, downsampled_len + 1, latent_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        # windows: (B, w, 24)
        B = windows.size(0)
        
        x = windows.transpose(1, 2)  # (B, 24, w)
        x = self.cnn(x)              # (B, latent_dim, w')
        x = x.transpose(1, 2)        # (B, w', latent_dim)
        
        seq_len = x.size(1)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = x + self.pos_emb[:, :seq_len + 1, :]
        x = self.transformer(x)
        
        z = x[:, 0, :]
        return self.norm(z)


class LatentForwardDynamics(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or (latent_dim * 2)
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, z_vis_detached: torch.Tensor, z_state: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_vis_detached, z_state], dim=-1)
        return self.net(x)


class EvidenceHeads(nn.Module):
    def __init__(self, latent_dim: int, task_dims: Dict[str, int], hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.vis = nn.ModuleDict()
        self.kin = nn.ModuleDict()
        for task, k in task_dims.items():
            self.vis[task] = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, k),
            )
            self.kin[task] = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, k),
            )

    def forward_vis(self, z_vis: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {task: self.vis[task](z_vis) for task in self.vis}

    def forward_kin(self, z_kin: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {task: self.kin[task](z_kin) for task in self.kin}


class LPCEAModel(nn.Module):
    def __init__(self, latent_dim: int, task_dims: Dict[str, int], freeze_visual_backbone: bool = True) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.visual = VisualPriorEncoder(latent_dim=latent_dim, freeze_backbone=freeze_visual_backbone)
        self.proprio = TransformerProprioceptiveEncoder(
            latent_dim=latent_dim,
            input_dim=24,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
            max_window_size=256,
        )
        self.forward_model = LatentForwardDynamics(latent_dim=latent_dim)
        self.evidence = EvidenceHeads(latent_dim=latent_dim, task_dims=task_dims)
        self.gate_gamma = nn.Parameter(torch.tensor([1.0]))

    def encode_visual(self, images: torch.Tensor) -> torch.Tensor:
        return self.visual(images)

    def encode_windows(self, windows: torch.Tensor) -> torch.Tensor:
        # windows: (B*T, w, 24)
        return self.proprio(windows)


def _num_stream_steps(valid_len: torch.Tensor, window_size: int, step_size: int) -> torch.Tensor:
    valid_len = valid_len.to(torch.long)
    enough = valid_len >= window_size
    steps = torch.where(enough, (valid_len - window_size) // step_size + 1, torch.zeros_like(valid_len))
    return steps


def _make_stream_windows(tactile: torch.Tensor, window_size: int, step_size: int) -> torch.Tensor:
    # tactile: (B, 24, L)
    # returns windows: (B, T_max, w, 24)
    windows = tactile.unfold(dimension=2, size=window_size, step=step_size)  # (B, 24, T, w)
    windows = windows.permute(0, 2, 3, 1).contiguous()  # (B, T, w, 24)
    return windows


def _pairwise_squared_euclidean(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # x,y: (B, k, D) -> (B, k, k)
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1).unsqueeze(1)
    xy = torch.matmul(x, y.transpose(1, 2))
    dist = x2 + y2 - 2.0 * xy
    return dist.clamp_min(0.0)


def soft_dtw_distance(x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Batched Soft-DTW distance using diagonal wavefront DP.

    Args:
        x: (B, k, D)
        y: (B, k, D)
        gamma: smoothing parameter (>0). Smaller -> closer to hard DTW.

    Returns:
        dist: (B,)
    """
    if gamma <= 0:
        raise ValueError("soft_dtw gamma must be > 0")
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError("soft_dtw expects (B,k,D) inputs")
    if x.shape != y.shape:
        raise ValueError(f"soft_dtw requires x and y have same shape, got {x.shape} vs {y.shape}")

    bsz, k, _ = x.shape
    device = x.device
    dtype = x.dtype

    c = _pairwise_squared_euclidean(x, y)  # (B, k, k)

    inf = torch.tensor(float("inf"), device=device, dtype=dtype)
    r = torch.full((bsz, k + 1, k + 1), inf, device=device, dtype=dtype)
    r[:, 0, 0] = 0.0

    gamma_t = torch.tensor(gamma, device=device, dtype=dtype)

    # Diagonals where i,j in [1..k] and i+j = d.
    for d in range(2, 2 * k + 1):
        i_start = max(1, d - k)
        i_end = min(k, d - 1)
        i_idx = torch.arange(i_start, i_end + 1, device=device)
        j_idx = d - i_idx

        r0 = r[:, i_idx - 1, j_idx]  # up
        r1 = r[:, i_idx, j_idx - 1]  # left
        r2 = r[:, i_idx - 1, j_idx - 1]  # diag

        stacked = torch.stack([r0, r1, r2], dim=-1)  # (B, L, 3)
        softmin = -gamma_t * torch.logsumexp(-stacked / gamma_t, dim=-1)  # (B, L)

        r[:, i_idx, j_idx] = c[:, i_idx - 1, j_idx - 1] + softmin

    return r[:, k, k]


def dirichlet_kl_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """
    KL(Dir(alpha) || Dir(1)).
    alpha: (B, K), alpha > 0
    returns: (B,)
    """
    if alpha.ndim != 2:
        raise ValueError("alpha must be (B,K)")
    k = alpha.size(1)
    sum_alpha = alpha.sum(dim=1, keepdim=True)
    log_b_alpha = torch.lgamma(sum_alpha) - torch.lgamma(alpha).sum(dim=1, keepdim=True)
    log_b_uniform = torch.lgamma(torch.tensor(float(k), device=alpha.device, dtype=alpha.dtype))
    digamma_sum = torch.digamma(sum_alpha)
    digamma_alpha = torch.digamma(alpha)
    kl = log_b_alpha - log_b_uniform + ((alpha - 1.0) * (digamma_alpha - digamma_sum)).sum(dim=1, keepdim=True)
    return kl.squeeze(1)


def edl_dirichlet_classification_loss(
    alpha: torch.Tensor,
    target: torch.Tensor,
    epoch: int,
    kl_anneal_epochs: int,
    beta_kl: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sensoy et al. (2018) classification evidential loss:
      L = A + anneal * beta_kl * KL(Dir(tilde_alpha) || Dir(1))
    where A = sum_j y_j (psi(S) - psi(alpha_j)) and tilde_alpha removes evidence for the true class.

    Returns:
        loss_mean, a_term_mean, kl_term_mean
    """
    if alpha.ndim != 2:
        raise ValueError("alpha must be (B,K)")
    if target.ndim != 1:
        raise ValueError("target must be (B,)")
    if alpha.size(0) != target.size(0):
        raise ValueError("alpha and target batch size mismatch")

    bsz, num_classes = alpha.shape
    target = target.to(torch.long)
    y = F.one_hot(target, num_classes=num_classes).to(dtype=alpha.dtype)

    sum_alpha = alpha.sum(dim=1)  # (B,)
    alpha_y = alpha.gather(1, target.view(-1, 1)).squeeze(1)  # (B,)
    a_term = torch.digamma(sum_alpha) - torch.digamma(alpha_y)  # (B,)

    tilde_alpha = (alpha - 1.0) * (1.0 - y) + 1.0
    kl_term = dirichlet_kl_to_uniform(tilde_alpha)  # (B,)

    anneal = min(1.0, float(epoch) / max(1, int(kl_anneal_epochs)))
    loss = a_term + (anneal * float(beta_kl)) * kl_term
    return loss.mean(), a_term.mean(), kl_term.mean()


def _sample_nce_pairs(
    num_steps: torch.Tensor, pairs_per_seq: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    b_indices: List[torch.Tensor] = []
    t_indices: List[torch.Tensor] = []
    for b, steps in enumerate(num_steps.tolist()):
        max_t = int(steps) - 2
        if max_t < 0:
            continue
        n_candidates = max_t + 1
        if n_candidates >= pairs_per_seq:
            t = torch.randperm(n_candidates, device=device)[:pairs_per_seq]
        else:
            t = torch.randint(0, n_candidates, (pairs_per_seq,), device=device)
        b_indices.append(torch.full((pairs_per_seq,), b, device=device, dtype=torch.long))
        t_indices.append(t.to(torch.long))
    if not b_indices:
        return torch.empty((0,), device=device, dtype=torch.long), torch.empty((0,), device=device, dtype=torch.long)
    return torch.cat(b_indices, dim=0), torch.cat(t_indices, dim=0)


def _rollout_forward(
    forward_model: LatentForwardDynamics, z_vis_detached: torch.Tensor, z0: torch.Tensor, rollout_k: int
) -> torch.Tensor:
    cur = z0
    preds: List[torch.Tensor] = []
    for _ in range(int(rollout_k)):
        cur = forward_model(z_vis_detached, cur)
        preds.append(cur)
    return torch.stack(preds, dim=1)  # (B, k, D)


def _compute_gate(
    forward_model: LatentForwardDynamics,
    z_vis_detached: torch.Tensor,
    z_kin: torch.Tensor,
    num_steps: torch.Tensor,
    rollout_k: int,
    surprise_metric: str,
    gate_gamma: torch.Tensor,
    soft_dtw_gamma: float,
    t0_strategy: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes surprise r and gate g=exp(-gate_gamma*r).

    For sequences with insufficient steps (< rollout_k+1), returns r=0, g=1 (skip gating).
    """
    device = z_kin.device
    bsz, t_max, latent_dim = z_kin.shape

    r = torch.zeros((bsz,), device=device, dtype=z_kin.dtype)
    g = torch.ones((bsz,), device=device, dtype=z_kin.dtype)

    rollout_k = int(rollout_k)
    if rollout_k <= 0:
        return r, g

    valid_gate = num_steps >= (rollout_k + 1)
    if not valid_gate.any():
        return r, g

    v_idx = torch.nonzero(valid_gate, as_tuple=False).squeeze(1)
    num_steps_v = num_steps[v_idx].to(torch.long)
    max_t0_v = num_steps_v - rollout_k - 1  # >= 0 by construction

    if t0_strategy == "random":
        t0_v = torch.floor(torch.rand_like(max_t0_v.to(torch.float32)) * (max_t0_v.to(torch.float32) + 1.0)).to(
            torch.long
        )
    elif t0_strategy == "first":
        t0_v = torch.zeros_like(max_t0_v)
    elif t0_strategy == "middle":
        t0_v = max_t0_v // 2
    elif t0_strategy == "last":
        t0_v = max_t0_v
    else:
        raise ValueError(f"Unknown gate t0 strategy: {t0_strategy}")

    z0_v = z_kin[v_idx, t0_v]  # (Bv, D)
    idx = t0_v.unsqueeze(1) + torch.arange(1, rollout_k + 1, device=device).view(1, -1)  # (Bv, k)
    z_true_v = torch.gather(z_kin[v_idx], dim=1, index=idx.unsqueeze(-1).expand(-1, rollout_k, latent_dim))
    z_pred_v = _rollout_forward(forward_model, z_vis_detached[v_idx], z0_v, rollout_k=rollout_k)

    # Normalize latents before computing distances so r is scale-stable w.r.t. latent_dim.
    z_true_m = F.normalize(z_true_v, dim=-1)
    z_pred_m = F.normalize(z_pred_v, dim=-1)

    if surprise_metric == "l2":
        # Already averages over time, giving a per-step surprise in [0, 4] for unit-normalized vectors.
        r_v = (z_pred_m - z_true_m).pow(2).sum(dim=-1).mean(dim=-1)
    elif surprise_metric == "soft_dtw":
        # Soft-DTW returns an accumulated path cost ~ O(k); normalize by k to match the per-step scale.
        r_v = soft_dtw_distance(z_pred_m, z_true_m, gamma=float(soft_dtw_gamma))
        r_v = r_v / float(max(1, int(rollout_k)))
    else:
        raise ValueError(f"Unknown surprise_metric: {surprise_metric}")

    r[v_idx] = r_v
    g[v_idx] = torch.clamp(torch.exp(-gate_gamma * r_v), min=0.0, max=1.0)
    return r, g


def _forward_batch(
    model: LPCEAModel,
    images: torch.Tensor,
    tactile: torch.Tensor,
    padding_mask: torch.Tensor,
    labels: Optional[Dict[str, torch.Tensor]],
    args: argparse.Namespace,
    epoch: int,
    compute_nce: bool,
    train_mode: bool,
) -> Dict[str, torch.Tensor]:
    device = images.device
    bsz = images.size(0)

    valid_len = (~padding_mask).sum(dim=1).to(torch.long)  # (B,)
    num_steps = _num_stream_steps(valid_len, window_size=args.window_size, step_size=args.step_size)  # (B,)

    if args.window_size > tactile.size(-1):
        raise ValueError(f"window_size ({args.window_size}) must be <= tactile length ({tactile.size(-1)})")

    windows = _make_stream_windows(tactile, window_size=args.window_size, step_size=args.step_size)  # (B,T,w,24)
    t_max = windows.size(1)
    steps_mask = (torch.arange(t_max, device=device).view(1, -1) < num_steps.view(-1, 1))  # (B,T)

    z_vis = model.encode_visual(images)  # (B,D)
    windows_flat = windows.view(bsz * t_max, args.window_size, 24)
    z_kin_flat = model.encode_windows(windows_flat)  # (B*T,D)
    z_kin = z_kin_flat.view(bsz, t_max, args.latent_dim)  # (B,T,D)

    # Surprise gating (detach visual prior for dynamics).
    r, g = _compute_gate(
        forward_model=model.forward_model,
        z_vis_detached=z_vis.detach(),
        z_kin=z_kin,
        num_steps=num_steps,
        rollout_k=args.rollout_k,
        surprise_metric=args.surprise_metric,
        gate_gamma=model.gate_gamma,
        soft_dtw_gamma=args.soft_dtw_gamma,
        t0_strategy=args.train_gate_t0_strategy if train_mode else args.eval_gate_t0_strategy,
    )

    # Evidence from vision (static) and kinematics (stream).
    e_vis_raw = model.evidence.forward_vis(z_vis)
    e_kin_raw = model.evidence.forward_kin(z_kin_flat)

    alpha: Dict[str, torch.Tensor] = {}
    prob: Dict[str, torch.Tensor] = {}
    uncertainty: Dict[str, torch.Tensor] = {}

    for task in TASKS:
        e_vis = F.softplus(e_vis_raw[task]) * g.view(-1, 1)  # (B,K)
        e_kin_all = F.softplus(e_kin_raw[task]).view(bsz, t_max, -1)  # (B,T,K)
        e_kin_all = e_kin_all * steps_mask.to(e_kin_all.dtype).unsqueeze(-1)
        e_kin_sum = e_kin_all.sum(dim=1)  # (B,K)

        a = 1.0 + e_vis + e_kin_sum
        alpha[task] = a
        s = a.sum(dim=1, keepdim=True)
        p = a / (s + 1e-8)
        prob[task] = p
        uncertainty[task] = float(a.size(1)) / (s.squeeze(1) + 1e-8)

    out: Dict[str, torch.Tensor] = {
        "z_vis": z_vis,
        "z_kin": z_kin,
        "num_steps": num_steps,
        "surprise": r,
        "gate": g,
    }
    for task in TASKS:
        out[f"alpha_{task}"] = alpha[task]
        out[f"prob_{task}"] = prob[task]
        out[f"unc_{task}"] = uncertainty[task]

    if labels is None:
        return out

    # EDL losses (Sensoy 2018): A + annealed KL(tilde_alpha || 1).
    loss_edl = torch.tensor(0.0, device=device)
    loss_a = torch.tensor(0.0, device=device)
    loss_kl = torch.tensor(0.0, device=device)
    for task in TASKS:
        l, a_term, kl_term = edl_dirichlet_classification_loss(
            alpha=alpha[task],
            target=labels[task],
            epoch=epoch,
            kl_anneal_epochs=args.kl_anneal_epochs,
            beta_kl=args.beta_kl,
        )
        loss_edl = loss_edl + l
        loss_a = loss_a + a_term
        loss_kl = loss_kl + kl_term

    # InfoNCE on latent forward dynamics (optional in eval).
    loss_nce = torch.tensor(0.0, device=device)
    if compute_nce and float(args.lambda_nce) > 0.0:
        b_idx, t_idx = _sample_nce_pairs(num_steps=num_steps, pairs_per_seq=args.nce_pairs_per_seq, device=device)
        if b_idx.numel() > 0:
            z_t = z_kin[b_idx, t_idx]  # (N,D)
            z_next = z_kin[b_idx, t_idx + 1]  # (N,D)
            z_vis_cond = z_vis.detach()[b_idx]  # (N,D)
            hat_next = model.forward_model(z_vis_cond, z_t)  # (N,D)

            q = F.normalize(hat_next, dim=1)
            k = F.normalize(z_next, dim=1)
            logits = torch.matmul(q, k.t()) / float(args.temperature)  # (N,N)
            labels_nce = torch.arange(logits.size(0), device=device)
            loss_nce = F.cross_entropy(logits, labels_nce)

    loss_total = loss_edl + float(args.lambda_nce) * loss_nce
    out.update(
        {
            "loss_total": loss_total,
            "loss_edl": loss_edl,
            "loss_a": loss_a,
            "loss_kl": loss_kl,
            "loss_nce": loss_nce,
        }
    )
    return out


def compute_metrics(
    model: LPCEAModel,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    train_mode: bool,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, float]:
    if train_mode:
        model.train()
    else:
        model.eval()

    total_samples = 0
    totals: Dict[str, float] = {
        "loss": 0.0,
        "loss_edl": 0.0,
        "loss_a": 0.0,
        "loss_kl": 0.0,
        "loss_nce": 0.0,
        "gate": 0.0,
        "surprise": 0.0,
    }
    correct = {task: 0 for task in TASKS}

    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, leave=False, desc="train" if train_mode else "eval")

    for batch_idx, batch in enumerate(iterator, start=1):
        images = batch["image"].to(device)
        tactile = batch["tactile"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        labels = {task: batch[task].to(device) for task in TASKS}

        if train_mode:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            out = _forward_batch(
                model=model,
                images=images,
                tactile=tactile,
                padding_mask=padding_mask,
                labels=labels,
                args=args,
                epoch=epoch,
                compute_nce=True,
                train_mode=True,
            )
            loss = out["loss_total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        else:
            with torch.no_grad():
                out = _forward_batch(
                    model=model,
                    images=images,
                    tactile=tactile,
                    padding_mask=padding_mask,
                    labels=labels,
                    args=args,
                    epoch=epoch,
                    compute_nce=False,
                    train_mode=False,
                )
                loss = out["loss_total"]

        bsz = images.size(0)
        total_samples += bsz

        totals["loss"] += float(loss.item()) * bsz
        totals["loss_edl"] += float(out["loss_edl"].item()) * bsz
        totals["loss_a"] += float(out["loss_a"].item()) * bsz
        totals["loss_kl"] += float(out["loss_kl"].item()) * bsz
        totals["loss_nce"] += float(out["loss_nce"].item()) * bsz
        totals["gate"] += float(out["gate"].mean().item()) * bsz
        totals["surprise"] += float(out["surprise"].mean().item()) * bsz

        for task in TASKS:
            pred = out[f"prob_{task}"].argmax(dim=1)
            correct[task] += (pred == labels[task]).sum().item()

        if tqdm is not None and hasattr(iterator, "set_postfix"):
            running_loss = totals["loss"] / max(1, total_samples)
            postfix = {
                "loss": f"{running_loss:.4f}",
                "gate": f"{totals['gate'] / max(1, total_samples):.3f}",
                "step": batch_idx,
            }
            for task in TASKS:
                postfix[task] = f"{correct[task] / max(1, total_samples):.2%}"
            iterator.set_postfix(postfix)

    metrics = {
        "loss": totals["loss"] / max(1, total_samples),
        "loss_edl": totals["loss_edl"] / max(1, total_samples),
        "loss_a": totals["loss_a"] / max(1, total_samples),
        "loss_kl": totals["loss_kl"] / max(1, total_samples),
        "loss_nce": totals["loss_nce"] / max(1, total_samples),
        "gate": totals["gate"] / max(1, total_samples),
        "surprise": totals["surprise"] / max(1, total_samples),
    }
    for task in TASKS:
        metrics[task] = correct[task] / max(1, total_samples)
    return metrics


def eval_split(args: argparse.Namespace, split_name: str, checkpoint_path: Optional[Path] = None) -> Dict[str, float]:
    try:
        from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
    except ImportError as exc:
        raise ImportError("eval requires scikit-learn") from exc

    ckpt_path = Path(checkpoint_path) if checkpoint_path is not None else Path(args.checkpoint)
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    ckpt_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

    eval_args = argparse.Namespace(**vars(args))
    # In eval mode, default to checkpoint training-time structural/runtime parameters
    # so users do not need to manually pass matching flags.
    sync_fields = [
        "latent_dim",
        "window_size",
        "step_size",
        "rollout_k",
        "gamma",
        "soft_dtw_gamma",
        "surprise_metric",
        "max_tactile_len",
        "freeze_visual_backbone",
    ]
    for field in sync_fields:
        if field in ckpt_config:
            setattr(eval_args, field, ckpt_config[field])

    data_root = Path(eval_args.data_root)
    split_dir = data_root / split_name
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split dir not found: {split_dir}")

    loader = build_loader(
        split_dir,
        eval_args.batch_size,
        eval_args.max_tactile_len,
        eval_args.num_workers,
        shuffle=False,
        strict_data=eval_args.strict_data,
        max_data_warnings=eval_args.max_data_warnings,
    )
    dataset = loader.dataset

    task_dims = {
        "mass": len(dataset.mass_to_idx),
        "stiffness": len(dataset.stiffness_to_idx),
        "material": len(dataset.material_to_idx),
    }

    model = LPCEAModel(
        latent_dim=eval_args.latent_dim,
        task_dims=task_dims,
        freeze_visual_backbone=eval_args.freeze_visual_backbone,
    )
    device = resolve_device(eval_args.device)
    model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    base_metrics = compute_metrics(
        model, loader, device, eval_args, epoch=int(checkpoint.get("epoch", 1)), train_mode=False
    )

    all_preds = {task: [] for task in TASKS}
    all_labels = {task: [] for task in TASKS}
    all_unc = {task: [] for task in TASKS}
    all_correct = {task: [] for task in TASKS}
    all_gate = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            tactile = batch["tactile"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            labels = {task: batch[task].to(device) for task in TASKS}
            out = _forward_batch(
                model=model,
                images=images,
                tactile=tactile,
                padding_mask=padding_mask,
                labels=None,
                args=eval_args,
                epoch=int(checkpoint.get("epoch", 1)),
                compute_nce=False,
                train_mode=False,
            )
            all_gate.extend(out["gate"].cpu().tolist())
            for task in TASKS:
                probs = out[f"prob_{task}"]
                preds = probs.argmax(dim=1)
                all_preds[task].extend(preds.cpu().tolist())
                all_labels[task].extend(labels[task].cpu().tolist())
                unc = out[f"unc_{task}"]
                all_unc[task].extend(unc.cpu().tolist())
                all_correct[task].extend((preds == labels[task]).cpu().tolist())

    def idx_to_name(mapping: Dict[str, int]) -> List[str]:
        return [name for name, idx in sorted(mapping.items(), key=lambda item: item[1])]

    label_names = {
        "mass": idx_to_name(dataset.mass_to_idx),
        "stiffness": idx_to_name(dataset.stiffness_to_idx),
        "material": idx_to_name(dataset.material_to_idx),
    }

    results: Dict[str, Dict] = {}
    for task in TASKS:
        preds = all_preds[task]
        labels = all_labels[task]
        names = label_names[task]
        all_class_labels = list(range(len(names)))
        acc = accuracy_score(labels, preds)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, preds, labels=all_class_labels, average=None, zero_division=0
        )
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, preds, labels=all_class_labels, average="macro", zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            labels, preds, labels=all_class_labels, average="weighted", zero_division=0
        )
        report_text = classification_report(
            labels, preds, labels=all_class_labels, target_names=names, digits=4, zero_division=0
        )
        results[task] = {
            "accuracy": float(acc),
            "macro": {"precision": float(precision_macro), "recall": float(recall_macro), "f1": float(f1_macro)},
            "weighted": {
                "precision": float(precision_weighted),
                "recall": float(recall_weighted),
                "f1": float(f1_weighted),
            },
            "per_class": {
                names[i]: {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i]),
                }
                for i in range(len(names))
            },
            "classification_report": report_text,
        }

    avg_acc = float(np.mean([results[t]["accuracy"] for t in TASKS]))
    avg_macro_f1 = float(np.mean([results[t]["macro"]["f1"] for t in TASKS]))
    avg_weighted_f1 = float(np.mean([results[t]["weighted"]["f1"] for t in TASKS]))

    if eval_args.output_dir:
        output_dir = Path(eval_args.output_dir)
    else:
        output_dir = ckpt_path.parent / f"eval_{split_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    for task in TASKS:
        _plot_confusion_matrix(all_labels[task], all_preds[task], label_names[task], task, output_dir)

    _plot_summary(results, output_dir)

    uncertainty_summary = {}
    for task in TASKS:
        unc = np.asarray(all_unc[task], dtype=np.float32)
        corr = np.asarray(all_correct[task], dtype=bool)
        mean_all = float(np.mean(unc)) if unc.size else 0.0
        mean_corr = float(np.mean(unc[corr])) if np.any(corr) else 0.0
        mean_inc = float(np.mean(unc[~corr])) if np.any(~corr) else 0.0
        uncertainty_summary[task] = {
            "mean": mean_all,
            "mean_correct": mean_corr,
            "mean_incorrect": mean_inc,
        }

    full_result = {
        "split": split_name,
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "num_samples": len(dataset),
        "loss": float(base_metrics["loss"]),
        "mass": float(base_metrics["mass"]),
        "stiffness": float(base_metrics["stiffness"]),
        "material": float(base_metrics["material"]),
        "summary": {
            "average_accuracy": avg_acc,
            "average_macro_f1": avg_macro_f1,
            "average_weighted_f1": avg_weighted_f1,
        },
        "uncertainty": uncertainty_summary,
        "gate": {"mean": float(np.mean(np.asarray(all_gate, dtype=np.float32))) if all_gate else 0.0},
        "tasks": results,
    }
    (output_dir / "evaluation_results.json").write_text(json.dumps(full_result, indent=2, ensure_ascii=False))
    return full_result


def train(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(f"Expected both train/ and val/ under {data_root}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    device = resolve_device(args.device)
    print(f"device: {device}")

    train_loader = build_loader(
        train_dir,
        args.batch_size,
        args.max_tactile_len,
        args.num_workers,
        shuffle=True,
        strict_data=args.strict_data,
        max_data_warnings=args.max_data_warnings,
    )
    val_loader = build_loader(
        val_dir,
        args.batch_size,
        args.max_tactile_len,
        args.num_workers,
        shuffle=False,
        strict_data=args.strict_data,
        max_data_warnings=args.max_data_warnings,
    )
    print(f"train samples: {len(train_loader.dataset)} | val samples: {len(val_loader.dataset)}")

    task_dims = {
        "mass": len(train_loader.dataset.mass_to_idx),
        "stiffness": len(train_loader.dataset.stiffness_to_idx),
        "material": len(train_loader.dataset.material_to_idx),
    }

    model = LPCEAModel(
        latent_dim=args.latent_dim,
        task_dims=task_dims,
        freeze_visual_backbone=args.freeze_visual_backbone,
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    def lr_lambda(epoch_idx: int) -> float:
        if epoch_idx < args.warmup_epochs:
            return (epoch_idx + 1) / max(1, args.warmup_epochs)
        progress = (epoch_idx - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history: List[Dict] = []
    best_val_acc = -1.0
    best_epoch = -1
    live_plotter = None
    if args.live_plot:
        try:
            live_plotter = LiveTrainingPlotter(save_dir)
            print("live plot: enabled")
        except Exception as exc:
            print(f"live plot disabled: {exc}")
            live_plotter = None

    for epoch in range(1, args.epochs + 1):
        train_metrics = compute_metrics(
            model=model,
            loader=train_loader,
            device=device,
            args=args,
            epoch=epoch,
            train_mode=True,
            optimizer=optimizer,
        )
        val_metrics = compute_metrics(
            model=model,
            loader=val_loader,
            device=device,
            args=args,
            epoch=epoch,
            train_mode=False,
        )
        scheduler.step()

        avg_val = (val_metrics["mass"] + val_metrics["stiffness"] + val_metrics["material"]) / 3.0
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics, "avg_val_acc": avg_val})
        if live_plotter is not None:
            live_plotter.update(epoch, train_metrics, val_metrics)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} | "
            f"val_mass={val_metrics['mass']:.2%} val_stiff={val_metrics['stiffness']:.2%} val_mat={val_metrics['material']:.2%} | "
            f"gate={val_metrics['gate']:.3f}"
        )

        if avg_val > best_val_acc:
            best_val_acc = avg_val
            best_epoch = epoch
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": vars(args),
                "val_metrics": val_metrics,
            }
            torch.save(ckpt, save_dir / "best_model.pth")

        if epoch % args.save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": vars(args),
                },
                save_dir / f"checkpoint_epoch_{epoch}.pth",
            )

    (save_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    try:
        _plot_training_curves(history, save_dir)
    except Exception as exc:
        print(f"plotting skipped: {exc}")
    if live_plotter is not None:
        live_plotter.save()
        live_plotter.close()

    print(f"best epoch: {best_epoch} | best val avg acc: {best_val_acc:.2%}")

    for split_name in ["test", "ood_test"]:
        split_dir = data_root / split_name
        if split_dir.is_dir():
            metrics = eval_split(args, split_name=split_name, checkpoint_path=save_dir / "best_model.pth")
            print(
                f"{split_name}: loss={metrics['loss']:.4f}, "
                f"mass={metrics['mass']:.2%}, stiffness={metrics['stiffness']:.2%}, material={metrics['material']:.2%}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LPC-EA training script (streaming + Soft-DTW gating + EDL)")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--data_root", type=str, default="/home/martina/Y3_Project/Plaintextdataset")
    parser.add_argument("--save_dir", type=str, default="outputs/lpc_ea")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--eval_split", choices=["val", "test", "ood_test"], default="test")
    parser.add_argument("--output_dir", type=str, default="")
    parser.set_defaults(live_plot=True)
    parser.add_argument("--live_plot", dest="live_plot", action="store_true", help="Enable live training plots")
    parser.add_argument("--no_live_plot", dest="live_plot", action="store_false", help="Disable live training plots")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_tactile_len", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--strict_data", action="store_true", help="Raise immediately on data load errors.")
    parser.add_argument(
        "--max_data_warnings",
        type=int,
        default=10,
        help="Max number of image-load warnings before suppressing further warnings (non-strict mode).",
    )

    # LPC-EA specifics.
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--window_size", type=int, default=128)
    parser.add_argument("--step_size", type=int, default=20)
    parser.add_argument("--rollout_k", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=1.0, help="gate coefficient in g=exp(-gamma*r)")
    parser.add_argument("--soft_dtw_gamma", type=float, default=0.1, help="Soft-DTW smoothing parameter")
    parser.add_argument("--lambda_nce", type=float, default=1.0)
    parser.add_argument("--beta_kl", type=float, default=1e-3)
    parser.add_argument("--kl_anneal_epochs", type=int, default=10)
    parser.add_argument("--nce_pairs_per_seq", type=int, default=1)
    parser.add_argument("--surprise_metric", choices=["soft_dtw", "l2"], default="soft_dtw")
    parser.add_argument(
        "--train_gate_t0_strategy",
        choices=["random", "first", "middle", "last"],
        default="random",
        help="How to pick t0 when computing gate during training.",
    )
    parser.add_argument(
        "--eval_gate_t0_strategy",
        choices=["random", "first", "middle", "last"],
        default="middle",
        help="How to pick t0 when computing gate during evaluation (deterministic by default).",
    )

    parser.set_defaults(freeze_visual_backbone=True)
    parser.add_argument("--freeze_visual_backbone", dest="freeze_visual_backbone", action="store_true")
    parser.add_argument("--unfreeze_visual_backbone", dest="freeze_visual_backbone", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train(args)
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required in eval mode")
        m = eval_split(args, split_name=args.eval_split, checkpoint_path=Path(args.checkpoint))
        print(
            f"{args.eval_split}: loss={m['loss']:.4f}, "
            f"mass={m['mass']:.2%}, stiffness={m['stiffness']:.2%}, material={m['material']:.2%}, "
            f"avg_acc={m['summary']['average_accuracy']:.2%}"
        )
