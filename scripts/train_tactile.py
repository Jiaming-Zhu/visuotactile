import argparse
import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


# Keep identical normalization constants with legacy script.
TACTILE_STATS = {
    "joint_position": {"mean": 21.70, "std": 38.13},
    "joint_load": {"mean": 7.21, "std": 14.03},
    "joint_current": {"mean": 52.56, "std": 133.43},
    "joint_velocity": {"mean": 0.13, "std": 9.79},
}


class TactileOnlyModel(nn.Module):
    def __init__(
        self,
        fusion_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_layers: int = 4,
        max_tactile_len: int = 3000,
        mass_classes: int = 4,
        stiffness_classes: int = 4,
        material_classes: int = 5,
    ) -> None:
        super().__init__()
        self.fusion_dim = fusion_dim

        self.tac_encoder = nn.Sequential(
            nn.Conv1d(24, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, fusion_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 24, max_tactile_len)
            tac_feat = self.tac_encoder(dummy)
            num_tac_tokens = tac_feat.shape[-1]

        self.cls_token = nn.Parameter(torch.randn(1, 1, fusion_dim))
        self.pos_emb = nn.Parameter(torch.randn(1, 1 + num_tac_tokens, fusion_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=num_heads,
            dim_feedforward=fusion_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head_mass = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, mass_classes),
        )
        self.head_stiffness = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, stiffness_classes),
        )
        self.head_material = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, material_classes),
        )

    def forward(self, tac: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        bsz = tac.shape[0]
        device = tac.device

        t = self.tac_encoder(tac)
        t_tokens = t.transpose(1, 2)
        num_tac_tokens = t_tokens.shape[1]

        cls_token = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat([cls_token, t_tokens], dim=1)

        if x.shape[1] > self.pos_emb.shape[1]:
            raise ValueError(
                f"Token length {x.shape[1]} exceeds position embedding length {self.pos_emb.shape[1]}"
            )
        x = x + self.pos_emb[:, : x.shape[1], :]

        full_mask = None
        if padding_mask is not None:
            tac_mask = padding_mask.float().unsqueeze(1)
            tac_mask = torch.nn.functional.max_pool1d(tac_mask, kernel_size=2, stride=2)
            tac_mask = torch.nn.functional.max_pool1d(tac_mask, kernel_size=2, stride=2)
            tac_mask = torch.nn.functional.max_pool1d(tac_mask, kernel_size=2, stride=2)
            tac_mask = tac_mask.squeeze(1) > 0.5
            tac_mask = tac_mask[:, :num_tac_tokens]

            cls_mask = torch.zeros(bsz, 1, dtype=torch.bool, device=device)
            full_mask = torch.cat([cls_mask, tac_mask], dim=1)

        x = self.transformer_encoder(x, src_key_padding_mask=full_mask)
        cls_out = x[:, 0, :]
        return {
            "mass": self.head_mass(cls_out),
            "stiffness": self.head_stiffness(cls_out),
            "material": self.head_material(cls_out),
        }


@dataclass
class Sample:
    tactile_path: Path
    labels: Dict[str, int]


class RoboticGraspDataset(Dataset):
    def __init__(
        self,
        split_dir: Path,
        max_tactile_len: int = 3000,
        tactile_stats: Optional[Dict[str, Dict[str, float]]] = None,
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
                tactile_path = episode_dir / "tactile_data.pkl"
                if tactile_path.exists():
                    samples.append(Sample(tactile_path=tactile_path, labels=labels))

        if not samples:
            raise RuntimeError(f"No valid samples found in {self.split_dir}")
        return samples

    def _normalize(self, arr: np.ndarray, key: str) -> np.ndarray:
        mean = self.tactile_stats[key]["mean"]
        std = self.tactile_stats[key]["std"]
        return (np.asarray(arr) - mean) / (std + 1e-8)

    def _load_tactile(self, tactile_path: Path) -> torch.Tensor:
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
        tactile, padding_mask = self._load_tactile(sample.tactile_path)
        return {
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    return device


def apply_tactile_dropout(tactile: torch.Tensor, tactile_drop_prob: float = 0.0) -> torch.Tensor:
    if tactile_drop_prob <= 0:
        return tactile
    bsz = tactile.size(0)
    device = tactile.device
    tac_mask = (torch.rand(bsz, device=device) < tactile_drop_prob).view(-1, 1, 1).float()
    return tactile * (1 - tac_mask)


def compute_metrics(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    train_mode: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None,
    tactile_drop_prob: float = 0.0,
) -> Dict[str, float]:
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0
    correct = {"mass": 0, "stiffness": 0, "material": 0}

    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, leave=False, desc="train" if train_mode else "eval")

    for batch_idx, batch in enumerate(iterator, start=1):
        tactile = batch["tactile"].to(device, non_blocking=True)
        padding_mask = batch["padding_mask"].to(device, non_blocking=True)
        labels = {k: batch[k].to(device, non_blocking=True) for k in ["mass", "stiffness", "material"]}

        if train_mode:
            tactile = apply_tactile_dropout(tactile, tactile_drop_prob)
            optimizer.zero_grad()
            outputs = model(tactile, padding_mask=padding_mask)
            loss = (
                criterion(outputs["mass"], labels["mass"])
                + criterion(outputs["stiffness"], labels["stiffness"])
                + criterion(outputs["material"], labels["material"])
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        else:
            with torch.no_grad():
                outputs = model(tactile, padding_mask=padding_mask)
                loss = (
                    criterion(outputs["mass"], labels["mass"])
                    + criterion(outputs["stiffness"], labels["stiffness"])
                    + criterion(outputs["material"], labels["material"])
                )

        bsz = tactile.size(0)
        total_loss += loss.item() * bsz
        total_samples += bsz
        for task in correct:
            correct[task] += (outputs[task].argmax(dim=1) == labels[task]).sum().item()

        if tqdm is not None and hasattr(iterator, "set_postfix"):
            running_loss = total_loss / max(1, total_samples)
            running_mass = correct["mass"] / max(1, total_samples)
            running_stiffness = correct["stiffness"] / max(1, total_samples)
            running_material = correct["material"] / max(1, total_samples)
            iterator.set_postfix(
                {
                    "loss": f"{running_loss:.4f}",
                    "mass": f"{running_mass:.2%}",
                    "stiff": f"{running_stiffness:.2%}",
                    "mat": f"{running_material:.2%}",
                    "step": batch_idx,
                }
            )

    return {
        "loss": total_loss / total_samples,
        "mass": correct["mass"] / total_samples,
        "stiffness": correct["stiffness"] / total_samples,
        "material": correct["material"] / total_samples,
    }


def build_loader(split_dir: Path, batch_size: int, max_tactile_len: int, num_workers: int, shuffle: bool) -> DataLoader:
    dataset = RoboticGraspDataset(split_dir=split_dir, max_tactile_len=max_tactile_len)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": True,
        "drop_last": shuffle,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    return DataLoader(
        dataset,
        **loader_kwargs,
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
    fig.suptitle("Tactile-Only Model Evaluation Summary", fontsize=14, fontweight="bold")

    tasks = ["mass", "stiffness", "material"]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]

    accs = [results[t]["accuracy"] for t in tasks]
    axes[0].bar(tasks, accs, color=colors)
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
            c=colors[i],
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
        self.fig.suptitle("Tactile-Only Training (Live)", fontsize=14, fontweight="bold")

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


def _load_checkpoint(path: Path, device: torch.device) -> Dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _idx_to_names(mapping: Dict[str, int]) -> List[str]:
    return [name for name, _ in sorted(mapping.items(), key=lambda kv: kv[1])]


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
    print(
        f"early stopping: patience={args.early_stop_patience}, "
        f"min_epoch={args.early_stop_min_epoch}, target_acc={args.early_stop_acc}"
    )

    train_loader = build_loader(train_dir, args.batch_size, args.max_tactile_len, args.num_workers, shuffle=True)
    val_loader = build_loader(val_dir, args.batch_size, args.max_tactile_len, args.num_workers, shuffle=False)
    print(f"train samples: {len(train_loader.dataset)} | val samples: {len(val_loader.dataset)}")

    mass_classes = len(train_loader.dataset.mass_to_idx)
    stiffness_classes = len(train_loader.dataset.stiffness_to_idx)
    material_classes = len(train_loader.dataset.material_to_idx)
    if mass_classes != len(val_loader.dataset.mass_to_idx):
        raise ValueError("train/ and val/ mass class counts do not match")
    if stiffness_classes != len(val_loader.dataset.stiffness_to_idx):
        raise ValueError("train/ and val/ stiffness class counts do not match")
    if material_classes != len(val_loader.dataset.material_to_idx):
        raise ValueError("train/ and val/ material class counts do not match")

    args.mass_classes = mass_classes
    args.stiffness_classes = stiffness_classes
    args.material_classes = material_classes
    print(
        "class counts: "
        f"mass={mass_classes}, stiffness={stiffness_classes}, material={material_classes}"
    )

    model = TactileOnlyModel(
        fusion_dim=args.fusion_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        num_layers=args.num_layers,
        max_tactile_len=args.max_tactile_len,
        mass_classes=mass_classes,
        stiffness_classes=stiffness_classes,
        material_classes=material_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
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

    history = []
    best_val_acc = -1.0
    best_epoch = -1
    early_stop_streak = 0
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
            criterion=criterion,
            device=device,
            train_mode=True,
            optimizer=optimizer,
            tactile_drop_prob=args.tactile_drop_prob,
        )
        val_metrics = compute_metrics(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
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
            f"val_mass={val_metrics['mass']:.2%} val_stiff={val_metrics['stiffness']:.2%} "
            f"val_mat={val_metrics['material']:.2%}"
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

        if args.early_stop_patience > 0:
            if avg_val >= (args.early_stop_acc - 1e-12):
                early_stop_streak += 1
            else:
                early_stop_streak = 0
            if epoch >= args.early_stop_min_epoch and early_stop_streak >= args.early_stop_patience:
                print(
                    f"early stop: avg_val_acc={avg_val:.2%} for {early_stop_streak} epochs "
                    f"(min_epoch={args.early_stop_min_epoch}, patience={args.early_stop_patience})"
                )
                break

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
    _plot_training_curves(history, save_dir)
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
                f"mass={metrics['mass']:.2%}, stiffness={metrics['stiffness']:.2%}, "
                f"material={metrics['material']:.2%}"
            )


def eval_split(args: argparse.Namespace, split_name: str, checkpoint_path: Optional[Path] = None) -> Dict:
    try:
        from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
    except ImportError as exc:
        raise ImportError("eval mode requires scikit-learn") from exc

    data_root = Path(args.data_root)
    split_dir = data_root / split_name
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split not found: {split_dir}")

    ckpt_path = Path(checkpoint_path) if checkpoint_path else Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = resolve_device(args.device)
    loader = build_loader(split_dir, args.batch_size, args.max_tactile_len, args.num_workers, shuffle=False)
    dataset = loader.dataset

    checkpoint = _load_checkpoint(ckpt_path, device)
    cfg = checkpoint.get("config", {})
    ckpt_state = checkpoint["model_state_dict"]

    dataset_mass_classes = len(dataset.mass_to_idx)
    dataset_stiffness_classes = len(dataset.stiffness_to_idx)
    dataset_material_classes = len(dataset.material_to_idx)
    mass_classes = int(cfg.get("mass_classes", ckpt_state["head_mass.3.weight"].shape[0]))
    stiffness_classes = int(cfg.get("stiffness_classes", ckpt_state["head_stiffness.3.weight"].shape[0]))
    material_classes = int(cfg.get("material_classes", ckpt_state["head_material.3.weight"].shape[0]))

    if mass_classes != dataset_mass_classes:
        raise ValueError(
            f"Checkpoint expects {mass_classes} mass classes, but dataset defines {dataset_mass_classes}. "
            "Retrain the model after changing mass labels."
        )
    if stiffness_classes != dataset_stiffness_classes:
        raise ValueError(
            f"Checkpoint expects {stiffness_classes} stiffness classes, but dataset defines {dataset_stiffness_classes}."
        )
    if material_classes != dataset_material_classes:
        raise ValueError(
            f"Checkpoint expects {material_classes} material classes, but dataset defines {dataset_material_classes}."
        )

    model = TactileOnlyModel(
        fusion_dim=cfg.get("fusion_dim", args.fusion_dim),
        num_heads=cfg.get("num_heads", args.num_heads),
        dropout=cfg.get("dropout", args.dropout),
        num_layers=cfg.get("num_layers", args.num_layers),
        max_tactile_len=cfg.get("max_tactile_len", args.max_tactile_len),
        mass_classes=mass_classes,
        stiffness_classes=stiffness_classes,
        material_classes=material_classes,
    ).to(device)
    model.load_state_dict(ckpt_state)

    criterion = nn.CrossEntropyLoss()
    base_metrics = compute_metrics(model=model, loader=loader, criterion=criterion, device=device, train_mode=False)

    all_preds = {"mass": [], "stiffness": [], "material": []}
    all_labels = {"mass": [], "stiffness": [], "material": []}

    model.eval()
    with torch.no_grad():
        for batch in loader:
            tactile = batch["tactile"].to(device, non_blocking=True)
            padding_mask = batch["padding_mask"].to(device, non_blocking=True)
            outputs = model(tactile, padding_mask=padding_mask)
            for task in all_preds:
                all_preds[task].extend(outputs[task].argmax(dim=1).cpu().tolist())
                all_labels[task].extend(batch[task].tolist())

    label_names = {
        "mass": _idx_to_names(dataset.mass_to_idx),
        "stiffness": _idx_to_names(dataset.stiffness_to_idx),
        "material": _idx_to_names(dataset.material_to_idx),
    }

    results = {}
    for task in ["mass", "stiffness", "material"]:
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
            "macro": {
                "precision": float(precision_macro),
                "recall": float(recall_macro),
                "f1": float(f1_macro),
            },
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

    avg_acc = float(np.mean([results[t]["accuracy"] for t in ["mass", "stiffness", "material"]]))
    avg_macro_f1 = float(np.mean([results[t]["macro"]["f1"] for t in ["mass", "stiffness", "material"]]))
    avg_weighted_f1 = float(np.mean([results[t]["weighted"]["f1"] for t in ["mass", "stiffness", "material"]]))

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = ckpt_path.parent / f"eval_{split_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    for task in ["mass", "stiffness", "material"]:
        _plot_confusion_matrix(all_labels[task], all_preds[task], label_names[task], task, output_dir)

    _plot_summary(results, output_dir)

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
        "tasks": results,
    }
    (output_dir / "evaluation_results.json").write_text(json.dumps(full_result, indent=2, ensure_ascii=False))
    return full_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training script for tactile-only baseline")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--data_root", type=str, default="/home/martina/Y3_Project/Plaintextdataset")
    parser.add_argument("--save_dir", type=str, default="outputs/tactile_model_clean")
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

    parser.add_argument("--fusion_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--tactile_drop_prob", type=float, default=0.0)
    parser.add_argument("--early_stop_patience", type=int, default=0, help="Disable if 0; stop if val avg acc reaches threshold for N epochs")
    parser.add_argument("--early_stop_acc", type=float, default=1.0, help="Validation avg acc threshold for early stop")
    parser.add_argument("--early_stop_min_epoch", type=int, default=0, help="Do not early stop before this epoch")
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
