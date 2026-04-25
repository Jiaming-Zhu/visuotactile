import argparse
import json
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


TASKS = ("mass", "stiffness", "material")

TACTILE_STATS = {
    "joint_position": {"mean": 21.70, "std": 38.13},
    "joint_load": {"mean": 7.21, "std": 14.03},
    "joint_current": {"mean": 52.56, "std": 133.43},
    "joint_velocity": {"mean": 0.13, "std": 9.79},
}


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07, eps: float = 1e-12) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.eps = float(eps)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or labels.ndim != 1 or features.size(0) != labels.size(0):
            raise ValueError("invalid shapes for supervised contrastive loss")
        if features.size(0) < 2:
            return features.new_zeros(())

        features = torch.nn.functional.normalize(features, dim=-1)
        logits = torch.matmul(features, features.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        batch_size = labels.size(0)
        same_label = labels.unsqueeze(0) == labels.unsqueeze(1)
        identity = torch.eye(batch_size, device=labels.device, dtype=torch.bool)
        positive_mask = same_label & ~identity
        valid_anchor = positive_mask.any(dim=1)
        if not valid_anchor.any():
            return features.new_zeros(())

        logits_mask = ~identity
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + self.eps)
        positive_log_prob = (positive_mask * log_prob).sum(dim=1) / positive_mask.sum(dim=1).clamp(min=1)
        return -positive_log_prob[valid_anchor].mean()


def parse_prefix_ratios(raw: str) -> List[float]:
    ratios: List[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value <= 0.0 or value > 1.0:
            raise ValueError(f"prefix ratio must be in (0, 1], got {value}")
        ratios.append(value)
    if not ratios:
        raise ValueError("at least one prefix ratio is required")
    return sorted(set(ratios))


def compute_valid_lengths(padding_mask: torch.Tensor) -> torch.Tensor:
    return (~padding_mask).sum(dim=1).long().clamp(min=1)


def build_prefix_padding_mask(
    padding_mask: torch.Tensor,
    prefix_lengths: torch.Tensor,
) -> torch.Tensor:
    seq_len = padding_mask.size(1)
    positions = torch.arange(seq_len, device=padding_mask.device).unsqueeze(0)
    prefix_valid = positions < prefix_lengths.unsqueeze(1)
    original_valid = ~padding_mask
    return ~(original_valid & prefix_valid)


def fixed_prefix_lengths(
    padding_mask: torch.Tensor,
    prefix_ratio: float,
    min_prefix_len: int,
) -> torch.Tensor:
    valid_lengths = compute_valid_lengths(padding_mask)
    ratio_lengths = torch.ceil(valid_lengths.float() * prefix_ratio).long()
    return torch.minimum(valid_lengths, torch.clamp(ratio_lengths, min=min_prefix_len))


def sample_prefix_lengths(
    padding_mask: torch.Tensor,
    min_prefix_ratio: float,
    min_prefix_len: int,
) -> torch.Tensor:
    valid_lengths = compute_valid_lengths(padding_mask)
    lower = torch.ceil(valid_lengths.float() * min_prefix_ratio).long()
    lower = torch.clamp(lower, min=min_prefix_len)
    lower = torch.minimum(lower, valid_lengths)
    span = (valid_lengths - lower + 1).clamp(min=1)
    rand = torch.rand(valid_lengths.size(0), device=padding_mask.device)
    sampled = lower + torch.floor(rand * span.float()).long()
    return torch.minimum(sampled, valid_lengths).clamp(min=1)


def effective_padding_mask(
    padding_mask: torch.Tensor,
    train_mode: bool,
    online_train_prob: float,
    online_min_prefix_ratio: float,
    min_prefix_len: int,
    fixed_ratio: Optional[float] = None,
) -> torch.Tensor:
    if fixed_ratio is not None:
        prefix_lengths = fixed_prefix_lengths(padding_mask, fixed_ratio, min_prefix_len)
        return build_prefix_padding_mask(padding_mask, prefix_lengths)

    if not train_mode or online_train_prob <= 0:
        return padding_mask

    if online_train_prob < 1.0:
        use_prefix = torch.rand(padding_mask.size(0), device=padding_mask.device) < online_train_prob
        if not use_prefix.any():
            return padding_mask
        sampled_lengths = sample_prefix_lengths(padding_mask, online_min_prefix_ratio, min_prefix_len)
        full_lengths = compute_valid_lengths(padding_mask)
        prefix_lengths = torch.where(use_prefix, sampled_lengths, full_lengths)
    else:
        prefix_lengths = sample_prefix_lengths(padding_mask, online_min_prefix_ratio, min_prefix_len)

    return build_prefix_padding_mask(padding_mask, prefix_lengths)


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
    device = torch.device(device_arg if requested else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    return device


def apply_modality_dropout(
    images: torch.Tensor,
    tactile: torch.Tensor,
    visual_drop_prob: float = 0.0,
    tactile_drop_prob: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz = images.size(0)
    device = images.device
    if visual_drop_prob > 0:
        vis_mask = (torch.rand(bsz, device=device) < visual_drop_prob).view(-1, 1, 1, 1).float()
        images = images * (1.0 - vis_mask)
    if tactile_drop_prob > 0:
        tac_mask = (torch.rand(bsz, device=device) < tactile_drop_prob).view(-1, 1, 1).float()
        tactile = tactile * (1.0 - tac_mask)
    return images, tactile


def apply_modality_block(
    images: torch.Tensor,
    tactile: torch.Tensor,
    block_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if block_mode == "visual":
        return torch.zeros_like(images), tactile
    if block_mode == "tactile":
        return images, torch.zeros_like(tactile)
    return images, tactile


def build_train_transform(augment_policy: str) -> transforms.Compose:
    if augment_policy == "classical":
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.03),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.30),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


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
        train_mode: bool = False,
        augment_policy: str = "none",
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
        self.transform = build_train_transform(augment_policy) if train_mode else build_eval_transform()
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
            if not obj_class_dir.is_dir() or obj_class_dir.name.startswith("analysis"):
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

    def _load_tactile(self, tactile_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
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
        image = Image.open(sample.img_path).convert("RGB")
        image_tensor = self.transform(image)
        tactile_tensor, padding_mask = self._load_tactile(sample.tactile_path)
        return {
            "image": image_tensor,
            "tactile": tactile_tensor,
            "padding_mask": padding_mask,
            "mass": torch.tensor(sample.labels["mass"], dtype=torch.long),
            "stiffness": torch.tensor(sample.labels["stiffness"], dtype=torch.long),
            "material": torch.tensor(sample.labels["material"], dtype=torch.long),
        }


def build_loader(
    split_dir: Path,
    batch_size: int,
    max_tactile_len: int,
    num_workers: int,
    shuffle: bool,
    augment_policy: str = "none",
) -> DataLoader:
    dataset = RoboticGraspDataset(
        split_dir=split_dir,
        max_tactile_len=max_tactile_len,
        train_mode=shuffle,
        augment_policy=augment_policy if shuffle else "none",
    )
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
    return DataLoader(dataset, **loader_kwargs)


class FusionModel(nn.Module):
    def __init__(
        self,
        fusion_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_layers: int = 4,
        freeze_visual: bool = True,
        mass_classes: int = 4,
        stiffness_classes: int = 4,
        material_classes: int = 5,
        enable_tactile_aux: bool = False,
        enable_supcon: bool = False,
    ) -> None:
        super().__init__()
        self.enable_tactile_aux = bool(enable_tactile_aux)
        self.enable_supcon = bool(enable_supcon)

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.vis_backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.vis_proj = nn.Conv2d(512, fusion_dim, kernel_size=1)
        if freeze_visual:
            for param in self.vis_backbone.parameters():
                param.requires_grad = False

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

        self.cls_token = nn.Parameter(torch.randn(1, 1, fusion_dim))
        self.pos_emb = nn.Parameter(torch.randn(1, 425, fusion_dim))
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

        if self.enable_tactile_aux:
            self.aux_head_mass = nn.Sequential(nn.Linear(fusion_dim, 64), nn.GELU(), nn.Linear(64, mass_classes))
            self.aux_head_stiffness = nn.Sequential(
                nn.Linear(fusion_dim, 64), nn.GELU(), nn.Linear(64, stiffness_classes)
            )
            self.aux_head_material = nn.Sequential(
                nn.Linear(fusion_dim, 64), nn.GELU(), nn.Linear(64, material_classes)
            )
        if self.enable_supcon:
            self.tactile_projection_head = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.GELU(),
                nn.Linear(fusion_dim, 128),
            )

    def _masked_tactile_global(self, t_tokens: torch.Tensor, padding_mask: Optional[torch.Tensor]) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        num_tac_tokens = t_tokens.shape[1]
        tac_mask = None
        if padding_mask is not None:
            tac_mask = padding_mask.float().unsqueeze(1)
            tac_mask = torch.nn.functional.max_pool1d(tac_mask, kernel_size=2, stride=2)
            tac_mask = torch.nn.functional.max_pool1d(tac_mask, kernel_size=2, stride=2)
            tac_mask = torch.nn.functional.max_pool1d(tac_mask, kernel_size=2, stride=2)
            tac_mask = tac_mask.squeeze(1) > 0.5
            tac_mask = tac_mask[:, :num_tac_tokens]
            tac_mask_float = (~tac_mask).unsqueeze(-1).float()
            t_global = (t_tokens * tac_mask_float).sum(dim=1) / (tac_mask_float.sum(dim=1) + 1e-8)
            return t_global, tac_mask
        return t_tokens.mean(dim=1), tac_mask

    def forward(
        self,
        img: torch.Tensor,
        tac: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        bsz = img.shape[0]
        device = img.device

        v = self.vis_backbone(img)
        v = self.vis_proj(v)
        v_tokens = v.flatten(2).transpose(1, 2)
        num_vis_tokens = v_tokens.shape[1]

        t = self.tac_encoder(tac)
        t_tokens = t.transpose(1, 2)
        t_global, tac_mask = self._masked_tactile_global(t_tokens, padding_mask=padding_mask)

        cls_token = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat([cls_token, v_tokens, t_tokens], dim=1)
        seq_len = x.shape[1]
        x = x + self.pos_emb[:, :seq_len, :]

        full_mask = None
        if tac_mask is not None:
            cls_vis_mask = torch.zeros(bsz, 1 + num_vis_tokens, dtype=torch.bool, device=device)
            full_mask = torch.cat([cls_vis_mask, tac_mask], dim=1)

        x = self.transformer_encoder(x, src_key_padding_mask=full_mask)
        cls_out = x[:, 0, :]
        outputs = {
            "mass": self.head_mass(cls_out),
            "stiffness": self.head_stiffness(cls_out),
            "material": self.head_material(cls_out),
        }
        if self.enable_tactile_aux:
            outputs["aux_mass"] = self.aux_head_mass(t_global)
            outputs["aux_stiffness"] = self.aux_head_stiffness(t_global)
            outputs["aux_material"] = self.aux_head_material(t_global)
        if self.enable_supcon:
            outputs["contrastive_embedding"] = torch.nn.functional.normalize(
                self.tactile_projection_head(t_global),
                dim=-1,
            )
        return outputs


def compute_metrics(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    train_mode: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None,
    fixed_prefix_ratio: Optional[float] = None,
) -> Dict[str, float]:
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_aux = 0.0
    total_supcon = 0.0
    total_samples = 0
    correct = {task: 0 for task in TASKS}
    supcon_criterion = None
    if args.lambda_supcon > 0.0:
        supcon_criterion = SupervisedContrastiveLoss(temperature=args.supcon_temperature)

    iterator = loader
    if tqdm is not None:
        desc = "train" if train_mode else ("prefix_eval" if fixed_prefix_ratio is not None else "eval")
        iterator = tqdm(loader, leave=False, desc=desc)

    for batch_idx, batch in enumerate(iterator, start=1):
        images = batch["image"].to(device, non_blocking=True)
        tactile = batch["tactile"].to(device, non_blocking=True)
        padding_mask = batch["padding_mask"].to(device, non_blocking=True)
        labels = {task: batch[task].to(device, non_blocking=True) for task in TASKS}

        prefix_mask = effective_padding_mask(
            padding_mask=padding_mask,
            train_mode=train_mode,
            online_train_prob=args.online_train_prob,
            online_min_prefix_ratio=args.online_min_prefix_ratio,
            min_prefix_len=args.min_prefix_len,
            fixed_ratio=fixed_prefix_ratio,
        )

        images, tactile = apply_modality_block(images, tactile, args.block_modality)

        if train_mode:
            images, tactile = apply_modality_dropout(
                images,
                tactile,
                visual_drop_prob=args.visual_drop_prob,
                tactile_drop_prob=args.tactile_drop_prob,
            )
            if optimizer is None:
                raise ValueError("optimizer is required for train_mode=True")
            optimizer.zero_grad()
            outputs = model(images, tactile, padding_mask=prefix_mask)
            ce_loss = (
                criterion(outputs["mass"], labels["mass"])
                + criterion(outputs["stiffness"], labels["stiffness"])
                + criterion(outputs["material"], labels["material"])
            )
            aux_loss = ce_loss.new_zeros(())
            if args.lambda_aux > 0.0:
                aux_loss = (
                    criterion(outputs["aux_mass"], labels["mass"])
                    + criterion(outputs["aux_stiffness"], labels["stiffness"])
                    + criterion(outputs["aux_material"], labels["material"])
                )
            supcon_loss = ce_loss.new_zeros(())
            if supcon_criterion is not None:
                supcon_loss = supcon_criterion(outputs["contrastive_embedding"], labels[args.supcon_task])
            loss = ce_loss + args.lambda_aux * aux_loss + args.lambda_supcon * supcon_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        else:
            with torch.no_grad():
                outputs = model(images, tactile, padding_mask=prefix_mask)
                ce_loss = (
                    criterion(outputs["mass"], labels["mass"])
                    + criterion(outputs["stiffness"], labels["stiffness"])
                    + criterion(outputs["material"], labels["material"])
                )
                aux_loss = ce_loss.new_zeros(())
                if args.lambda_aux > 0.0:
                    aux_loss = (
                        criterion(outputs["aux_mass"], labels["mass"])
                        + criterion(outputs["aux_stiffness"], labels["stiffness"])
                        + criterion(outputs["aux_material"], labels["material"])
                    )
                supcon_loss = ce_loss.new_zeros(())
                if supcon_criterion is not None:
                    supcon_loss = supcon_criterion(outputs["contrastive_embedding"], labels[args.supcon_task])
                loss = ce_loss + args.lambda_aux * aux_loss + args.lambda_supcon * supcon_loss

        bsz = images.size(0)
        total_loss += loss.item() * bsz
        total_ce += ce_loss.item() * bsz
        total_aux += aux_loss.item() * bsz
        total_supcon += supcon_loss.item() * bsz
        total_samples += bsz
        for task in TASKS:
            correct[task] += int((outputs[task].argmax(dim=1) == labels[task]).sum().item())

        if tqdm is not None and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                {
                    "loss": f"{total_loss / max(1, total_samples):.4f}",
                    "mass": f"{correct['mass'] / max(1, total_samples):.2%}",
                    "stiff": f"{correct['stiffness'] / max(1, total_samples):.2%}",
                    "mat": f"{correct['material'] / max(1, total_samples):.2%}",
                    "step": batch_idx,
                }
            )

    result = {
        "loss": total_loss / max(1, total_samples),
        "ce_loss": total_ce / max(1, total_samples),
        "aux_loss": total_aux / max(1, total_samples),
        "supcon_loss": total_supcon / max(1, total_samples),
        "mass": correct["mass"] / max(1, total_samples),
        "stiffness": correct["stiffness"] / max(1, total_samples),
        "material": correct["material"] / max(1, total_samples),
    }
    result["average_accuracy"] = float(np.mean([result["mass"], result["stiffness"], result["material"]]))
    return result


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    args: argparse.Namespace,
    val_metrics: Dict[str, float],
) -> None:
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": vars(args),
        "val_metrics": val_metrics,
    }
    torch.save(ckpt, path)


def plot_training_curves(history: List[Dict[str, object]], save_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    epochs = [point["epoch"] for point in history]
    train_loss = [point["train"]["loss"] for point in history]
    val_loss = [point["val"]["loss"] for point in history]
    val_avg = [point["avg_val_acc"] for point in history]
    val_mass = [point["val"]["mass"] for point in history]
    val_stiff = [point["val"]["stiffness"] for point in history]
    val_mat = [point["val"]["material"] for point in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, train_loss, label="Train")
    axes[0].plot(epochs, val_loss, label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, val_mass, label="Mass")
    axes[1].plot(epochs, val_stiff, label="Stiffness")
    axes[1].plot(epochs, val_mat, label="Material")
    axes[1].plot(epochs, val_avg, label="Average", linestyle="--")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(save_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_model_from_config(cfg: Dict[str, object], dataset: RoboticGraspDataset) -> FusionModel:
    return FusionModel(
        fusion_dim=int(cfg.get("fusion_dim", 256)),
        num_heads=int(cfg.get("num_heads", 8)),
        dropout=float(cfg.get("dropout", 0.1)),
        num_layers=int(cfg.get("num_layers", 4)),
        freeze_visual=True,
        mass_classes=len(dataset.mass_to_idx),
        stiffness_classes=len(dataset.stiffness_to_idx),
        material_classes=len(dataset.material_to_idx),
        enable_tactile_aux=float(cfg.get("lambda_aux", 0.0)) > 0.0,
        enable_supcon=float(cfg.get("lambda_supcon", 0.0)) > 0.0,
    )


def build_loader_for_eval(args: argparse.Namespace, split_dir: Path) -> DataLoader:
    return build_loader(
        split_dir=split_dir,
        batch_size=args.batch_size,
        max_tactile_len=args.max_tactile_len,
        num_workers=args.num_workers,
        shuffle=False,
        augment_policy="none",
    )


def eval_split(args: argparse.Namespace, split_name: str, checkpoint_path: Optional[Path] = None) -> Dict[str, object]:
    data_root = Path(args.data_root)
    split_dir = data_root / split_name
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split not found: {split_dir}")
    ckpt_path = Path(checkpoint_path) if checkpoint_path else Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = resolve_device(args.device)
    loader = build_loader_for_eval(args, split_dir)
    dataset = loader.dataset
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})
    model = build_model_from_config(cfg, dataset).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()
    base_metrics = compute_metrics(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        args=args,
        train_mode=False,
        fixed_prefix_ratio=None,
    )
    result = {
        "split": split_name,
        "checkpoint": str(ckpt_path),
        "num_samples": len(dataset),
        "summary": {"average_accuracy": base_metrics["average_accuracy"]},
        "tasks": {
            task: {"accuracy": float(base_metrics[task])}
            for task in TASKS
        },
        "loss": float(base_metrics["loss"]),
        "ce_loss": float(base_metrics["ce_loss"]),
        "aux_loss": float(base_metrics["aux_loss"]),
        "supcon_loss": float(base_metrics["supcon_loss"]),
    }

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = ckpt_path.parent / f"eval_{split_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def online_eval_split(
    args: argparse.Namespace,
    split_name: str,
    checkpoint_path: Optional[Path] = None,
) -> Dict[str, object]:
    data_root = Path(args.data_root)
    split_dir = data_root / split_name
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split not found: {split_dir}")
    ckpt_path = Path(checkpoint_path) if checkpoint_path else Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = resolve_device(args.device)
    loader = build_loader_for_eval(args, split_dir)
    dataset = loader.dataset
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})
    model = build_model_from_config(cfg, dataset).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss()

    prefix_results = []
    for prefix_ratio in parse_prefix_ratios(args.online_eval_prefixes):
        metrics = compute_metrics(
            model=model,
            loader=loader,
            criterion=criterion,
            device=device,
            args=args,
            train_mode=False,
            fixed_prefix_ratio=prefix_ratio,
        )
        prefix_results.append(
            {
                "prefix_ratio": prefix_ratio,
                "summary": {"average_accuracy": metrics["average_accuracy"]},
                "tasks": {task: {"accuracy": float(metrics[task])} for task in TASKS},
                "loss": float(metrics["loss"]),
            }
        )

    result = {
        "split": split_name,
        "checkpoint": str(ckpt_path),
        "num_samples": len(dataset),
        "prefix_results": prefix_results,
    }
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = ckpt_path.parent / f"online_eval_{split_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "online_evaluation_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def train(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError("train/val splits are required")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = resolve_device(args.device)

    train_loader = build_loader(
        split_dir=train_dir,
        batch_size=args.batch_size,
        max_tactile_len=args.max_tactile_len,
        num_workers=args.num_workers,
        shuffle=True,
        augment_policy=args.augment_policy,
    )
    val_loader = build_loader(
        split_dir=val_dir,
        batch_size=args.batch_size,
        max_tactile_len=args.max_tactile_len,
        num_workers=args.num_workers,
        shuffle=False,
        augment_policy="none",
    )

    mass_classes = len(train_loader.dataset.mass_to_idx)
    stiffness_classes = len(train_loader.dataset.stiffness_to_idx)
    material_classes = len(train_loader.dataset.material_to_idx)
    args.mass_classes = mass_classes
    args.stiffness_classes = stiffness_classes
    args.material_classes = material_classes

    model = FusionModel(
        fusion_dim=args.fusion_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        num_layers=args.num_layers,
        freeze_visual=args.freeze_visual,
        mass_classes=mass_classes,
        stiffness_classes=stiffness_classes,
        material_classes=material_classes,
        enable_tactile_aux=args.lambda_aux > 0.0,
        enable_supcon=args.lambda_supcon > 0.0,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(epoch_idx: int) -> float:
        if epoch_idx < args.warmup_epochs:
            return (epoch_idx + 1) / max(1, args.warmup_epochs)
        progress = (epoch_idx - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()
    history: List[Dict[str, object]] = []
    best_val_acc = -1.0
    best_epoch = -1

    print(
        f"variant={args.variant_name} | online_train_prob={args.online_train_prob} | "
        f"lambda_aux={args.lambda_aux} | lambda_supcon={args.lambda_supcon} | augment={args.augment_policy}"
    )

    for epoch in range(1, args.epochs + 1):
        train_metrics = compute_metrics(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            args=args,
            train_mode=True,
            optimizer=optimizer,
            fixed_prefix_ratio=None,
        )
        val_metrics = compute_metrics(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            args=args,
            train_mode=False,
            fixed_prefix_ratio=None,
        )
        scheduler.step()

        avg_val = float(np.mean([val_metrics["mass"], val_metrics["stiffness"], val_metrics["material"]]))
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics, "avg_val_acc": avg_val})

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} | "
            f"val_mass={val_metrics['mass']:.2%} val_stiff={val_metrics['stiffness']:.2%} "
            f"val_mat={val_metrics['material']:.2%} val_avg={avg_val:.2%}"
        )

        if avg_val > best_val_acc:
            best_val_acc = avg_val
            best_epoch = epoch
            save_checkpoint(
                path=save_dir / "best_model.pth",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                val_metrics=val_metrics,
            )

        if epoch % args.save_every == 0:
            save_checkpoint(
                path=save_dir / f"checkpoint_epoch_{epoch}.pth",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                val_metrics=val_metrics,
            )

    (save_dir / "training_history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False))
    plot_training_curves(history, save_dir)
    (save_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False))
    print(f"best epoch: {best_epoch} | best val avg acc: {best_val_acc:.2%}")

    for split_name in ("test", "ood_test"):
        split_dir = data_root / split_name
        if split_dir.is_dir():
            metrics = eval_split(args, split_name=split_name, checkpoint_path=save_dir / "best_model.pth")
            print(
                f"{split_name}: avg={metrics['summary']['average_accuracy']:.2%} | "
                f"mass={metrics['tasks']['mass']['accuracy']:.2%} "
                f"stiff={metrics['tasks']['stiffness']['accuracy']:.2%} "
                f"mat={metrics['tasks']['material']['accuracy']:.2%}"
            )
            online_eval_split(args, split_name=split_name, checkpoint_path=save_dir / "best_model.pth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standard fusion ablation runner with optional online prefix training, tactile aux/supcon, and classical visual augmentation."
    )
    parser.add_argument("--mode", choices=["train", "eval", "online_eval"], default="train")
    parser.add_argument("--data_root", type=str, default="/home/jiaming/Y3_Project/Plaintextdataset")
    parser.add_argument("--save_dir", type=str, default="/home/jiaming/Y3_Project/visuotactile/outputs/fusion_standard_ablation")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--eval_split", choices=["val", "test", "ood_test"], default="test")
    parser.add_argument("--output_dir", type=str, default="")

    parser.add_argument("--variant_name", type=str, default="standard_baseline")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_tactile_len", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--fusion_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.set_defaults(freeze_visual=True)
    parser.add_argument("--freeze_visual", dest="freeze_visual", action="store_true")
    parser.add_argument("--unfreeze_visual", dest="freeze_visual", action="store_false")

    parser.add_argument("--block_modality", choices=["none", "visual", "tactile"], default="none")
    parser.add_argument("--visual_drop_prob", type=float, default=0.0)
    parser.add_argument("--tactile_drop_prob", type=float, default=0.0)

    parser.add_argument("--online_train_prob", type=float, default=0.0)
    parser.add_argument("--online_min_prefix_ratio", type=float, default=0.1)
    parser.add_argument("--min_prefix_len", type=int, default=16)
    parser.add_argument("--online_eval_prefixes", type=str, default="0.1,0.2,0.4,0.6,0.8,1.0")

    parser.add_argument("--lambda_aux", type=float, default=0.0)
    parser.add_argument("--lambda_supcon", type=float, default=0.0)
    parser.add_argument("--supcon_temperature", type=float, default=0.07)
    parser.add_argument("--supcon_task", choices=list(TASKS), default="material")

    parser.add_argument("--augment_policy", choices=["none", "classical"], default="none")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        train(args)
        return
    if not args.checkpoint:
        raise ValueError("--checkpoint is required in eval/online_eval mode")
    if args.mode == "eval":
        metrics = eval_split(args, split_name=args.eval_split, checkpoint_path=Path(args.checkpoint))
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        return
    result = online_eval_split(args, split_name=args.eval_split, checkpoint_path=Path(args.checkpoint))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
