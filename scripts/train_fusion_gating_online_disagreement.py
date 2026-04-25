import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    from train_fusion_gating_online import (
        TASKS,
        SupervisedContrastiveLoss,
        effective_padding_mask,
        parse_prefix_ratios,
    )
    from train_fusion_gating2 import (
        LiveTrainingPlotter,
        _plot_confusion_matrix,
        _plot_summary,
        _plot_training_curves,
        apply_modality_block,
        apply_modality_dropout,
        build_loader,
        max_sequence_length,
        num_cls_tokens,
        resolve_device,
        set_seed,
        task_cls_indices,
    )
except ImportError:  # pragma: no cover
    from visuotactile.scripts.train_fusion_gating_online import (  # type: ignore
        TASKS,
        SupervisedContrastiveLoss,
        effective_padding_mask,
        parse_prefix_ratios,
    )
    from visuotactile.scripts.train_fusion_gating2 import (  # type: ignore
        LiveTrainingPlotter,
        _plot_confusion_matrix,
        _plot_summary,
        _plot_training_curves,
        apply_modality_block,
        apply_modality_dropout,
        build_loader,
        max_sequence_length,
        num_cls_tokens,
        resolve_device,
        set_seed,
        task_cls_indices,
    )


class DisagreementFusionModel(nn.Module):
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
        separate_cls_tokens: bool = False,
    ) -> None:
        super().__init__()
        self.fusion_dim = int(fusion_dim)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.separate_cls_tokens = bool(separate_cls_tokens)
        self.cls_indices = task_cls_indices(self.separate_cls_tokens)

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.vis_backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.vis_proj = nn.Conv2d(512, self.fusion_dim, kernel_size=1)
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
            nn.Conv1d(128, self.fusion_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(self.fusion_dim),
            nn.ReLU(),
        )

        if self.separate_cls_tokens:
            self.task_cls_tokens = nn.Parameter(torch.randn(1, num_cls_tokens(True), self.fusion_dim))
        else:
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.fusion_dim))
        self.pos_emb = nn.Parameter(torch.randn(1, max_sequence_length(self.separate_cls_tokens), self.fusion_dim))

        self.t_null = nn.Parameter(torch.randn(1, 1, self.fusion_dim))
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.fusion_dim * 2, self.fusion_dim),
            nn.ReLU(),
            nn.Linear(self.fusion_dim, 1),
            nn.Sigmoid(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.fusion_dim,
            nhead=self.num_heads,
            dim_feedforward=self.fusion_dim * 4,
            dropout=self.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head_mass = nn.Sequential(
            nn.Linear(self.fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, mass_classes),
        )
        self.head_stiffness = nn.Sequential(
            nn.Linear(self.fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, stiffness_classes),
        )
        self.head_material = nn.Sequential(
            nn.Linear(self.fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, material_classes),
        )

        self.aux_head_mass = nn.Sequential(
            nn.Linear(self.fusion_dim, 64),
            nn.GELU(),
            nn.Linear(64, mass_classes),
        )
        self.aux_head_stiffness = nn.Sequential(
            nn.Linear(self.fusion_dim, 64),
            nn.GELU(),
            nn.Linear(64, stiffness_classes),
        )
        self.aux_head_material = nn.Sequential(
            nn.Linear(self.fusion_dim, 64),
            nn.GELU(),
            nn.Linear(64, material_classes),
        )

        self.visual_head_mass = nn.Sequential(
            nn.Linear(self.fusion_dim, 64),
            nn.GELU(),
            nn.Linear(64, mass_classes),
        )
        self.visual_head_stiffness = nn.Sequential(
            nn.Linear(self.fusion_dim, 64),
            nn.GELU(),
            nn.Linear(64, stiffness_classes),
        )
        self.visual_head_material = nn.Sequential(
            nn.Linear(self.fusion_dim, 64),
            nn.GELU(),
            nn.Linear(64, material_classes),
        )

        self.tactile_projection_head = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim),
            nn.GELU(),
            nn.Linear(self.fusion_dim, 128),
        )

    def _encode_modalities(
        self,
        img: torch.Tensor,
        tac: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        bsz = img.shape[0]
        device = img.device

        v = self.vis_backbone(img)
        v = self.vis_proj(v)
        v_tokens = v.flatten(2).transpose(1, 2)
        num_vis_tokens = v_tokens.shape[1]
        v_global = v_tokens.mean(dim=1)

        t = self.tac_encoder(tac)
        t_tokens = t.transpose(1, 2)
        num_tac_tokens = t_tokens.shape[1]

        full_mask = None
        if padding_mask is not None:
            tac_mask = padding_mask.float().unsqueeze(1)
            tac_mask = torch.nn.functional.max_pool1d(tac_mask, kernel_size=2, stride=2)
            tac_mask = torch.nn.functional.max_pool1d(tac_mask, kernel_size=2, stride=2)
            tac_mask = torch.nn.functional.max_pool1d(tac_mask, kernel_size=2, stride=2)
            tac_mask = tac_mask.squeeze(1) > 0.5
            tac_mask = tac_mask[:, :num_tac_tokens]
            tac_mask_float = (~tac_mask).unsqueeze(-1).float()
            t_global = (t_tokens * tac_mask_float).sum(dim=1) / (tac_mask_float.sum(dim=1) + 1e-8)
            cls_vis_mask = torch.zeros(
                bsz,
                num_cls_tokens(self.separate_cls_tokens) + num_vis_tokens,
                dtype=torch.bool,
                device=device,
            )
            full_mask = torch.cat([cls_vis_mask, tac_mask], dim=1)
        else:
            t_global = t_tokens.mean(dim=1)

        return {
            "v_tokens": v_tokens,
            "v_global": v_global,
            "t_tokens": t_tokens,
            "t_global": t_global,
            "full_mask": full_mask,
        }

    def forward(
        self,
        img: torch.Tensor,
        tac: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        force_gate_value: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        bsz = img.shape[0]
        encoded = self._encode_modalities(img=img, tac=tac, padding_mask=padding_mask)
        v_tokens = encoded["v_tokens"]
        v_global = encoded["v_global"]
        t_tokens = encoded["t_tokens"]
        t_global = encoded["t_global"]
        full_mask = encoded["full_mask"]

        if force_gate_value is None:
            vt_global = torch.cat([v_global, t_global], dim=-1)
            g = self.gate_mlp(vt_global)
        else:
            g = v_tokens.new_full((bsz, 1), float(force_gate_value))

        g_expand = g.unsqueeze(1)
        v_tokens_gated = g_expand * v_tokens + (1.0 - g_expand) * self.t_null

        if self.separate_cls_tokens:
            cls_tokens = self.task_cls_tokens.expand(bsz, -1, -1)
        else:
            cls_tokens = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat([cls_tokens, v_tokens_gated, t_tokens], dim=1)
        seq_len = x.shape[1]
        x = x + self.pos_emb[:, :seq_len, :]
        x = self.transformer_encoder(x, src_key_padding_mask=full_mask)

        mass_cls = x[:, self.cls_indices["mass"], :]
        stiffness_cls = x[:, self.cls_indices["stiffness"], :]
        material_cls = x[:, self.cls_indices["material"], :]
        return {
            "mass": self.head_mass(mass_cls),
            "stiffness": self.head_stiffness(stiffness_cls),
            "material": self.head_material(material_cls),
            "aux_mass": self.aux_head_mass(t_global),
            "aux_stiffness": self.aux_head_stiffness(t_global),
            "aux_material": self.aux_head_material(t_global),
            "vis_mass": self.visual_head_mass(v_global),
            "vis_stiffness": self.visual_head_stiffness(v_global),
            "vis_material": self.visual_head_material(v_global),
            "tactile_global": t_global,
            "visual_global": v_global,
            "contrastive_embedding": torch.nn.functional.normalize(
                self.tactile_projection_head(t_global),
                dim=-1,
            ),
            "gate_score": g.squeeze(-1),
        }


def build_model(
    cfg: Dict,
    args: argparse.Namespace,
    mass_classes: int,
    stiffness_classes: int,
    material_classes: int,
) -> DisagreementFusionModel:
    return DisagreementFusionModel(
        fusion_dim=cfg.get("fusion_dim", args.fusion_dim),
        num_heads=cfg.get("num_heads", args.num_heads),
        dropout=cfg.get("dropout", args.dropout),
        num_layers=cfg.get("num_layers", args.num_layers),
        freeze_visual=cfg.get("freeze_visual", True),
        mass_classes=mass_classes,
        stiffness_classes=stiffness_classes,
        material_classes=material_classes,
        separate_cls_tokens=cfg.get("separate_cls_tokens", getattr(args, "separate_cls_tokens", False)),
    )


def compute_gate_regularization(
    gate_score: torch.Tensor,
    reg_type: str,
    gate_target_mean: float,
    gate_entropy_eps: float,
) -> torch.Tensor:
    if reg_type == "polarization":
        return (gate_score * (1.0 - gate_score)).mean()
    if reg_type == "sparsity":
        return gate_score.mean()
    if reg_type == "mean":
        return (gate_score.mean() - gate_target_mean).pow(2)
    if reg_type == "center":
        return (gate_score - 0.5).pow(2).mean()
    if reg_type == "entropy":
        g_clamped = torch.clamp(gate_score, gate_entropy_eps, 1.0 - gate_entropy_eps)
        entropy = -(
            g_clamped * torch.log(g_clamped)
            + (1.0 - g_clamped) * torch.log(1.0 - g_clamped)
        ).mean()
        return gate_score.new_tensor(math.log(2.0)) - entropy
    return gate_score.new_zeros(())


def compute_clean_losses(
    outputs: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    criterion: nn.Module,
    supcon_criterion: Optional[nn.Module],
    lambda_reg: float,
    lambda_aux: float,
    lambda_visual_aux: float,
    lambda_supcon: float,
    supcon_task: str,
    reg_type: str,
    gate_target_mean: float,
    gate_entropy_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ce_loss = (
        criterion(outputs["mass"], labels["mass"])
        + criterion(outputs["stiffness"], labels["stiffness"])
        + criterion(outputs["material"], labels["material"])
    )
    reg_loss = compute_gate_regularization(
        gate_score=outputs["gate_score"],
        reg_type=reg_type,
        gate_target_mean=gate_target_mean,
        gate_entropy_eps=gate_entropy_eps,
    )
    tactile_aux_loss = (
        criterion(outputs["aux_mass"], labels["mass"])
        + criterion(outputs["aux_stiffness"], labels["stiffness"])
        + criterion(outputs["aux_material"], labels["material"])
    )
    visual_aux_loss = (
        criterion(outputs["vis_mass"], labels["mass"])
        + criterion(outputs["vis_stiffness"], labels["stiffness"])
        + criterion(outputs["vis_material"], labels["material"])
    )
    if supcon_criterion is not None and lambda_supcon > 0.0:
        supcon_loss = supcon_criterion(outputs["contrastive_embedding"], labels[supcon_task])
    else:
        supcon_loss = ce_loss.new_zeros(())
    total_loss = (
        ce_loss
        + lambda_reg * reg_loss
        + lambda_aux * tactile_aux_loss
        + lambda_visual_aux * visual_aux_loss
        + lambda_supcon * supcon_loss
    )
    return total_loss, reg_loss, supcon_loss, tactile_aux_loss, visual_aux_loss


def detached_kl_per_sample(
    visual_logits: torch.Tensor,
    tactile_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    temp = max(float(temperature), 1e-6)
    visual_log_probs = F.log_softmax(visual_logits.detach() / temp, dim=-1)
    tactile_probs = F.softmax(tactile_logits.detach() / temp, dim=-1)
    return F.kl_div(visual_log_probs, tactile_probs, reduction="none").sum(dim=-1) * (temp ** 2)


def compute_detached_disagreement(
    outputs: Dict[str, torch.Tensor],
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    disagreements = []
    agreements = []
    for task in TASKS:
        visual_key = f"vis_{task}"
        tactile_key = f"aux_{task}"
        disagreements.append(
            detached_kl_per_sample(
                visual_logits=outputs[visual_key],
                tactile_logits=outputs[tactile_key],
                temperature=temperature,
            )
        )
        agreements.append(
            (
                outputs[visual_key].detach().argmax(dim=1)
                == outputs[tactile_key].detach().argmax(dim=1)
            ).float()
        )
    disagreement = torch.stack(disagreements, dim=1).mean(dim=1)
    agreement = torch.stack(agreements, dim=1).mean(dim=1)
    return disagreement, agreement


def build_label_signature(labels: Dict[str, torch.Tensor]) -> torch.Tensor:
    return labels["mass"] * 100 + labels["stiffness"] * 10 + labels["material"]


def build_visual_mismatch_batch(
    images: torch.Tensor,
    labels: Dict[str, torch.Tensor],
    mismatch_prob: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = images.size(0)
    device = images.device
    if mismatch_prob <= 0.0 or batch_size < 2:
        return images, torch.zeros(batch_size, dtype=torch.bool, device=device)

    if mismatch_prob >= 1.0:
        mismatch_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    else:
        mismatch_mask = torch.rand(batch_size, device=device) < mismatch_prob
        if not mismatch_mask.any():
            return images, mismatch_mask

    signatures = build_label_signature(labels).detach().cpu().tolist()
    target_indices = mismatch_mask.nonzero(as_tuple=False).flatten().tolist()
    source_indices: List[int] = []
    valid_targets: List[int] = []
    all_indices = list(range(batch_size))

    for target_idx in target_indices:
        diff_candidates = [idx for idx in all_indices if idx != target_idx and signatures[idx] != signatures[target_idx]]
        if not diff_candidates:
            diff_candidates = [idx for idx in all_indices if idx != target_idx]
        if not diff_candidates:
            continue
        rand_pos = torch.randint(len(diff_candidates), (1,), device=device).item()
        valid_targets.append(target_idx)
        source_indices.append(diff_candidates[rand_pos])

    if not source_indices:
        return images, torch.zeros(batch_size, dtype=torch.bool, device=device)

    mismatch_images = images.clone()
    target_tensor = torch.tensor(valid_targets, dtype=torch.long, device=device)
    source_tensor = torch.tensor(source_indices, dtype=torch.long, device=device)
    mismatch_images[target_tensor] = images[source_tensor]

    valid_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    valid_mask[target_tensor] = True
    return mismatch_images, valid_mask


def compute_metrics(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    train_mode: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lambda_reg: float = 0.0,
    fixed_prefix_ratio: Optional[float] = None,
) -> Dict[str, float]:
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_reg = 0.0
    total_supcon = 0.0
    total_tactile_aux = 0.0
    total_visual_aux = 0.0
    total_gate = 0.0
    total_disagreement_penalty = 0.0
    total_disagreement = 0.0
    total_agreement = 0.0
    total_samples = 0

    correct_fused = {task: 0 for task in TASKS}
    correct_tactile = {task: 0 for task in TASKS}
    correct_visual = {task: 0 for task in TASKS}

    supcon_criterion = None
    if args.lambda_supcon > 0.0:
        supcon_criterion = SupervisedContrastiveLoss(temperature=args.supcon_temperature)

    iterator = loader
    if tqdm is not None:
        desc = "train" if train_mode else ("prefix_eval" if fixed_prefix_ratio is not None else "eval")
        iterator = tqdm(loader, leave=False, desc=desc)

    for batch_idx, batch in enumerate(iterator, start=1):
        images = batch["image"].to(device)
        tactile = batch["tactile"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        labels = {task: batch[task].to(device) for task in TASKS}

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

        if train_mode:
            if optimizer is None:
                raise ValueError("optimizer is required when train_mode=True")
            optimizer.zero_grad()
            outputs = model(images, tactile, padding_mask=prefix_mask)
            clean_loss, reg_loss, supcon_loss, tactile_aux_loss, visual_aux_loss = compute_clean_losses(
                outputs=outputs,
                labels=labels,
                criterion=criterion,
                supcon_criterion=supcon_criterion,
                lambda_reg=lambda_reg,
                lambda_aux=args.lambda_aux,
                lambda_visual_aux=args.lambda_visual_aux,
                lambda_supcon=args.lambda_supcon,
                supcon_task=args.supcon_task,
                reg_type=args.reg_type,
                gate_target_mean=args.gate_target_mean,
                gate_entropy_eps=args.gate_entropy_eps,
            )
            disagreement_score, agreement_score = compute_detached_disagreement(
                outputs=outputs,
                temperature=args.disagreement_temperature,
            )
            disagreement_penalty = (outputs["gate_score"] * disagreement_score).mean()
            total_objective = clean_loss + args.lambda_disagreement * disagreement_penalty
            total_objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss = total_objective
        else:
            with torch.no_grad():
                outputs = model(images, tactile, padding_mask=prefix_mask)
                clean_loss, reg_loss, supcon_loss, tactile_aux_loss, visual_aux_loss = compute_clean_losses(
                    outputs=outputs,
                    labels=labels,
                    criterion=criterion,
                    supcon_criterion=supcon_criterion,
                    lambda_reg=lambda_reg,
                    lambda_aux=args.lambda_aux,
                    lambda_visual_aux=args.lambda_visual_aux,
                    lambda_supcon=args.lambda_supcon,
                    supcon_task=args.supcon_task,
                    reg_type=args.reg_type,
                    gate_target_mean=args.gate_target_mean,
                    gate_entropy_eps=args.gate_entropy_eps,
                )
                disagreement_score, agreement_score = compute_detached_disagreement(
                    outputs=outputs,
                    temperature=args.disagreement_temperature,
                )
                disagreement_penalty = (outputs["gate_score"] * disagreement_score).mean()
                loss = clean_loss + args.lambda_disagreement * disagreement_penalty

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_reg += reg_loss.item() * batch_size
        total_supcon += supcon_loss.item() * batch_size
        total_tactile_aux += tactile_aux_loss.item() * batch_size
        total_visual_aux += visual_aux_loss.item() * batch_size
        total_gate += outputs["gate_score"].sum().item()
        total_disagreement_penalty += disagreement_penalty.item() * batch_size
        total_disagreement += disagreement_score.sum().item()
        total_agreement += agreement_score.sum().item()
        total_samples += batch_size

        for task in TASKS:
            correct_fused[task] += (outputs[task].argmax(dim=1) == labels[task]).sum().item()
            correct_tactile[task] += (outputs[f"aux_{task}"].argmax(dim=1) == labels[task]).sum().item()
            correct_visual[task] += (outputs[f"vis_{task}"].argmax(dim=1) == labels[task]).sum().item()

        if tqdm is not None and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                {
                    "loss": f"{total_loss / max(1, total_samples):.4f}",
                    "mass": f"{correct_fused['mass'] / max(1, total_samples):.2%}",
                    "stiff": f"{correct_fused['stiffness'] / max(1, total_samples):.2%}",
                    "mat": f"{correct_fused['material'] / max(1, total_samples):.2%}",
                    "g": f"{total_gate / max(1, total_samples):.3f}",
                    "dis": f"{total_disagreement / max(1, total_samples):.4f}",
                    "gdis": f"{total_disagreement_penalty / max(1, total_samples):.4f}",
                    "step": batch_idx,
                }
            )

    fused_avg = float(np.mean([correct_fused[task] / max(1, total_samples) for task in TASKS]))
    tactile_avg = float(np.mean([correct_tactile[task] / max(1, total_samples) for task in TASKS]))
    visual_avg = float(np.mean([correct_visual[task] / max(1, total_samples) for task in TASKS]))
    return {
        "loss": total_loss / max(1, total_samples),
        "reg_loss": total_reg / max(1, total_samples),
        "aux_loss": total_tactile_aux / max(1, total_samples),
        "visual_aux_loss": total_visual_aux / max(1, total_samples),
        "supcon_loss": total_supcon / max(1, total_samples),
        "gate_score": total_gate / max(1, total_samples),
        "mass": correct_fused["mass"] / max(1, total_samples),
        "stiffness": correct_fused["stiffness"] / max(1, total_samples),
        "material": correct_fused["material"] / max(1, total_samples),
        "tactile_mass": correct_tactile["mass"] / max(1, total_samples),
        "tactile_stiffness": correct_tactile["stiffness"] / max(1, total_samples),
        "tactile_material": correct_tactile["material"] / max(1, total_samples),
        "visual_mass": correct_visual["mass"] / max(1, total_samples),
        "visual_stiffness": correct_visual["stiffness"] / max(1, total_samples),
        "visual_material": correct_visual["material"] / max(1, total_samples),
        "fused_average_accuracy": fused_avg,
        "tactile_average_accuracy": tactile_avg,
        "visual_average_accuracy": visual_avg,
        "detached_disagreement": total_disagreement / max(1, total_samples),
        "gate_weighted_disagreement": total_disagreement_penalty / max(1, total_samples),
        "unimodal_prediction_agreement": total_agreement / max(1, total_samples),
        "num_samples": float(total_samples),
    }


def extract_val_disagreement_metrics(val_metrics: Dict[str, float]) -> Dict[str, float]:
    return {
        "detached_disagreement": val_metrics["detached_disagreement"],
        "gate_weighted_disagreement": val_metrics["gate_weighted_disagreement"],
        "unimodal_prediction_agreement": val_metrics["unimodal_prediction_agreement"],
        "visual_average_accuracy": val_metrics["visual_average_accuracy"],
        "tactile_average_accuracy": val_metrics["tactile_average_accuracy"],
        "num_samples": val_metrics["num_samples"],
    }


def reliable_selection_start_epoch(args: argparse.Namespace) -> int:
    if args.reliable_selection_start_epoch > 0:
        return int(args.reliable_selection_start_epoch)
    if args.reg_type == "none":
        return 1
    return max(1, args.gate_reg_warmup_epochs + 1)


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    args: argparse.Namespace,
    val_metrics: Dict[str, float],
    val_disagreement_metrics: Dict[str, float],
    avg_val_acc: float,
    lambda_reg_eff: float,
    role: str,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": vars(args),
            "val_metrics": val_metrics,
            "val_disagreement_metrics": val_disagreement_metrics,
            "avg_val_acc": avg_val_acc,
            "lambda_reg_eff": lambda_reg_eff,
            "checkpoint_role": role,
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(f"Expected both train/ and val/ under {data_root}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed, args.device)

    device = resolve_device(args.device)
    reliable_epoch_start = reliable_selection_start_epoch(args)
    print(f"device: {device}")
    print(f"block_modality: {args.block_modality}")
    print(
        f"online training: prob={args.online_train_prob}, "
        f"min_prefix_ratio={args.online_min_prefix_ratio}, min_prefix_len={args.min_prefix_len}"
    )
    print(
        f"gating regularization: {args.reg_type} with weight {args.lambda_reg} "
        f"(warmup={args.gate_reg_warmup_epochs}, ramp={args.gate_reg_ramp_epochs}, target_mean={args.gate_target_mean})"
    )
    print(
        f"disagreement safeguard: lambda={args.lambda_disagreement}, "
        f"temperature={args.disagreement_temperature}, visual_aux={args.lambda_visual_aux}"
    )
    print(
        f"reliable checkpoint selection starts at epoch {reliable_epoch_start}, "
        f"primary={args.primary_checkpoint}"
    )
    print(
        f"supcon: weight={args.lambda_supcon}, task={args.supcon_task}, "
        f"temperature={args.supcon_temperature}"
    )

    train_loader = build_loader(train_dir, args.batch_size, args.max_tactile_len, args.num_workers, shuffle=True)
    val_loader = build_loader(val_dir, args.batch_size, args.max_tactile_len, args.num_workers, shuffle=False)
    print(f"train samples: {len(train_loader.dataset)} | val samples: {len(val_loader.dataset)}")

    args.mass_classes = len(train_loader.dataset.mass_to_idx)
    args.stiffness_classes = len(train_loader.dataset.stiffness_to_idx)
    args.material_classes = len(train_loader.dataset.material_to_idx)

    model = build_model(
        vars(args),
        args,
        args.mass_classes,
        args.stiffness_classes,
        args.material_classes,
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
    best_acc_value = -1.0
    best_acc_epoch = -1
    best_reliable_score = None
    best_reliable_epoch = -1
    best_reliable_avg = -1.0
    early_stop_streak = 0

    live_plotter = None
    if args.live_plot:
        try:
            live_plotter = LiveTrainingPlotter(save_dir)
        except Exception as exc:
            print(f"live plot disabled: {exc}")

    for epoch in range(1, args.epochs + 1):
        if args.reg_type == "none":
            lambda_reg_eff = 0.0
        elif epoch <= args.gate_reg_warmup_epochs:
            lambda_reg_eff = 0.0
        elif args.gate_reg_ramp_epochs > 0:
            progress = (epoch - args.gate_reg_warmup_epochs) / max(1, args.gate_reg_ramp_epochs)
            lambda_reg_eff = float(args.lambda_reg) * float(min(1.0, max(0.0, progress)))
        else:
            lambda_reg_eff = float(args.lambda_reg)

        train_metrics = compute_metrics(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            args=args,
            train_mode=True,
            optimizer=optimizer,
            lambda_reg=lambda_reg_eff,
        )
        val_metrics = compute_metrics(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            args=args,
            train_mode=False,
            lambda_reg=lambda_reg_eff,
        )
        val_disagreement_metrics = extract_val_disagreement_metrics(val_metrics)
        scheduler.step()

        avg_val = float(np.mean([val_metrics[task] for task in TASKS]))
        best_acc_updated = False
        best_reliable_updated = False

        if avg_val > best_acc_value:
            best_acc_value = avg_val
            best_acc_epoch = epoch
            best_acc_updated = True
            save_checkpoint(
                path=save_dir / "best_acc.pth",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                val_metrics=val_metrics,
                val_disagreement_metrics=val_disagreement_metrics,
                avg_val_acc=avg_val,
                lambda_reg_eff=lambda_reg_eff,
                role="best_acc",
            )

        reliable_candidate = epoch >= reliable_epoch_start and val_disagreement_metrics["num_samples"] > 0
        reliable_score = (
            avg_val,
            val_disagreement_metrics["tactile_average_accuracy"],
            val_disagreement_metrics["visual_average_accuracy"],
            -val_disagreement_metrics["gate_weighted_disagreement"],
            -val_disagreement_metrics["detached_disagreement"],
            -val_metrics["reg_loss"],
            -val_metrics["loss"],
            epoch,
        )
        if reliable_candidate and (best_reliable_score is None or reliable_score > best_reliable_score):
            best_reliable_score = reliable_score
            best_reliable_epoch = epoch
            best_reliable_avg = avg_val
            best_reliable_updated = True
            save_checkpoint(
                path=save_dir / "best_reliable.pth",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                val_metrics=val_metrics,
                val_disagreement_metrics=val_disagreement_metrics,
                avg_val_acc=avg_val,
                lambda_reg_eff=lambda_reg_eff,
                role="best_reliable",
            )

        history.append(
            {
                "epoch": epoch,
                "lambda_reg_eff": lambda_reg_eff,
                "train": train_metrics,
                "val": val_metrics,
                "val_disagreement": val_disagreement_metrics,
                "avg_val_acc": avg_val,
                "reliable_candidate": reliable_candidate,
                "best_acc_updated": best_acc_updated,
                "best_reliable_updated": best_reliable_updated,
            }
        )
        if live_plotter is not None:
            live_plotter.update(epoch, train_metrics, val_metrics)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} | "
            f"val_mass={val_metrics['mass']:.2%} val_stiff={val_metrics['stiffness']:.2%} "
            f"val_mat={val_metrics['material']:.2%} | val_g={val_metrics['gate_score']:.3f} "
            f"val_dis={val_disagreement_metrics['detached_disagreement']:.4f} "
            f"val_gdis={val_disagreement_metrics['gate_weighted_disagreement']:.4f} "
            f"val_agree={val_disagreement_metrics['unimodal_prediction_agreement']:.3f}"
        )

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

    (save_dir / "training_history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False))
    _plot_training_curves(history, save_dir)
    if live_plotter is not None:
        live_plotter.save()
        live_plotter.close()

    preferred_name = "best_reliable.pth" if args.primary_checkpoint == "reliable" and best_reliable_epoch > 0 else "best_acc.pth"
    preferred_path = save_dir / preferred_name
    if not preferred_path.exists():
        raise FileNotFoundError(f"Preferred checkpoint missing: {preferred_path}")
    shutil.copyfile(preferred_path, save_dir / "best_model.pth")

    checkpoint_summary = {
        "primary_checkpoint": preferred_name,
        "reliable_selection_start_epoch": reliable_epoch_start,
        "best_acc": {
            "epoch": best_acc_epoch,
            "avg_val_acc": best_acc_value,
            "path": str(save_dir / "best_acc.pth"),
        },
        "best_reliable": {
            "epoch": best_reliable_epoch,
            "avg_val_acc": best_reliable_avg,
            "path": str(save_dir / "best_reliable.pth") if best_reliable_epoch > 0 else "",
        },
    }
    (save_dir / "checkpoint_selection_summary.json").write_text(
        json.dumps(checkpoint_summary, indent=2, ensure_ascii=False)
    )

    print(f"best acc epoch: {best_acc_epoch} | best val avg acc: {best_acc_value:.2%}")
    if best_reliable_epoch > 0:
        print(f"best reliable epoch: {best_reliable_epoch} | best reliable avg acc: {best_reliable_avg:.2%}")
    else:
        print("best reliable epoch: none")
    print(f"primary checkpoint: {preferred_name}")

    for split_name in ["test", "ood_test"]:
        split_dir = data_root / split_name
        if split_dir.is_dir():
            metrics = eval_split(args, split_name=split_name, checkpoint_path=save_dir / "best_model.pth")
            print(
                f"{split_name}: loss={metrics['loss']:.4f}, "
                f"mass={metrics['mass']:.2%}, stiffness={metrics['stiffness']:.2%}, "
                f"material={metrics['material']:.2%}, avg_acc={metrics['summary']['average_accuracy']:.2%}, "
                f"avg_g={metrics['avg_gate_score']:.3f}"
            )


def eval_split(args: argparse.Namespace, split_name: str, checkpoint_path: Optional[Path] = None) -> Dict[str, float]:
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

    checkpoint = torch.load(ckpt_path, map_location=resolve_device(args.device), weights_only=False)
    cfg = checkpoint.get("config", {})

    device = resolve_device(args.device)
    loader = build_loader(split_dir, args.batch_size, args.max_tactile_len, args.num_workers, shuffle=False)
    dataset = loader.dataset

    model = build_model(
        cfg,
        args,
        mass_classes=len(dataset.mass_to_idx),
        stiffness_classes=len(dataset.stiffness_to_idx),
        material_classes=len(dataset.material_to_idx),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()
    base_metrics = compute_metrics(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        args=args,
        train_mode=False,
        lambda_reg=cfg.get("lambda_reg", args.lambda_reg),
    )

    all_preds = {task: [] for task in TASKS}
    all_labels = {task: [] for task in TASKS}
    all_gate_scores: List[float] = []

    visual_correct = {task: 0 for task in TASKS}
    tactile_correct = {task: 0 for task in TASKS}
    disagreement_sum = 0.0
    agreement_sum = 0.0
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            tactile = batch["tactile"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            prefix_mask = effective_padding_mask(
                padding_mask=padding_mask,
                train_mode=False,
                online_train_prob=0.0,
                online_min_prefix_ratio=args.online_min_prefix_ratio,
                min_prefix_len=args.min_prefix_len,
                fixed_ratio=None,
            )
            images, tactile = apply_modality_block(images, tactile, args.block_modality)
            outputs = model(images, tactile, padding_mask=prefix_mask)
            disagreement, agreement = compute_detached_disagreement(
                outputs=outputs,
                temperature=cfg.get("disagreement_temperature", args.disagreement_temperature),
            )
            batch_size = images.size(0)
            total_samples += batch_size
            disagreement_sum += disagreement.sum().item()
            agreement_sum += agreement.sum().item()
            for task in TASKS:
                preds = outputs[task].argmax(dim=1)
                labels = batch[task].to(device)
                all_preds[task].extend(preds.cpu().tolist())
                all_labels[task].extend(batch[task].tolist())
                visual_correct[task] += int((outputs[f"vis_{task}"].argmax(dim=1) == labels).sum().item())
                tactile_correct[task] += int((outputs[f"aux_{task}"].argmax(dim=1) == labels).sum().item())
            all_gate_scores.extend(outputs["gate_score"].cpu().tolist())

    label_names = {
        "mass": list(dataset.mass_to_idx.keys()),
        "stiffness": list(dataset.stiffness_to_idx.keys()),
        "material": list(dataset.material_to_idx.keys()),
    }

    results = {}
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

    avg_acc = float(np.mean([results[task]["accuracy"] for task in TASKS]))
    avg_macro_f1 = float(np.mean([results[task]["macro"]["f1"] for task in TASKS]))
    avg_weighted_f1 = float(np.mean([results[task]["weighted"]["f1"] for task in TASKS]))

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        suffix = "" if args.block_modality == "none" else f"_block_{args.block_modality}"
        output_dir = ckpt_path.parent / f"eval_{split_name}{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    for task in TASKS:
        _plot_confusion_matrix(all_labels[task], all_preds[task], label_names[task], task, output_dir)
    _plot_summary(results, output_dir)

    visual_summary = {
        task: float(visual_correct[task] / max(1, total_samples))
        for task in TASKS
    }
    tactile_summary = {
        task: float(tactile_correct[task] / max(1, total_samples))
        for task in TASKS
    }

    full_result = {
        "split": split_name,
        "block_modality": args.block_modality,
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "num_samples": len(dataset),
        "loss": float(base_metrics["loss"]),
        "mass": float(base_metrics["mass"]),
        "stiffness": float(base_metrics["stiffness"]),
        "material": float(base_metrics["material"]),
        "avg_gate_score": float(np.mean(all_gate_scores)) if all_gate_scores else 0.0,
        "summary": {
            "average_accuracy": avg_acc,
            "average_macro_f1": avg_macro_f1,
            "average_weighted_f1": avg_weighted_f1,
        },
        "visual_only": {
            **visual_summary,
            "average_accuracy": float(np.mean(list(visual_summary.values()))),
        },
        "tactile_only": {
            **tactile_summary,
            "average_accuracy": float(np.mean(list(tactile_summary.values()))),
        },
        "disagreement": {
            "average_kl": disagreement_sum / max(1, total_samples),
            "gate_weighted_average": float(base_metrics["gate_weighted_disagreement"]),
            "prediction_agreement": agreement_sum / max(1, total_samples),
        },
        "gate_scores": all_gate_scores,
        "tasks": results,
    }
    (output_dir / "evaluation_results.json").write_text(json.dumps(full_result, indent=2, ensure_ascii=False))
    return full_result


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

    checkpoint = torch.load(ckpt_path, map_location=resolve_device(args.device), weights_only=False)
    cfg = checkpoint.get("config", {})
    device = resolve_device(args.device)
    loader = build_loader(split_dir, args.batch_size, args.max_tactile_len, args.num_workers, shuffle=False)
    dataset = loader.dataset

    model = build_model(
        cfg,
        args,
        mass_classes=len(dataset.mass_to_idx),
        stiffness_classes=len(dataset.stiffness_to_idx),
        material_classes=len(dataset.material_to_idx),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()
    curves = []
    for ratio in parse_prefix_ratios(args.prefix_ratios):
        metrics = compute_metrics(
            model=model,
            loader=loader,
            criterion=criterion,
            device=device,
            args=args,
            train_mode=False,
            lambda_reg=cfg.get("lambda_reg", args.lambda_reg),
            fixed_prefix_ratio=ratio,
        )
        curves.append(
            {
                "prefix_ratio": ratio,
                "loss": float(metrics["loss"]),
                "gate_score": float(metrics["gate_score"]),
                "mass": float(metrics["mass"]),
                "stiffness": float(metrics["stiffness"]),
                "material": float(metrics["material"]),
                "average_accuracy": float(np.mean([metrics[task] for task in TASKS])),
                "visual_average_accuracy": float(metrics["visual_average_accuracy"]),
                "tactile_average_accuracy": float(metrics["tactile_average_accuracy"]),
                "disagreement": float(metrics["detached_disagreement"]),
                "gate_weighted_disagreement": float(metrics["gate_weighted_disagreement"]),
            }
        )

    result = {
        "split": split_name,
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "num_samples": len(dataset),
        "prefix_curves": curves,
    }

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = ckpt_path.parent / f"online_eval_{split_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "online_evaluation_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


@torch.no_grad()
def evaluate_condition(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    fixed_prefix_ratio: float,
    force_gate_value: Optional[float],
    use_visual_mismatch: bool,
) -> Dict[str, object]:
    model.eval()
    total_samples = 0
    total_gate = 0.0
    correct = {task: 0 for task in TASKS}
    all_gate_scores: List[float] = []

    for batch in loader:
        images = batch["image"].to(device)
        tactile = batch["tactile"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        labels = {task: batch[task].to(device) for task in TASKS}

        prefix_mask = effective_padding_mask(
            padding_mask=padding_mask,
            train_mode=False,
            online_train_prob=0.0,
            online_min_prefix_ratio=args.online_min_prefix_ratio,
            min_prefix_len=args.min_prefix_len,
            fixed_ratio=None if fixed_prefix_ratio >= 1.0 else fixed_prefix_ratio,
        )
        images, tactile = apply_modality_block(images, tactile, args.block_modality)
        if use_visual_mismatch:
            images, _ = build_visual_mismatch_batch(images=images, labels=labels, mismatch_prob=1.0)

        outputs = model(
            images,
            tactile,
            padding_mask=prefix_mask,
            force_gate_value=force_gate_value,
        )
        batch_size = images.size(0)
        total_samples += batch_size
        gate_scores = outputs["gate_score"].detach().cpu().tolist()
        all_gate_scores.extend(gate_scores)
        total_gate += float(np.sum(gate_scores))
        for task in TASKS:
            preds = outputs[task].argmax(dim=1)
            correct[task] += int((preds == labels[task]).sum().item())

    task_acc = {task: correct[task] / max(1, total_samples) for task in TASKS}
    avg_acc = float(np.mean(list(task_acc.values())))
    return {
        "num_samples": total_samples,
        "avg_gate_score": total_gate / max(1, total_samples),
        "summary": {
            "average_accuracy": avg_acc,
        },
        "tasks": {task: {"accuracy": float(task_acc[task])} for task in TASKS},
        "gate_scores": all_gate_scores,
    }


def diagnose_visual_residual_contribution(
    args: argparse.Namespace,
    checkpoint_path: Optional[Path] = None,
) -> Dict[str, object]:
    ckpt_path = Path(checkpoint_path) if checkpoint_path else Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=resolve_device(args.device), weights_only=False)
    cfg = checkpoint.get("config", {})
    device = resolve_device(args.device)
    criterion = nn.CrossEntropyLoss()
    split_names = [item.strip() for item in args.diagnostic_splits.split(",") if item.strip()]

    conditions = {}
    for split_name in split_names:
        split_dir = Path(args.data_root) / split_name
        if not split_dir.is_dir():
            continue
        loader = build_loader(split_dir, args.batch_size, args.max_tactile_len, args.num_workers, shuffle=False)
        dataset = loader.dataset
        model = build_model(
            cfg,
            args,
            mass_classes=len(dataset.mass_to_idx),
            stiffness_classes=len(dataset.stiffness_to_idx),
            material_classes=len(dataset.material_to_idx),
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        conditions[split_name] = {
            "original": evaluate_condition(
                model=model,
                loader=loader,
                criterion=criterion,
                device=device,
                args=args,
                fixed_prefix_ratio=args.diagnostic_prefix_ratio,
                force_gate_value=None,
                use_visual_mismatch=False,
            ),
            "force_gate_zero": evaluate_condition(
                model=model,
                loader=loader,
                criterion=criterion,
                device=device,
                args=args,
                fixed_prefix_ratio=args.diagnostic_prefix_ratio,
                force_gate_value=0.0,
                use_visual_mismatch=False,
            ),
            "visual_mismatch": evaluate_condition(
                model=model,
                loader=loader,
                criterion=criterion,
                device=device,
                args=args,
                fixed_prefix_ratio=args.diagnostic_prefix_ratio,
                force_gate_value=None,
                use_visual_mismatch=True,
            ),
        }

    result = {
        "checkpoint": str(ckpt_path),
        "device": str(device),
        "prefix_ratio": float(args.diagnostic_prefix_ratio),
        "conditions": conditions,
    }

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = ckpt_path.parent / "visual_residual_diagnostic"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "visual_residual_diagnostic.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved diagnostic results to {output_path}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Online-prefix gating fusion with disagreement-based safeguard and detached unimodal disagreement"
    )
    parser.add_argument("--mode", choices=["train", "eval", "online_eval", "diagnose_visual"], default="train")
    parser.add_argument("--data_root", type=str, default="/home/martina/Y3_Project/Plaintextdataset")
    parser.add_argument("--save_dir", type=str, default="outputs/fusion_gating_online_disagreement")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--eval_split", choices=["val", "test", "ood_test"], default="test")
    parser.add_argument("--output_dir", type=str, default="")
    parser.set_defaults(live_plot=True)
    parser.add_argument("--live_plot", dest="live_plot", action="store_true", help="Enable live training plots")
    parser.add_argument("--no_live_plot", dest="live_plot", action="store_false", help="Disable live training plots")
    parser.add_argument(
        "--block_modality",
        type=str,
        default="none",
        choices=["none", "visual", "tactile"],
        help="Block specific modality: none | visual | tactile",
    )

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
    parser.set_defaults(separate_cls_tokens=False)
    parser.add_argument("--separate_cls_tokens", dest="separate_cls_tokens", action="store_true")
    parser.add_argument("--shared_cls_token", dest="separate_cls_tokens", action="store_false")
    parser.set_defaults(freeze_visual=True)
    parser.add_argument("--freeze_visual", dest="freeze_visual", action="store_true")
    parser.add_argument("--unfreeze_visual", dest="freeze_visual", action="store_false")
    parser.add_argument("--visual_drop_prob", type=float, default=0.0)
    parser.add_argument("--tactile_drop_prob", type=float, default=0.0)
    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--early_stop_acc", type=float, default=1.0)
    parser.add_argument("--early_stop_min_epoch", type=int, default=0)

    parser.add_argument("--lambda_reg", type=float, default=0.1)
    parser.add_argument("--lambda_aux", type=float, default=0.5)
    parser.add_argument("--lambda_visual_aux", type=float, default=0.5)
    parser.add_argument("--lambda_supcon", type=float, default=0.0)
    parser.add_argument("--lambda_disagreement", type=float, default=1.0)
    parser.add_argument("--disagreement_temperature", type=float, default=1.0)
    parser.add_argument("--supcon_temperature", type=float, default=0.07)
    parser.add_argument("--supcon_task", type=str, default="material", choices=list(TASKS))
    parser.add_argument(
        "--reg_type",
        type=str,
        default="entropy",
        choices=["polarization", "sparsity", "mean", "center", "entropy", "none"],
    )
    parser.add_argument("--gate_target_mean", type=float, default=0.5)
    parser.add_argument("--gate_entropy_eps", type=float, default=1e-6)
    parser.add_argument("--gate_reg_warmup_epochs", type=int, default=5)
    parser.add_argument("--gate_reg_ramp_epochs", type=int, default=10)

    parser.add_argument("--online_train_prob", type=float, default=1.0, help="Probability of random prefix training")
    parser.add_argument("--online_min_prefix_ratio", type=float, default=0.2, help="Minimum prefix ratio during training")
    parser.add_argument("--min_prefix_len", type=int, default=64, help="Minimum tactile prefix length in raw timesteps")
    parser.add_argument(
        "--prefix_ratios",
        type=str,
        default="0.1,0.2,0.4,0.6,0.8,1.0",
        help="Comma-separated ratios for online_eval",
    )
    parser.add_argument(
        "--reliable_selection_start_epoch",
        type=int,
        default=0,
        help="First epoch allowed to compete for best_reliable.pth; 0 means auto",
    )
    parser.add_argument(
        "--primary_checkpoint",
        type=str,
        default="reliable",
        choices=["reliable", "acc"],
        help="Which saved checkpoint should also be exported as best_model.pth",
    )
    parser.add_argument("--diagnostic_splits", type=str, default="test,ood_test")
    parser.add_argument("--diagnostic_prefix_ratio", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.mode == "train":
        train(cli_args)
    elif cli_args.mode == "eval":
        if not cli_args.checkpoint:
            raise ValueError("--checkpoint is required in eval mode")
        metrics = eval_split(cli_args, split_name=cli_args.eval_split, checkpoint_path=Path(cli_args.checkpoint))
        print(
            f"{cli_args.eval_split}: loss={metrics['loss']:.4f}, "
            f"mass={metrics['mass']:.2%}, stiffness={metrics['stiffness']:.2%}, "
            f"material={metrics['material']:.2%}, avg_acc={metrics['summary']['average_accuracy']:.2%}, "
            f"avg_g={metrics['avg_gate_score']:.3f}"
        )
    elif cli_args.mode == "online_eval":
        if not cli_args.checkpoint:
            raise ValueError("--checkpoint is required in online_eval mode")
        result = online_eval_split(cli_args, split_name=cli_args.eval_split, checkpoint_path=Path(cli_args.checkpoint))
        for point in result["prefix_curves"]:
            print(
                f"prefix={point['prefix_ratio']:.2f}: "
                f"avg_acc={point['average_accuracy']:.2%}, "
                f"mass={point['mass']:.2%}, stiffness={point['stiffness']:.2%}, "
                f"material={point['material']:.2%}, g={point['gate_score']:.3f}"
            )
    else:
        if not cli_args.checkpoint:
            raise ValueError("--checkpoint is required in diagnose_visual mode")
        diagnose_visual_residual_contribution(cli_args, checkpoint_path=Path(cli_args.checkpoint))
