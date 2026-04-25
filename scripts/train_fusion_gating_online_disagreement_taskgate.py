import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

try:
    import train_fusion_gating_online_disagreement as base
except ImportError:  # pragma: no cover
    from visuotactile.scripts import train_fusion_gating_online_disagreement as base  # type: ignore


TASKS = base.TASKS
TASK_TO_INDEX = {task: idx for idx, task in enumerate(TASKS)}
ORIGINAL_EXTRACT_VAL_DISAGREEMENT_METRICS = base.extract_val_disagreement_metrics


class TaskGateDisagreementFusionModel(nn.Module):
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
        del separate_cls_tokens
        self.fusion_dim = int(fusion_dim)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)

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

        self.task_cls_tokens = nn.Parameter(torch.randn(1, len(TASKS), self.fusion_dim))
        self.pos_emb = nn.Parameter(torch.randn(1, base.max_sequence_length(False), self.fusion_dim))

        self.t_null = nn.Parameter(torch.randn(1, 1, self.fusion_dim))
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.fusion_dim * 2, self.fusion_dim),
            nn.ReLU(),
            nn.Linear(self.fusion_dim, len(TASKS)),
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
            cls_vis_mask = torch.zeros(bsz, 1 + num_vis_tokens, dtype=torch.bool, device=device)
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
            task_gates = self.gate_mlp(vt_global)
        else:
            task_gates = v_tokens.new_full((bsz, len(TASKS)), float(force_gate_value))

        task_features: Dict[str, torch.Tensor] = {}
        for task, task_idx in TASK_TO_INDEX.items():
            g_task = task_gates[:, task_idx : task_idx + 1]
            v_tokens_gated = g_task.unsqueeze(1) * v_tokens + (1.0 - g_task.unsqueeze(1)) * self.t_null
            cls_token = self.task_cls_tokens[:, task_idx : task_idx + 1, :].expand(bsz, -1, -1)
            x_task = torch.cat([cls_token, v_tokens_gated, t_tokens], dim=1)
            x_task = x_task + self.pos_emb[:, : x_task.shape[1], :]
            x_task = self.transformer_encoder(x_task, src_key_padding_mask=full_mask)
            task_features[task] = x_task[:, 0, :]

        return {
            "mass": self.head_mass(task_features["mass"]),
            "stiffness": self.head_stiffness(task_features["stiffness"]),
            "material": self.head_material(task_features["material"]),
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
            "task_gate_scores": task_gates,
            "gate_score": task_gates.mean(dim=1),
            "gate_mass": task_gates[:, TASK_TO_INDEX["mass"]],
            "gate_stiffness": task_gates[:, TASK_TO_INDEX["stiffness"]],
            "gate_material": task_gates[:, TASK_TO_INDEX["material"]],
        }


def build_model(
    cfg: Dict,
    args: argparse.Namespace,
    mass_classes: int,
    stiffness_classes: int,
    material_classes: int,
) -> TaskGateDisagreementFusionModel:
    return TaskGateDisagreementFusionModel(
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
    reg_loss = base.compute_gate_regularization(
        gate_score=outputs["task_gate_scores"],
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


def compute_detached_task_disagreement(
    outputs: Dict[str, torch.Tensor],
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    disagreements = []
    agreements = []
    for task in TASKS:
        visual_key = f"vis_{task}"
        tactile_key = f"aux_{task}"
        disagreements.append(
            base.detached_kl_per_sample(
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
    disagreement_matrix = torch.stack(disagreements, dim=1)
    agreement_matrix = torch.stack(agreements, dim=1)
    return disagreement_matrix, agreement_matrix


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
    total_task_gate = {task: 0.0 for task in TASKS}
    total_task_disagreement = {task: 0.0 for task in TASKS}
    total_task_agreement = {task: 0.0 for task in TASKS}

    correct_fused = {task: 0 for task in TASKS}
    correct_tactile = {task: 0 for task in TASKS}
    correct_visual = {task: 0 for task in TASKS}

    supcon_criterion = None
    if args.lambda_supcon > 0.0:
        supcon_criterion = base.SupervisedContrastiveLoss(temperature=args.supcon_temperature)

    iterator = loader
    if base.tqdm is not None:
        desc = "train" if train_mode else ("prefix_eval" if fixed_prefix_ratio is not None else "eval")
        iterator = base.tqdm(loader, leave=False, desc=desc)

    for batch_idx, batch in enumerate(iterator, start=1):
        images = batch["image"].to(device)
        tactile = batch["tactile"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        labels = {task: batch[task].to(device) for task in TASKS}

        prefix_mask = base.effective_padding_mask(
            padding_mask=padding_mask,
            train_mode=train_mode,
            online_train_prob=args.online_train_prob,
            online_min_prefix_ratio=args.online_min_prefix_ratio,
            min_prefix_len=args.min_prefix_len,
            fixed_ratio=fixed_prefix_ratio,
        )

        images, tactile = base.apply_modality_block(images, tactile, args.block_modality)
        if train_mode:
            images, tactile = base.apply_modality_dropout(
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
            disagreement_matrix, agreement_matrix = compute_detached_task_disagreement(
                outputs=outputs,
                temperature=args.disagreement_temperature,
            )
            disagreement_penalty = (outputs["task_gate_scores"] * disagreement_matrix).mean()
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
                disagreement_matrix, agreement_matrix = compute_detached_task_disagreement(
                    outputs=outputs,
                    temperature=args.disagreement_temperature,
                )
                disagreement_penalty = (outputs["task_gate_scores"] * disagreement_matrix).mean()
                loss = clean_loss + args.lambda_disagreement * disagreement_penalty

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_reg += reg_loss.item() * batch_size
        total_supcon += supcon_loss.item() * batch_size
        total_tactile_aux += tactile_aux_loss.item() * batch_size
        total_visual_aux += visual_aux_loss.item() * batch_size
        total_gate += outputs["gate_score"].sum().item()
        total_disagreement_penalty += disagreement_penalty.item() * batch_size
        total_disagreement += disagreement_matrix.mean(dim=1).sum().item()
        total_agreement += agreement_matrix.mean(dim=1).sum().item()
        total_samples += batch_size

        for task in TASKS:
            task_idx = TASK_TO_INDEX[task]
            total_task_gate[task] += outputs["task_gate_scores"][:, task_idx].sum().item()
            total_task_disagreement[task] += disagreement_matrix[:, task_idx].sum().item()
            total_task_agreement[task] += agreement_matrix[:, task_idx].sum().item()
            correct_fused[task] += (outputs[task].argmax(dim=1) == labels[task]).sum().item()
            correct_tactile[task] += (outputs[f"aux_{task}"].argmax(dim=1) == labels[task]).sum().item()
            correct_visual[task] += (outputs[f"vis_{task}"].argmax(dim=1) == labels[task]).sum().item()

        if base.tqdm is not None and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                {
                    "loss": f"{total_loss / max(1, total_samples):.4f}",
                    "mass": f"{correct_fused['mass'] / max(1, total_samples):.2%}",
                    "stiff": f"{correct_fused['stiffness'] / max(1, total_samples):.2%}",
                    "mat": f"{correct_fused['material'] / max(1, total_samples):.2%}",
                    "g": f"{total_gate / max(1, total_samples):.3f}",
                    "gm": f"{total_task_gate['mass'] / max(1, total_samples):.3f}",
                    "gs": f"{total_task_gate['stiffness'] / max(1, total_samples):.3f}",
                    "gmat": f"{total_task_gate['material'] / max(1, total_samples):.3f}",
                    "dis": f"{total_disagreement / max(1, total_samples):.4f}",
                    "gdis": f"{total_disagreement_penalty / max(1, total_samples):.4f}",
                    "step": batch_idx,
                }
            )

    fused_avg = float(np.mean([correct_fused[task] / max(1, total_samples) for task in TASKS]))
    tactile_avg = float(np.mean([correct_tactile[task] / max(1, total_samples) for task in TASKS]))
    visual_avg = float(np.mean([correct_visual[task] / max(1, total_samples) for task in TASKS]))
    metrics = {
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
    for task in TASKS:
        metrics[f"gate_{task}"] = total_task_gate[task] / max(1, total_samples)
        metrics[f"disagreement_{task}"] = total_task_disagreement[task] / max(1, total_samples)
        metrics[f"agreement_{task}"] = total_task_agreement[task] / max(1, total_samples)
    return metrics


def extract_val_disagreement_metrics(val_metrics: Dict[str, float]) -> Dict[str, float]:
    metrics = ORIGINAL_EXTRACT_VAL_DISAGREEMENT_METRICS(val_metrics)
    for task in TASKS:
        metrics[f"gate_{task}"] = val_metrics[f"gate_{task}"]
        metrics[f"disagreement_{task}"] = val_metrics[f"disagreement_{task}"]
        metrics[f"agreement_{task}"] = val_metrics[f"agreement_{task}"]
    return metrics


def eval_split(
    args: argparse.Namespace,
    split_name: str,
    checkpoint_path: Optional[Path] = None,
) -> Dict[str, object]:
    try:
        from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
    except ImportError as exc:  # pragma: no cover
        raise ImportError("eval mode requires scikit-learn") from exc

    data_root = Path(args.data_root)
    split_dir = data_root / split_name
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split not found: {split_dir}")

    ckpt_path = Path(checkpoint_path) if checkpoint_path else Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=base.resolve_device(args.device), weights_only=False)
    cfg = checkpoint.get("config", {})

    device = base.resolve_device(args.device)
    loader = base.build_loader(split_dir, args.batch_size, args.max_tactile_len, args.num_workers, shuffle=False)
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
    task_gate_scores = {task: [] for task in TASKS}

    visual_correct = {task: 0 for task in TASKS}
    tactile_correct = {task: 0 for task in TASKS}
    disagreement_sum = 0.0
    agreement_sum = 0.0
    per_task_disagreement = {task: 0.0 for task in TASKS}
    per_task_agreement = {task: 0.0 for task in TASKS}
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            tactile = batch["tactile"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            prefix_mask = base.effective_padding_mask(
                padding_mask=padding_mask,
                train_mode=False,
                online_train_prob=0.0,
                online_min_prefix_ratio=args.online_min_prefix_ratio,
                min_prefix_len=args.min_prefix_len,
                fixed_ratio=None,
            )
            images, tactile = base.apply_modality_block(images, tactile, args.block_modality)
            outputs = model(images, tactile, padding_mask=prefix_mask)
            disagreement_matrix, agreement_matrix = compute_detached_task_disagreement(
                outputs=outputs,
                temperature=cfg.get("disagreement_temperature", args.disagreement_temperature),
            )
            batch_size = images.size(0)
            total_samples += batch_size
            disagreement_sum += disagreement_matrix.mean(dim=1).sum().item()
            agreement_sum += agreement_matrix.mean(dim=1).sum().item()
            for task in TASKS:
                task_idx = TASK_TO_INDEX[task]
                per_task_disagreement[task] += disagreement_matrix[:, task_idx].sum().item()
                per_task_agreement[task] += agreement_matrix[:, task_idx].sum().item()
                task_gate_scores[task].extend(outputs["task_gate_scores"][:, task_idx].cpu().tolist())
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
        base._plot_confusion_matrix(all_labels[task], all_preds[task], label_names[task], task, output_dir)
    base._plot_summary(results, output_dir)

    visual_summary = {task: float(visual_correct[task] / max(1, total_samples)) for task in TASKS}
    tactile_summary = {task: float(tactile_correct[task] / max(1, total_samples)) for task in TASKS}
    task_gate_average = {
        task: float(np.mean(task_gate_scores[task])) if task_gate_scores[task] else 0.0
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
        "task_gate_average": task_gate_average,
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
            "per_task_average_kl": {
                task: per_task_disagreement[task] / max(1, total_samples) for task in TASKS
            },
            "per_task_prediction_agreement": {
                task: per_task_agreement[task] / max(1, total_samples) for task in TASKS
            },
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

    checkpoint = torch.load(ckpt_path, map_location=base.resolve_device(args.device), weights_only=False)
    cfg = checkpoint.get("config", {})
    device = base.resolve_device(args.device)
    loader = base.build_loader(split_dir, args.batch_size, args.max_tactile_len, args.num_workers, shuffle=False)
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
    for ratio in base.parse_prefix_ratios(args.prefix_ratios):
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
                "task_gate_average": {
                    task: float(metrics[f"gate_{task}"]) for task in TASKS
                },
                "mass": float(metrics["mass"]),
                "stiffness": float(metrics["stiffness"]),
                "material": float(metrics["material"]),
                "average_accuracy": float(np.mean([metrics[task] for task in TASKS])),
                "visual_average_accuracy": float(metrics["visual_average_accuracy"]),
                "tactile_average_accuracy": float(metrics["tactile_average_accuracy"]),
                "disagreement": float(metrics["detached_disagreement"]),
                "gate_weighted_disagreement": float(metrics["gate_weighted_disagreement"]),
                "per_task_disagreement": {
                    task: float(metrics[f"disagreement_{task}"]) for task in TASKS
                },
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


def parse_args() -> argparse.Namespace:
    args = base.parse_args()
    if args.save_dir == "outputs/fusion_gating_online_disagreement":
        args.save_dir = "outputs/fusion_gating_online_disagreement_taskgate"
    return args


base.build_model = build_model
base.compute_metrics = compute_metrics
base.eval_split = eval_split
base.online_eval_split = online_eval_split
base.extract_val_disagreement_metrics = extract_val_disagreement_metrics


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.mode == "train":
        base.train(cli_args)
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
                f"material={point['material']:.2%}, g={point['gate_score']:.3f}, "
                f"gm={point['task_gate_average']['mass']:.3f}, "
                f"gs={point['task_gate_average']['stiffness']:.3f}, "
                f"gmat={point['task_gate_average']['material']:.3f}"
            )
    else:
        if not cli_args.checkpoint:
            raise ValueError("--checkpoint is required in diagnose_visual mode")
        base.diagnose_visual_residual_contribution(cli_args, checkpoint_path=Path(cli_args.checkpoint))
