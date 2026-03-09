import argparse
import json
import math
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
    from train_fusion import (
        LiveTrainingPlotter,
        _plot_confusion_matrix,
        _plot_summary,
        _plot_training_curves,
        apply_modality_block,
        apply_modality_dropout,
        build_loader,
        resolve_device,
        set_seed,
    )
except ImportError:  # pragma: no cover
    from visuotactile.scripts.train_fusion import (
        LiveTrainingPlotter,
        _plot_confusion_matrix,
        _plot_summary,
        _plot_training_curves,
        apply_modality_block,
        apply_modality_dropout,
        build_loader,
        resolve_device,
        set_seed,
    )


TASKS = ("mass", "stiffness", "material")


class ChannelLayerNorm1d(nn.Module):
    """LayerNorm over channels only, preserving temporal causality."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class CausalConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.left_padding = kernel_size - 1
        self.stride = stride
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )
        self.norm = ChannelLayerNorm1d(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_padding, 0))
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return self.dropout(x)


def build_resnet18_backbone(pretrained: bool) -> tuple[nn.Sequential, bool]:
    if pretrained:
        try:
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            return nn.Sequential(*list(resnet.children())[:-2]), True
        except Exception as exc:  # pragma: no cover
            print(f"warning: failed to load pretrained ResNet18 weights, falling back to random init: {exc}")
    resnet = models.resnet18(weights=None)
    return nn.Sequential(*list(resnet.children())[:-2]), False


class CausalFusionModel(nn.Module):
    def __init__(
        self,
        fusion_dim: int = 256,
        dropout: float = 0.1,
        num_layers: int = 1,
        freeze_visual: bool = True,
        pretrained_visual: bool = True,
        use_aux_heads: bool = True,
        mass_classes: int = 4,
        stiffness_classes: int = 4,
        material_classes: int = 5,
    ) -> None:
        super().__init__()
        self.fusion_dim = fusion_dim
        self.use_aux_heads = use_aux_heads
        self.stem_strides = (2, 2, 2)

        self.vis_backbone, pretrained_loaded = build_resnet18_backbone(pretrained_visual)
        self.vis_proj = nn.Conv2d(512, fusion_dim, kernel_size=1)
        self.vis_token_norm = nn.LayerNorm(fusion_dim)

        if freeze_visual and not pretrained_loaded:
            print("warning: visual backbone is randomly initialized, disabling freeze_visual for training stability.")
            freeze_visual = False
        if freeze_visual:
            for param in self.vis_backbone.parameters():
                param.requires_grad = False

        self.tac_stem = nn.Sequential(
            CausalConvBlock(24, 64, kernel_size=7, stride=2, dropout=dropout),
            CausalConvBlock(64, 128, kernel_size=5, stride=2, dropout=dropout),
            CausalConvBlock(128, fusion_dim, kernel_size=3, stride=2, dropout=dropout),
        )
        self.tactile_input_norm = nn.LayerNorm(fusion_dim)
        self.tactile_encoder = nn.GRU(
            input_size=fusion_dim,
            hidden_size=fusion_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.tactile_output_norm = nn.LayerNorm(fusion_dim)

        self.query_proj = nn.Linear(fusion_dim, fusion_dim)
        self.key_proj = nn.Linear(fusion_dim, fusion_dim)
        self.value_proj = nn.Linear(fusion_dim, fusion_dim)

        self.gate_mlp = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 1),
            nn.Sigmoid(),
        )
        self.null_visual = nn.Parameter(torch.zeros(1, fusion_dim))
        self.fusion_cell = nn.GRUCell(fusion_dim * 2, fusion_dim)

        self.head_mass = self._make_head(fusion_dim, dropout, mass_classes)
        self.head_stiffness = self._make_head(fusion_dim, dropout, stiffness_classes)
        self.head_material = self._make_head(fusion_dim, dropout, material_classes)

        if use_aux_heads:
            self.aux_head_mass = self._make_head(fusion_dim, dropout, mass_classes)
            self.aux_head_stiffness = self._make_head(fusion_dim, dropout, stiffness_classes)
            self.aux_head_material = self._make_head(fusion_dim, dropout, material_classes)

    @staticmethod
    def _make_head(input_dim: int, dropout: float, output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
        )

    def _downsample_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        downsampled = lengths.clone()
        for stride in self.stem_strides:
            downsampled = (downsampled + stride - 1) // stride
        return downsampled.clamp(min=1)

    @staticmethod
    def _sequence_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        positions = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return positions < lengths.unsqueeze(1)

    @staticmethod
    def _gather_last_valid(sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        indices = (lengths - 1).clamp(min=0)
        batch_index = torch.arange(sequence.size(0), device=sequence.device)
        return sequence[batch_index, indices]

    def encode_visual(self, image: torch.Tensor) -> torch.Tensor:
        visual_feat = self.vis_backbone(image)
        visual_feat = self.vis_proj(visual_feat)
        visual_tokens = visual_feat.flatten(2).transpose(1, 2)
        return self.vis_token_norm(visual_tokens)

    def encode_tactile(
        self,
        tactile: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if padding_mask is None:
            raw_lengths = torch.full(
                (tactile.size(0),),
                tactile.size(-1),
                dtype=torch.long,
                device=tactile.device,
            )
        else:
            raw_lengths = (~padding_mask).sum(dim=1).long().clamp(min=1)

        token_lengths = self._downsample_lengths(raw_lengths)
        tactile_feat = self.tac_stem(tactile)
        tactile_tokens = tactile_feat.transpose(1, 2)
        tactile_tokens = self.tactile_input_norm(tactile_tokens)

        packed = nn.utils.rnn.pack_padded_sequence(
            tactile_tokens,
            token_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.tactile_encoder(packed)
        tactile_states, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=tactile_tokens.size(1),
        )
        tactile_states = self.tactile_output_norm(tactile_states)
        token_mask = self._sequence_mask(token_lengths, tactile_states.size(1))
        return tactile_states, token_lengths, token_mask

    def forward(
        self,
        image: torch.Tensor,
        tactile: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        visual_tokens = self.encode_visual(image)
        tactile_states, token_lengths, token_mask = self.encode_tactile(tactile, padding_mask)

        queries = self.query_proj(tactile_states)
        keys = self.key_proj(visual_tokens)
        values = self.value_proj(visual_tokens)

        attn_scores = torch.matmul(queries, keys.transpose(1, 2)) / math.sqrt(self.fusion_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        visual_context = torch.matmul(attn_weights, values)

        gate_input = torch.cat([visual_context, tactile_states], dim=-1)
        gate_values = self.gate_mlp(gate_input)
        gated_visual = gate_values * visual_context + (1.0 - gate_values) * self.null_visual.view(1, 1, -1)

        hidden = tactile_states.new_zeros(tactile_states.size(0), self.fusion_dim)
        fused_states: List[torch.Tensor] = []
        for step_idx in range(tactile_states.size(1)):
            fusion_input = torch.cat([gated_visual[:, step_idx], tactile_states[:, step_idx]], dim=-1)
            next_hidden = self.fusion_cell(fusion_input, hidden)
            valid_step = token_mask[:, step_idx].unsqueeze(-1)
            hidden = torch.where(valid_step, next_hidden, hidden)
            fused_states.append(hidden)
        fused_states = torch.stack(fused_states, dim=1)

        outputs = {
            "mass_seq": self.head_mass(fused_states),
            "stiffness_seq": self.head_stiffness(fused_states),
            "material_seq": self.head_material(fused_states),
            "gate_values": gate_values.squeeze(-1),
            "token_lengths": token_lengths,
            "token_mask": token_mask,
        }
        for task in TASKS:
            outputs[task] = self._gather_last_valid(outputs[f"{task}_seq"], token_lengths)

        if self.use_aux_heads:
            outputs["aux_mass_seq"] = self.aux_head_mass(tactile_states)
            outputs["aux_stiffness_seq"] = self.aux_head_stiffness(tactile_states)
            outputs["aux_material_seq"] = self.aux_head_material(tactile_states)
            for task in TASKS:
                outputs[f"aux_{task}"] = self._gather_last_valid(outputs[f"aux_{task}_seq"], token_lengths)

        return outputs


def masked_sequence_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    expanded_labels = labels.unsqueeze(1).expand(-1, logits.size(1))
    loss = F.cross_entropy(logits.transpose(1, 2), expanded_labels, reduction="none")
    weights = token_mask.float()
    per_sample = (loss * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
    return per_sample.mean()


def masked_scalar_mean(values: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    weights = token_mask.float()
    per_sample = (values * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
    return per_sample.mean()


def gate_smoothness_loss(gate_values: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    if gate_values.size(1) < 2:
        return gate_values.new_zeros(())
    diffs = (gate_values[:, 1:] - gate_values[:, :-1]).pow(2)
    smooth_mask = token_mask[:, 1:] & token_mask[:, :-1]
    if not smooth_mask.any():
        return gate_values.new_zeros(())
    return masked_scalar_mean(diffs, smooth_mask)


def compute_total_loss(
    outputs: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, Dict[str, float]]:
    token_mask = outputs["token_mask"]
    total_loss = outputs["mass"].new_zeros(())
    breakdown: Dict[str, float] = {}

    for task in TASKS:
        seq_loss = masked_sequence_cross_entropy(outputs[f"{task}_seq"], labels[task], token_mask)
        final_loss = F.cross_entropy(outputs[task], labels[task])
        task_loss = seq_loss + args.final_step_weight * final_loss
        total_loss = total_loss + task_loss
        breakdown[f"{task}_seq_loss"] = float(seq_loss.detach())
        breakdown[f"{task}_final_loss"] = float(final_loss.detach())

        aux_seq_key = f"aux_{task}_seq"
        aux_final_key = f"aux_{task}"
        if args.aux_loss_weight > 0.0 and aux_seq_key in outputs and aux_final_key in outputs:
            aux_seq_loss = masked_sequence_cross_entropy(outputs[aux_seq_key], labels[task], token_mask)
            aux_final_loss = F.cross_entropy(outputs[aux_final_key], labels[task])
            aux_loss = aux_seq_loss + args.final_step_weight * aux_final_loss
            total_loss = total_loss + args.aux_loss_weight * aux_loss
            breakdown[f"{task}_aux_seq_loss"] = float(aux_seq_loss.detach())
            breakdown[f"{task}_aux_final_loss"] = float(aux_final_loss.detach())

    gate_values = outputs["gate_values"]
    gate_mean = masked_scalar_mean(gate_values, token_mask)
    breakdown["gate_mean"] = float(gate_mean.detach())

    if args.gate_sparse_weight > 0.0:
        sparse_loss = gate_mean
        total_loss = total_loss + args.gate_sparse_weight * sparse_loss
        breakdown["gate_sparse_loss"] = float(sparse_loss.detach())

    if args.gate_polar_weight > 0.0:
        polar_loss = masked_scalar_mean(gate_values * (1.0 - gate_values), token_mask)
        total_loss = total_loss + args.gate_polar_weight * polar_loss
        breakdown["gate_polar_loss"] = float(polar_loss.detach())

    if args.gate_mean_weight > 0.0:
        mean_loss = (gate_mean - args.gate_target_mean).pow(2)
        total_loss = total_loss + args.gate_mean_weight * mean_loss
        breakdown["gate_mean_reg_loss"] = float(mean_loss.detach())

    if args.gate_smooth_weight > 0.0:
        smooth_loss = gate_smoothness_loss(gate_values, token_mask)
        total_loss = total_loss + args.gate_smooth_weight * smooth_loss
        breakdown["gate_smooth_loss"] = float(smooth_loss.detach())

    breakdown["loss"] = float(total_loss.detach())
    return total_loss, breakdown


def compute_metrics(
    model: nn.Module,
    loader,
    device: torch.device,
    args: argparse.Namespace,
    train_mode: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, float]:
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0
    total_gate_mean = 0.0
    correct = {task: 0 for task in TASKS}

    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, leave=False, desc="train" if train_mode else "eval")

    for batch_idx, batch in enumerate(iterator, start=1):
        images = batch["image"].to(device)
        tactile = batch["tactile"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        labels = {task: batch[task].to(device) for task in TASKS}

        images, tactile = apply_modality_block(images, tactile, args.block_modality)
        if train_mode:
            images, tactile = apply_modality_dropout(
                images,
                tactile,
                visual_drop_prob=args.visual_drop_prob,
                tactile_drop_prob=args.tactile_drop_prob,
            )
            optimizer.zero_grad()
            outputs = model(images, tactile, padding_mask=padding_mask)
            loss, loss_info = compute_total_loss(outputs, labels, args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            optimizer.step()
        else:
            with torch.no_grad():
                outputs = model(images, tactile, padding_mask=padding_mask)
                loss, loss_info = compute_total_loss(outputs, labels, args)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        total_gate_mean += loss_info["gate_mean"] * batch_size

        for task in TASKS:
            correct[task] += (outputs[task].argmax(dim=1) == labels[task]).sum().item()

        if tqdm is not None and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                {
                    "loss": f"{total_loss / max(1, total_samples):.4f}",
                    "mass": f"{correct['mass'] / max(1, total_samples):.2%}",
                    "stiff": f"{correct['stiffness'] / max(1, total_samples):.2%}",
                    "mat": f"{correct['material'] / max(1, total_samples):.2%}",
                    "gate": f"{total_gate_mean / max(1, total_samples):.3f}",
                    "step": batch_idx,
                }
            )

    return {
        "loss": total_loss / max(1, total_samples),
        "mass": correct["mass"] / max(1, total_samples),
        "stiffness": correct["stiffness"] / max(1, total_samples),
        "material": correct["material"] / max(1, total_samples),
        "gate_mean": total_gate_mean / max(1, total_samples),
    }


def build_model_from_args(args: argparse.Namespace, **override_kwargs) -> CausalFusionModel:
    model_kwargs = {
        "fusion_dim": override_kwargs.get("fusion_dim", args.fusion_dim),
        "dropout": override_kwargs.get("dropout", args.dropout),
        "num_layers": override_kwargs.get("num_layers", args.num_layers),
        "freeze_visual": override_kwargs.get("freeze_visual", args.freeze_visual),
        "pretrained_visual": override_kwargs.get("pretrained_visual", args.pretrained_visual),
        "use_aux_heads": override_kwargs.get("use_aux_heads", args.use_aux_heads),
        "mass_classes": override_kwargs["mass_classes"],
        "stiffness_classes": override_kwargs["stiffness_classes"],
        "material_classes": override_kwargs["material_classes"],
    }
    return CausalFusionModel(**model_kwargs)


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
    print(f"block_modality: {args.block_modality}")
    print(
        "loss config: "
        f"final_step_weight={args.final_step_weight}, "
        f"aux_loss_weight={args.aux_loss_weight}, "
        f"gate_sparse_weight={args.gate_sparse_weight}, "
        f"gate_polar_weight={args.gate_polar_weight}, "
        f"gate_mean_weight={args.gate_mean_weight}, "
        f"gate_smooth_weight={args.gate_smooth_weight}"
    )
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

    model = build_model_from_args(
        args,
        mass_classes=mass_classes,
        stiffness_classes=stiffness_classes,
        material_classes=material_classes,
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
            device=device,
            args=args,
            train_mode=True,
            optimizer=optimizer,
        )
        val_metrics = compute_metrics(
            model=model,
            loader=val_loader,
            device=device,
            args=args,
            train_mode=False,
        )
        scheduler.step()

        avg_val = float(np.mean([val_metrics[task] for task in TASKS]))
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "avg_val_acc": avg_val,
            }
        )
        if live_plotter is not None:
            live_plotter.update(epoch, train_metrics, val_metrics)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} | "
            f"val_mass={val_metrics['mass']:.2%} val_stiff={val_metrics['stiffness']:.2%} "
            f"val_mat={val_metrics['material']:.2%} gate_mean={val_metrics['gate_mean']:.3f}"
        )

        if avg_val > best_val_acc:
            best_val_acc = avg_val
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": vars(args),
                    "val_metrics": val_metrics,
                },
                save_dir / "best_model.pth",
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
    print(f"best epoch: {best_epoch} | best val avg acc: {best_val_acc:.2%}")

    for split_name in ["test", "ood_test"]:
        split_dir = data_root / split_name
        if split_dir.is_dir():
            metrics = eval_split(args, split_name=split_name, checkpoint_path=save_dir / "best_model.pth")
            print(
                f"{split_name}: loss={metrics['loss']:.4f}, "
                f"mass={metrics['mass']:.2%}, stiffness={metrics['stiffness']:.2%}, "
                f"material={metrics['material']:.2%}, gate_mean={metrics['gate_mean']:.3f}"
            )


def eval_split(
    args: argparse.Namespace,
    split_name: str,
    checkpoint_path: Optional[Path] = None,
) -> Dict[str, float]:
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

    device = resolve_device(args.device)
    loader = build_loader(split_dir, args.batch_size, args.max_tactile_len, args.num_workers, shuffle=False)
    dataset = loader.dataset

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
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
            f"Checkpoint expects {mass_classes} mass classes, but dataset defines {dataset_mass_classes}."
        )
    if stiffness_classes != dataset_stiffness_classes:
        raise ValueError(
            f"Checkpoint expects {stiffness_classes} stiffness classes, but dataset defines {dataset_stiffness_classes}."
        )
    if material_classes != dataset_material_classes:
        raise ValueError(
            f"Checkpoint expects {material_classes} material classes, but dataset defines {dataset_material_classes}."
        )

    model = build_model_from_args(
        args,
        fusion_dim=cfg.get("fusion_dim", args.fusion_dim),
        dropout=cfg.get("dropout", args.dropout),
        num_layers=cfg.get("num_layers", args.num_layers),
        freeze_visual=True,
        pretrained_visual=False,
        use_aux_heads=cfg.get("use_aux_heads", args.use_aux_heads),
        mass_classes=mass_classes,
        stiffness_classes=stiffness_classes,
        material_classes=material_classes,
    ).to(device)
    model.load_state_dict(ckpt_state)

    eval_args = argparse.Namespace(**vars(args))
    eval_args.final_step_weight = cfg.get("final_step_weight", args.final_step_weight)
    eval_args.aux_loss_weight = cfg.get("aux_loss_weight", args.aux_loss_weight)
    eval_args.gate_sparse_weight = cfg.get("gate_sparse_weight", args.gate_sparse_weight)
    eval_args.gate_polar_weight = cfg.get("gate_polar_weight", args.gate_polar_weight)
    eval_args.gate_mean_weight = cfg.get("gate_mean_weight", args.gate_mean_weight)
    eval_args.gate_target_mean = cfg.get("gate_target_mean", args.gate_target_mean)
    eval_args.gate_smooth_weight = cfg.get("gate_smooth_weight", args.gate_smooth_weight)

    base_metrics = compute_metrics(
        model=model,
        loader=loader,
        device=device,
        args=eval_args,
        train_mode=False,
    )

    all_preds = {task: [] for task in TASKS}
    all_labels = {task: [] for task in TASKS}

    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            tactile = batch["tactile"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            images, tactile = apply_modality_block(images, tactile, args.block_modality)
            outputs = model(images, tactile, padding_mask=padding_mask)
            for task in TASKS:
                all_preds[task].extend(outputs[task].argmax(dim=1).cpu().tolist())
                all_labels[task].extend(batch[task].tolist())

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
            labels,
            preds,
            labels=all_class_labels,
            average=None,
            zero_division=0,
        )
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            labels,
            preds,
            labels=all_class_labels,
            average="macro",
            zero_division=0,
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            labels,
            preds,
            labels=all_class_labels,
            average="weighted",
            zero_division=0,
        )
        report_text = classification_report(
            labels,
            preds,
            labels=all_class_labels,
            target_names=names,
            digits=4,
            zero_division=0,
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
        "gate_mean": float(base_metrics["gate_mean"]),
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
    parser = argparse.ArgumentParser(description="Training script for the causal visuotactile fusion model")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--data_root", type=str, default="/home/martina/Y3_Project/Plaintextdataset")
    parser.add_argument("--save_dir", type=str, default="outputs/causal_model_clean")
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
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--fusion_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_layers", type=int, default=1, help="Number of causal tactile GRU layers")
    parser.set_defaults(freeze_visual=True)
    parser.add_argument("--freeze_visual", dest="freeze_visual", action="store_true")
    parser.add_argument("--unfreeze_visual", dest="freeze_visual", action="store_false")
    parser.set_defaults(pretrained_visual=True)
    parser.add_argument("--pretrained_visual", dest="pretrained_visual", action="store_true")
    parser.add_argument("--no_pretrained_visual", dest="pretrained_visual", action="store_false")
    parser.set_defaults(use_aux_heads=True)
    parser.add_argument("--use_aux_heads", dest="use_aux_heads", action="store_true")
    parser.add_argument("--no_aux_heads", dest="use_aux_heads", action="store_false")

    parser.add_argument("--visual_drop_prob", type=float, default=0.0)
    parser.add_argument("--tactile_drop_prob", type=float, default=0.0)
    parser.add_argument("--final_step_weight", type=float, default=1.0)
    parser.add_argument("--aux_loss_weight", type=float, default=0.2)
    parser.add_argument("--gate_sparse_weight", type=float, default=0.0)
    parser.add_argument("--gate_polar_weight", type=float, default=0.0)
    parser.add_argument("--gate_mean_weight", type=float, default=0.0)
    parser.add_argument("--gate_target_mean", type=float, default=0.5)
    parser.add_argument("--gate_smooth_weight", type=float, default=0.0)
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help="Disable if 0; stop if val avg acc reaches threshold for N epochs",
    )
    parser.add_argument("--early_stop_acc", type=float, default=1.0, help="Validation avg acc threshold for early stop")
    parser.add_argument("--early_stop_min_epoch", type=int, default=0, help="Do not early stop before this epoch")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.mode == "train":
        train(cli_args)
    else:
        if not cli_args.checkpoint:
            raise ValueError("--checkpoint is required in eval mode")
        metrics = eval_split(cli_args, split_name=cli_args.eval_split, checkpoint_path=Path(cli_args.checkpoint))
        print(
            f"{cli_args.eval_split}: loss={metrics['loss']:.4f}, "
            f"mass={metrics['mass']:.2%}, stiffness={metrics['stiffness']:.2%}, "
            f"material={metrics['material']:.2%}, gate_mean={metrics['gate_mean']:.3f}, "
            f"avg_acc={metrics['summary']['average_accuracy']:.2%}"
        )
