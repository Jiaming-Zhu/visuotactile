import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    from train_fusion_standard_ablation import (
        TASKS,
        apply_modality_block,
        apply_modality_dropout,
        build_loader,
        effective_padding_mask,
        parse_prefix_ratios,
        plot_training_curves,
        resolve_device,
        save_checkpoint,
        set_seed,
    )
except ImportError:  # pragma: no cover
    from visuotactile.scripts.train_fusion_standard_ablation import (  # type: ignore
        TASKS,
        apply_modality_block,
        apply_modality_dropout,
        build_loader,
        effective_padding_mask,
        parse_prefix_ratios,
        plot_training_curves,
        resolve_device,
        save_checkpoint,
        set_seed,
    )


def build_mlp_head(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


def normalize_quality_weights(
    vision_quality: torch.Tensor,
    tactile_quality: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    safe_temperature = max(float(temperature), 1e-6)
    quality_logits = torch.cat([vision_quality, tactile_quality], dim=1) / safe_temperature
    return torch.softmax(quality_logits, dim=1)


class QMFModel(nn.Module):
    def __init__(
        self,
        fusion_dim: int = 256,
        dropout: float = 0.1,
        freeze_visual: bool = True,
        mass_classes: int = 4,
        stiffness_classes: int = 4,
        material_classes: int = 5,
        qmf_hidden_dim: int = 128,
        quality_temperature: float = 1.0,
        use_imagenet_weights: bool = True,
    ) -> None:
        super().__init__()
        self.quality_temperature = float(quality_temperature)

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if use_imagenet_weights else None
        resnet = models.resnet18(weights=weights)
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

        task_dims = {
            "mass": mass_classes,
            "stiffness": stiffness_classes,
            "material": material_classes,
        }
        self.visual_heads = nn.ModuleDict(
            {
                task: build_mlp_head(fusion_dim, qmf_hidden_dim, num_classes, dropout)
                for task, num_classes in task_dims.items()
            }
        )
        self.tactile_heads = nn.ModuleDict(
            {
                task: build_mlp_head(fusion_dim, qmf_hidden_dim, num_classes, dropout)
                for task, num_classes in task_dims.items()
            }
        )
        self.visual_quality_heads = nn.ModuleDict(
            {task: build_mlp_head(fusion_dim, qmf_hidden_dim, 1, dropout) for task in TASKS}
        )
        self.tactile_quality_heads = nn.ModuleDict(
            {task: build_mlp_head(fusion_dim, qmf_hidden_dim, 1, dropout) for task in TASKS}
        )

    def _masked_tactile_global(
        self,
        tactile_tokens: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        num_tactile_tokens = tactile_tokens.shape[1]
        if padding_mask is None:
            return tactile_tokens.mean(dim=1)

        pooled_mask = padding_mask.float().unsqueeze(1)
        pooled_mask = torch.nn.functional.max_pool1d(pooled_mask, kernel_size=2, stride=2)
        pooled_mask = torch.nn.functional.max_pool1d(pooled_mask, kernel_size=2, stride=2)
        pooled_mask = torch.nn.functional.max_pool1d(pooled_mask, kernel_size=2, stride=2)
        pooled_mask = pooled_mask.squeeze(1) > 0.5
        pooled_mask = pooled_mask[:, :num_tactile_tokens]

        valid_mask = (~pooled_mask).unsqueeze(-1).float()
        return (tactile_tokens * valid_mask).sum(dim=1) / (valid_mask.sum(dim=1) + 1e-8)

    def forward(
        self,
        img: torch.Tensor,
        tac: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        visual_features = self.vis_proj(self.vis_backbone(img))
        visual_tokens = visual_features.flatten(2).transpose(1, 2)
        visual_global = visual_tokens.mean(dim=1)

        tactile_features = self.tac_encoder(tac)
        tactile_tokens = tactile_features.transpose(1, 2)
        tactile_global = self._masked_tactile_global(tactile_tokens, padding_mask=padding_mask)

        outputs: Dict[str, torch.Tensor] = {}
        tactile_weight_values = []
        for task in TASKS:
            vision_logits = self.visual_heads[task](visual_global)
            tactile_logits = self.tactile_heads[task](tactile_global)

            weights = normalize_quality_weights(
                vision_quality=self.visual_quality_heads[task](visual_global),
                tactile_quality=self.tactile_quality_heads[task](tactile_global),
                temperature=self.quality_temperature,
            )
            fused_logits = weights[:, :1] * vision_logits + weights[:, 1:] * tactile_logits

            outputs[task] = fused_logits
            outputs[f"vision_{task}"] = vision_logits
            outputs[f"tactile_{task}"] = tactile_logits
            outputs[f"{task}_weights"] = weights
            tactile_weight_values.append(weights[:, 1])

        outputs["average_tactile_weight"] = torch.stack(tactile_weight_values, dim=1).mean(dim=1)
        return outputs


def compute_qmf_losses(
    outputs: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    criterion: nn.Module,
    lambda_unimodal: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fused_ce_loss = sum(criterion(outputs[task], labels[task]) for task in TASKS)
    unimodal_ce_loss = sum(
        criterion(outputs[f"vision_{task}"], labels[task]) + criterion(outputs[f"tactile_{task}"], labels[task])
        for task in TASKS
    )
    total_loss = fused_ce_loss + float(lambda_unimodal) * unimodal_ce_loss
    return total_loss, fused_ce_loss, unimodal_ce_loss


def compute_metrics(
    model: nn.Module,
    loader,
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
    total_fused_ce = 0.0
    total_unimodal_ce = 0.0
    total_tactile_weight = 0.0
    total_samples = 0
    correct = {task: 0 for task in TASKS}
    task_tactile_weight = {task: 0.0 for task in TASKS}

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
            if optimizer is None:
                raise ValueError("optimizer is required for train_mode=True")
            images, tactile = apply_modality_dropout(
                images,
                tactile,
                visual_drop_prob=args.visual_drop_prob,
                tactile_drop_prob=args.tactile_drop_prob,
            )
            optimizer.zero_grad()
            outputs = model(images, tactile, padding_mask=prefix_mask)
            loss, fused_ce_loss, unimodal_ce_loss = compute_qmf_losses(
                outputs=outputs,
                labels=labels,
                criterion=criterion,
                lambda_unimodal=args.lambda_unimodal,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        else:
            with torch.no_grad():
                outputs = model(images, tactile, padding_mask=prefix_mask)
                loss, fused_ce_loss, unimodal_ce_loss = compute_qmf_losses(
                    outputs=outputs,
                    labels=labels,
                    criterion=criterion,
                    lambda_unimodal=args.lambda_unimodal,
                )

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_fused_ce += fused_ce_loss.item() * batch_size
        total_unimodal_ce += unimodal_ce_loss.item() * batch_size
        total_tactile_weight += outputs["average_tactile_weight"].sum().item()
        total_samples += batch_size

        for task in TASKS:
            correct[task] += int((outputs[task].argmax(dim=1) == labels[task]).sum().item())
            task_tactile_weight[task] += outputs[f"{task}_weights"][:, 1].sum().item()

        if tqdm is not None and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                {
                    "loss": f"{total_loss / max(1, total_samples):.4f}",
                    "mass": f"{correct['mass'] / max(1, total_samples):.2%}",
                    "stiff": f"{correct['stiffness'] / max(1, total_samples):.2%}",
                    "mat": f"{correct['material'] / max(1, total_samples):.2%}",
                    "tw": f"{total_tactile_weight / max(1, total_samples):.3f}",
                    "step": batch_idx,
                }
            )

    result = {
        "loss": total_loss / max(1, total_samples),
        "fused_ce_loss": total_fused_ce / max(1, total_samples),
        "unimodal_ce_loss": total_unimodal_ce / max(1, total_samples),
        "average_tactile_weight": total_tactile_weight / max(1, total_samples),
        "mass": correct["mass"] / max(1, total_samples),
        "stiffness": correct["stiffness"] / max(1, total_samples),
        "material": correct["material"] / max(1, total_samples),
    }
    for task in TASKS:
        result[f"{task}_tactile_weight"] = task_tactile_weight[task] / max(1, total_samples)
    result["average_accuracy"] = float(np.mean([result["mass"], result["stiffness"], result["material"]]))
    return result


def build_model_from_config(cfg: Dict[str, object], dataset) -> QMFModel:
    return QMFModel(
        fusion_dim=int(cfg.get("fusion_dim", 256)),
        dropout=float(cfg.get("dropout", 0.1)),
        freeze_visual=bool(cfg.get("freeze_visual", True)),
        mass_classes=len(dataset.mass_to_idx),
        stiffness_classes=len(dataset.stiffness_to_idx),
        material_classes=len(dataset.material_to_idx),
        qmf_hidden_dim=int(cfg.get("qmf_hidden_dim", 128)),
        quality_temperature=float(cfg.get("quality_temperature", 1.0)),
        use_imagenet_weights=bool(cfg.get("use_imagenet_weights", True)),
    )


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

    args.mass_classes = len(train_loader.dataset.mass_to_idx)
    args.stiffness_classes = len(train_loader.dataset.stiffness_to_idx)
    args.material_classes = len(train_loader.dataset.material_to_idx)

    model = QMFModel(
        fusion_dim=args.fusion_dim,
        dropout=args.dropout,
        freeze_visual=args.freeze_visual,
        mass_classes=args.mass_classes,
        stiffness_classes=args.stiffness_classes,
        material_classes=args.material_classes,
        qmf_hidden_dim=args.qmf_hidden_dim,
        quality_temperature=args.quality_temperature,
        use_imagenet_weights=args.use_imagenet_weights,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(epoch_idx: int) -> float:
        if epoch_idx < args.warmup_epochs:
            return (epoch_idx + 1) / max(1, args.warmup_epochs)
        progress = (epoch_idx - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()
    history = []
    best_val_acc = -1.0
    best_epoch = -1

    print(
        f"variant={args.variant_name} | online_train_prob={args.online_train_prob} | "
        f"lambda_unimodal={args.lambda_unimodal} | quality_temperature={args.quality_temperature} | "
        f"augment={args.augment_policy}"
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
            f"val_mat={val_metrics['material']:.2%} val_avg={avg_val:.2%} "
            f"val_tactile_w={val_metrics['average_tactile_weight']:.3f}"
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
                f"mat={metrics['tasks']['material']['accuracy']:.2%} "
                f"tactile_w={metrics['summary']['average_tactile_weight']:.3f}"
            )
            online_eval_split(args, split_name=split_name, checkpoint_path=save_dir / "best_model.pth")


def eval_split(args: argparse.Namespace, split_name: str, checkpoint_path: Optional[Path] = None) -> Dict[str, object]:
    data_root = Path(args.data_root)
    split_dir = data_root / split_name
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split not found: {split_dir}")
    ckpt_path = Path(checkpoint_path) if checkpoint_path else Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = resolve_device(args.device)
    loader = build_loader(
        split_dir=split_dir,
        batch_size=args.batch_size,
        max_tactile_len=args.max_tactile_len,
        num_workers=args.num_workers,
        shuffle=False,
        augment_policy="none",
    )
    dataset = loader.dataset
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})
    model = build_model_from_config(cfg, dataset).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()
    metrics = compute_metrics(
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
        "summary": {
            "average_accuracy": metrics["average_accuracy"],
            "average_tactile_weight": metrics["average_tactile_weight"],
        },
        "tasks": {
            task: {
                "accuracy": float(metrics[task]),
                "average_tactile_weight": float(metrics[f"{task}_tactile_weight"]),
            }
            for task in TASKS
        },
        "loss": float(metrics["loss"]),
        "fused_ce_loss": float(metrics["fused_ce_loss"]),
        "unimodal_ce_loss": float(metrics["unimodal_ce_loss"]),
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
    loader = build_loader(
        split_dir=split_dir,
        batch_size=args.batch_size,
        max_tactile_len=args.max_tactile_len,
        num_workers=args.num_workers,
        shuffle=False,
        augment_policy="none",
    )
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
                "summary": {
                    "average_accuracy": metrics["average_accuracy"],
                    "average_tactile_weight": metrics["average_tactile_weight"],
                },
                "tasks": {
                    task: {
                        "accuracy": float(metrics[task]),
                        "average_tactile_weight": float(metrics[f"{task}_tactile_weight"]),
                    }
                    for task in TASKS
                },
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quality-aware multimodal fusion baseline for visuotactile property estimation"
    )
    parser.add_argument("--mode", choices=["train", "eval", "online_eval"], default="train")
    parser.add_argument("--data_root", type=str, default="/home/martina/Y3_Project/Plaintextdataset")
    parser.add_argument("--save_dir", type=str, default="outputs/fusion_qmf_baseline")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--eval_split", choices=["val", "test", "ood_test"], default="test")
    parser.add_argument("--output_dir", type=str, default="")

    parser.add_argument("--variant_name", type=str, default="qmf_baseline")
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
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--qmf_hidden_dim", type=int, default=128)
    parser.add_argument("--quality_temperature", type=float, default=1.0)

    parser.set_defaults(freeze_visual=True)
    parser.add_argument("--freeze_visual", dest="freeze_visual", action="store_true")
    parser.add_argument("--unfreeze_visual", dest="freeze_visual", action="store_false")
    parser.set_defaults(use_imagenet_weights=True)
    parser.add_argument("--use_imagenet_weights", dest="use_imagenet_weights", action="store_true")
    parser.add_argument("--no_imagenet_weights", dest="use_imagenet_weights", action="store_false")

    parser.add_argument("--block_modality", choices=["none", "visual", "tactile"], default="none")
    parser.add_argument("--visual_drop_prob", type=float, default=0.0)
    parser.add_argument("--tactile_drop_prob", type=float, default=0.0)

    parser.add_argument("--online_train_prob", type=float, default=0.0)
    parser.add_argument("--online_min_prefix_ratio", type=float, default=0.1)
    parser.add_argument("--min_prefix_len", type=int, default=16)
    parser.add_argument("--online_eval_prefixes", type=str, default="0.1,0.2,0.4,0.6,0.8,1.0")

    parser.add_argument("--lambda_unimodal", type=float, default=0.5)
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
