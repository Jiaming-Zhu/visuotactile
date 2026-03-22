import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    from train_tactile import (
        LiveTrainingPlotter,
        TactileOnlyModel,
        _idx_to_names,
        _load_checkpoint,
        _plot_confusion_matrix,
        _plot_summary,
        _plot_training_curves,
        apply_tactile_dropout,
        build_loader,
        resolve_device,
        set_seed,
    )
except ImportError:  # pragma: no cover
    from visuotactile.scripts.train_tactile import (
        LiveTrainingPlotter,
        TactileOnlyModel,
        _idx_to_names,
        _load_checkpoint,
        _plot_confusion_matrix,
        _plot_summary,
        _plot_training_curves,
        apply_tactile_dropout,
        build_loader,
        resolve_device,
        set_seed,
    )


TASKS = ("mass", "stiffness", "material")


def parse_prefix_ratios(raw: str) -> List[float]:
    ratios = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        ratio = float(item)
        if ratio <= 0 or ratio > 1:
            raise ValueError(f"prefix ratio must be in (0, 1], got {ratio}")
        ratios.append(ratio)
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


def compute_task_loss(
    outputs: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    criterion: nn.Module,
) -> torch.Tensor:
    return sum(criterion(outputs[task], labels[task]) for task in TASKS)


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
    total_samples = 0
    correct = {task: 0 for task in TASKS}

    iterator = loader
    if tqdm is not None:
        desc = "train" if train_mode else ("prefix_eval" if fixed_prefix_ratio is not None else "eval")
        iterator = tqdm(loader, leave=False, desc=desc)

    for batch_idx, batch in enumerate(iterator, start=1):
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

        if train_mode:
            tactile = apply_tactile_dropout(tactile, args.tactile_drop_prob)
            optimizer.zero_grad()
            outputs = model(tactile, padding_mask=prefix_mask)
            loss = compute_task_loss(outputs, labels, criterion)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        else:
            with torch.no_grad():
                outputs = model(tactile, padding_mask=prefix_mask)
                loss = compute_task_loss(outputs, labels, criterion)

        batch_size = tactile.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        for task in TASKS:
            correct[task] += (outputs[task].argmax(dim=1) == labels[task]).sum().item()

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

    return {
        "loss": total_loss / max(1, total_samples),
        "mass": correct["mass"] / max(1, total_samples),
        "stiffness": correct["stiffness"] / max(1, total_samples),
        "material": correct["material"] / max(1, total_samples),
    }


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
        f"online training: prob={args.online_train_prob}, "
        f"min_prefix_ratio={args.online_min_prefix_ratio}, min_prefix_len={args.min_prefix_len}"
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
            args=args,
            train_mode=True,
            optimizer=optimizer,
        )
        val_metrics = compute_metrics(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            args=args,
            train_mode=False,
        )
        scheduler.step()

        avg_val = float(np.mean([val_metrics[task] for task in TASKS]))
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
    base_metrics = compute_metrics(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        args=args,
        train_mode=False,
    )

    all_preds = {task: [] for task in TASKS}
    all_labels = {task: [] for task in TASKS}

    model.eval()
    with torch.no_grad():
        for batch in loader:
            tactile = batch["tactile"].to(device, non_blocking=True)
            padding_mask = batch["padding_mask"].to(device, non_blocking=True)
            prefix_mask = effective_padding_mask(
                padding_mask=padding_mask,
                train_mode=False,
                online_train_prob=0.0,
                online_min_prefix_ratio=args.online_min_prefix_ratio,
                min_prefix_len=args.min_prefix_len,
                fixed_ratio=None,
            )
            outputs = model(tactile, padding_mask=prefix_mask)
            for task in TASKS:
                all_preds[task].extend(outputs[task].argmax(dim=1).cpu().tolist())
                all_labels[task].extend(batch[task].tolist())

    label_names = {
        "mass": _idx_to_names(dataset.mass_to_idx),
        "stiffness": _idx_to_names(dataset.stiffness_to_idx),
        "material": _idx_to_names(dataset.material_to_idx),
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
        output_dir = ckpt_path.parent / f"eval_{split_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    for task in TASKS:
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
    loader = build_loader(split_dir, args.batch_size, args.max_tactile_len, args.num_workers, shuffle=False)
    dataset = loader.dataset

    checkpoint = _load_checkpoint(ckpt_path, device)
    cfg = checkpoint.get("config", {})
    ckpt_state = checkpoint["model_state_dict"]

    model = TactileOnlyModel(
        fusion_dim=cfg.get("fusion_dim", args.fusion_dim),
        num_heads=cfg.get("num_heads", args.num_heads),
        dropout=cfg.get("dropout", args.dropout),
        num_layers=cfg.get("num_layers", args.num_layers),
        max_tactile_len=cfg.get("max_tactile_len", args.max_tactile_len),
        mass_classes=int(cfg.get("mass_classes", ckpt_state["head_mass.3.weight"].shape[0])),
        stiffness_classes=int(cfg.get("stiffness_classes", ckpt_state["head_stiffness.3.weight"].shape[0])),
        material_classes=int(cfg.get("material_classes", ckpt_state["head_material.3.weight"].shape[0])),
    ).to(device)
    model.load_state_dict(ckpt_state)

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
            fixed_prefix_ratio=ratio,
        )
        curves.append(
            {
                "prefix_ratio": ratio,
                "loss": float(metrics["loss"]),
                "mass": float(metrics["mass"]),
                "stiffness": float(metrics["stiffness"]),
                "material": float(metrics["material"]),
                "average_accuracy": float(np.mean([metrics[task] for task in TASKS])),
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
    parser = argparse.ArgumentParser(description="Tactile-only training with prefix-aware online masking")
    parser.add_argument("--mode", choices=["train", "eval", "online_eval"], default="train")
    parser.add_argument("--data_root", type=str, default="/home/martina/Y3_Project/Plaintextdataset")
    parser.add_argument("--save_dir", type=str, default="outputs/tactile_model_online")
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
    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--early_stop_acc", type=float, default=1.0)
    parser.add_argument("--early_stop_min_epoch", type=int, default=0)

    parser.add_argument("--online_train_prob", type=float, default=1.0, help="Probability of random prefix training")
    parser.add_argument("--online_min_prefix_ratio", type=float, default=0.2, help="Minimum prefix ratio during training")
    parser.add_argument("--min_prefix_len", type=int, default=64, help="Minimum tactile prefix length in raw timesteps")
    parser.add_argument(
        "--prefix_ratios",
        type=str,
        default="0.1,0.2,0.4,0.6,0.8,1.0",
        help="Comma-separated ratios for online_eval",
    )
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
            f"material={metrics['material']:.2%}, avg_acc={metrics['summary']['average_accuracy']:.2%}"
        )
    else:
        if not cli_args.checkpoint:
            raise ValueError("--checkpoint is required in online_eval mode")
        result = online_eval_split(cli_args, split_name=cli_args.eval_split, checkpoint_path=Path(cli_args.checkpoint))
        for point in result["prefix_curves"]:
            print(
                f"prefix={point['prefix_ratio']:.2f} | loss={point['loss']:.4f} | "
                f"mass={point['mass']:.2%} stiffness={point['stiffness']:.2%} "
                f"material={point['material']:.2%} avg={point['average_accuracy']:.2%}"
            )
