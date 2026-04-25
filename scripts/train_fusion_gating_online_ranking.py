import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    from train_fusion_gating_online_reliable import (
        TASKS,
        SupervisedContrastiveLoss,
        build_model,
        build_visual_mismatch_batch,
        compute_clean_losses,
        effective_padding_mask,
        eval_split,
        online_eval_split,
        reliable_selection_start_epoch,
        save_checkpoint,
    )
    from train_fusion_gating2 import (
        LiveTrainingPlotter,
        _plot_training_curves,
        apply_modality_block,
        apply_modality_dropout,
        build_loader,
        resolve_device,
        set_seed,
    )
except ImportError:  # pragma: no cover
    from visuotactile.scripts.train_fusion_gating_online_reliable import (  # type: ignore
        TASKS,
        SupervisedContrastiveLoss,
        build_model,
        build_visual_mismatch_batch,
        compute_clean_losses,
        effective_padding_mask,
        eval_split,
        online_eval_split,
        reliable_selection_start_epoch,
        save_checkpoint,
    )
    from visuotactile.scripts.train_fusion_gating2 import (  # type: ignore
        LiveTrainingPlotter,
        _plot_training_curves,
        apply_modality_block,
        apply_modality_dropout,
        build_loader,
        resolve_device,
        set_seed,
    )


def compute_gate_ranking_loss(
    clean_gate: torch.Tensor,
    mismatch_gate: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    margin_value = float(max(0.0, margin))
    return F.relu(margin_value - (clean_gate - mismatch_gate)).mean()


def compute_gate_margin_success(
    clean_gate: torch.Tensor,
    mismatch_gate: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    margin_value = float(max(0.0, margin))
    return ((clean_gate - mismatch_gate) > margin_value).float().mean()


def compute_gate_order_success(
    clean_gate: torch.Tensor,
    mismatch_gate: torch.Tensor,
) -> torch.Tensor:
    return (clean_gate > mismatch_gate).float().mean()


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
    enable_mismatch_supervision: bool = False,
) -> Dict[str, float]:
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_reg = 0.0
    total_supcon = 0.0
    total_aux = 0.0
    total_gate = 0.0
    total_samples = 0
    correct = {task: 0 for task in TASKS}

    total_mismatch_rank_loss = 0.0
    total_mismatch_gate = 0.0
    total_clean_gate_on_mismatch = 0.0
    total_mismatch_margin_success = 0.0
    total_mismatch_order_success = 0.0
    total_mismatch_samples = 0

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

        mismatch_rank_loss = images.new_zeros(())
        mismatch_gate_score_sum = 0.0
        clean_gate_on_mismatch_sum = 0.0
        mismatch_margin_success_sum = 0.0
        mismatch_order_success_sum = 0.0
        mismatch_sample_count = 0

        if train_mode:
            if optimizer is None:
                raise ValueError("optimizer is required when train_mode=True")
            optimizer.zero_grad()
            outputs = model(images, tactile, padding_mask=prefix_mask)
            clean_loss, reg_loss, supcon_loss, aux_loss = compute_clean_losses(
                outputs=outputs,
                labels=labels,
                criterion=criterion,
                supcon_criterion=supcon_criterion,
                lambda_reg=lambda_reg,
                lambda_aux=args.lambda_aux,
                lambda_supcon=args.lambda_supcon,
                supcon_task=args.supcon_task,
                reg_type=args.reg_type,
                gate_target_mean=args.gate_target_mean,
                gate_entropy_eps=args.gate_entropy_eps,
            )

            total_objective = clean_loss
            if enable_mismatch_supervision and args.lambda_mismatch_rank > 0.0 and args.visual_mismatch_prob > 0.0:
                mismatch_images, mismatch_mask = build_visual_mismatch_batch(
                    images=images,
                    labels=labels,
                    mismatch_prob=args.visual_mismatch_prob,
                )
                if mismatch_mask.any():
                    mismatch_outputs = model(mismatch_images, tactile, padding_mask=prefix_mask)
                    clean_gate = outputs["gate_score"][mismatch_mask]
                    mismatch_gate = mismatch_outputs["gate_score"][mismatch_mask]
                    mismatch_rank_loss = compute_gate_ranking_loss(
                        clean_gate=clean_gate,
                        mismatch_gate=mismatch_gate,
                        margin=args.mismatch_rank_margin,
                    )
                    total_objective = total_objective + args.lambda_mismatch_rank * mismatch_rank_loss
                    mismatch_sample_count = int(mismatch_mask.sum().item())
                    mismatch_gate_score_sum = mismatch_gate.sum().item()
                    clean_gate_on_mismatch_sum = clean_gate.sum().item()
                    mismatch_margin_success_sum = (
                        compute_gate_margin_success(
                            clean_gate=clean_gate,
                            mismatch_gate=mismatch_gate,
                            margin=args.mismatch_rank_margin,
                        ).item()
                        * mismatch_sample_count
                    )
                    mismatch_order_success_sum = (
                        compute_gate_order_success(
                            clean_gate=clean_gate,
                            mismatch_gate=mismatch_gate,
                        ).item()
                        * mismatch_sample_count
                    )

            total_objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss = total_objective
        else:
            with torch.no_grad():
                outputs = model(images, tactile, padding_mask=prefix_mask)
                loss, reg_loss, supcon_loss, aux_loss = compute_clean_losses(
                    outputs=outputs,
                    labels=labels,
                    criterion=criterion,
                    supcon_criterion=supcon_criterion,
                    lambda_reg=lambda_reg,
                    lambda_aux=args.lambda_aux,
                    lambda_supcon=args.lambda_supcon,
                    supcon_task=args.supcon_task,
                    reg_type=args.reg_type,
                    gate_target_mean=args.gate_target_mean,
                    gate_entropy_eps=args.gate_entropy_eps,
                )

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_reg += reg_loss.item() * batch_size
        total_supcon += supcon_loss.item() * batch_size
        total_aux += aux_loss.item() * batch_size
        total_gate += outputs["gate_score"].sum().item()
        total_samples += batch_size

        total_mismatch_rank_loss += mismatch_rank_loss.item() * max(1, mismatch_sample_count)
        total_mismatch_gate += mismatch_gate_score_sum
        total_clean_gate_on_mismatch += clean_gate_on_mismatch_sum
        total_mismatch_margin_success += mismatch_margin_success_sum
        total_mismatch_order_success += mismatch_order_success_sum
        total_mismatch_samples += mismatch_sample_count

        for task in TASKS:
            correct[task] += (outputs[task].argmax(dim=1) == labels[task]).sum().item()

        if tqdm is not None and hasattr(iterator, "set_postfix"):
            postfix = {
                "loss": f"{total_loss / max(1, total_samples):.4f}",
                "mass": f"{correct['mass'] / max(1, total_samples):.2%}",
                "stiff": f"{correct['stiffness'] / max(1, total_samples):.2%}",
                "mat": f"{correct['material'] / max(1, total_samples):.2%}",
                "g": f"{total_gate / max(1, total_samples):.3f}",
                "supcon": f"{total_supcon / max(1, total_samples):.4f}",
                "step": batch_idx,
            }
            if total_mismatch_samples > 0:
                postfix["g_mis"] = f"{total_mismatch_gate / total_mismatch_samples:.3f}"
                postfix["rank"] = f"{total_mismatch_rank_loss / total_mismatch_samples:.4f}"
            iterator.set_postfix(postfix)

    mismatch_gate_mean = total_mismatch_gate / max(1, total_mismatch_samples)
    clean_gate_on_mismatch_mean = total_clean_gate_on_mismatch / max(1, total_mismatch_samples)
    return {
        "loss": total_loss / max(1, total_samples),
        "reg_loss": total_reg / max(1, total_samples),
        "aux_loss": total_aux / max(1, total_samples),
        "supcon_loss": total_supcon / max(1, total_samples),
        "gate_score": total_gate / max(1, total_samples),
        "mass": correct["mass"] / max(1, total_samples),
        "stiffness": correct["stiffness"] / max(1, total_samples),
        "material": correct["material"] / max(1, total_samples),
        "mismatch_rank_loss": total_mismatch_rank_loss / max(1, total_mismatch_samples),
        "mismatch_gate_score": mismatch_gate_mean if total_mismatch_samples > 0 else 0.0,
        "clean_gate_on_mismatch": clean_gate_on_mismatch_mean if total_mismatch_samples > 0 else 0.0,
        "mismatch_gate_gap": (
            clean_gate_on_mismatch_mean - mismatch_gate_mean if total_mismatch_samples > 0 else 0.0
        ),
        "mismatch_margin_success": (
            total_mismatch_margin_success / max(1, total_mismatch_samples) if total_mismatch_samples > 0 else 0.0
        ),
        "mismatch_order_success": (
            total_mismatch_order_success / max(1, total_mismatch_samples) if total_mismatch_samples > 0 else 0.0
        ),
        "mismatch_samples": float(total_mismatch_samples),
    }


def compute_mismatch_ranking_metrics(
    model: nn.Module,
    loader,
    device: torch.device,
    args: argparse.Namespace,
    fixed_prefix_ratio: Optional[float] = None,
) -> Dict[str, float]:
    model.eval()
    total_rank_loss = 0.0
    total_mismatch_gate = 0.0
    total_clean_gate = 0.0
    total_margin_success = 0.0
    total_order_success = 0.0
    total_samples = 0

    iterator = loader
    if tqdm is not None:
        desc = "val_mismatch" if fixed_prefix_ratio is None else "prefix_mismatch"
        iterator = tqdm(loader, leave=False, desc=desc)

    with torch.no_grad():
        for batch in iterator:
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
                fixed_ratio=fixed_prefix_ratio,
            )

            images, tactile = apply_modality_block(images, tactile, args.block_modality)
            mismatch_images, mismatch_mask = build_visual_mismatch_batch(
                images=images,
                labels=labels,
                mismatch_prob=args.visual_mismatch_eval_prob,
            )
            if not mismatch_mask.any():
                continue

            clean_outputs = model(images, tactile, padding_mask=prefix_mask)
            mismatch_outputs = model(mismatch_images, tactile, padding_mask=prefix_mask)
            clean_gate = clean_outputs["gate_score"][mismatch_mask]
            mismatch_gate = mismatch_outputs["gate_score"][mismatch_mask]

            total_rank_loss += compute_gate_ranking_loss(
                clean_gate=clean_gate,
                mismatch_gate=mismatch_gate,
                margin=args.mismatch_rank_margin,
            ).item() * mismatch_gate.numel()
            total_clean_gate += clean_gate.sum().item()
            total_mismatch_gate += mismatch_gate.sum().item()
            total_margin_success += compute_gate_margin_success(
                clean_gate=clean_gate,
                mismatch_gate=mismatch_gate,
                margin=args.mismatch_rank_margin,
            ).item() * mismatch_gate.numel()
            total_order_success += compute_gate_order_success(
                clean_gate=clean_gate,
                mismatch_gate=mismatch_gate,
            ).item() * mismatch_gate.numel()
            total_samples += mismatch_gate.numel()

    mismatch_gate_mean = total_mismatch_gate / max(1, total_samples)
    clean_gate_mean = total_clean_gate / max(1, total_samples)
    return {
        "rank_loss": total_rank_loss / max(1, total_samples),
        "gate_score": mismatch_gate_mean if total_samples > 0 else 0.0,
        "clean_gate_score": clean_gate_mean if total_samples > 0 else 0.0,
        "clean_minus_mismatch_gap": clean_gate_mean - mismatch_gate_mean if total_samples > 0 else 0.0,
        "margin_success": total_margin_success / max(1, total_samples) if total_samples > 0 else 0.0,
        "order_success": total_order_success / max(1, total_samples) if total_samples > 0 else 0.0,
        "num_samples": float(total_samples),
    }


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
        f"ranking supervision: prob={args.visual_mismatch_prob}, "
        f"lambda={args.lambda_mismatch_rank}, margin={args.mismatch_rank_margin}, "
        f"eval_prob={args.visual_mismatch_eval_prob}"
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
            enable_mismatch_supervision=True,
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
        val_mismatch_metrics = compute_mismatch_ranking_metrics(
            model=model,
            loader=val_loader,
            device=device,
            args=args,
        )
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
                val_mismatch_metrics=val_mismatch_metrics,
                avg_val_acc=avg_val,
                lambda_reg_eff=lambda_reg_eff,
                role="best_acc",
            )

        reliable_candidate = epoch >= reliable_epoch_start and val_mismatch_metrics["num_samples"] > 0
        reliable_score = (
            avg_val,
            -val_mismatch_metrics["rank_loss"],
            val_mismatch_metrics["margin_success"],
            val_mismatch_metrics["clean_minus_mismatch_gap"],
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
                val_mismatch_metrics=val_mismatch_metrics,
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
                "val_mismatch": val_mismatch_metrics,
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
            f"val_mis_g={val_mismatch_metrics['gate_score']:.3f} "
            f"gap={val_mismatch_metrics['clean_minus_mismatch_gap']:.3f} "
            f"rank={val_mismatch_metrics['rank_loss']:.4f} "
            f"margin_ok={val_mismatch_metrics['margin_success']:.3f}"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Online-prefix gating fusion with relative gate ranking supervision on visual mismatch"
    )
    parser.add_argument("--mode", choices=["train", "eval", "online_eval"], default="train")
    parser.add_argument("--data_root", type=str, default="/home/martina/Y3_Project/Plaintextdataset")
    parser.add_argument("--save_dir", type=str, default="outputs/fusion_gating_online_ranking")
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
    parser.add_argument("--lambda_supcon", type=float, default=0.0)
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
        "--visual_mismatch_prob",
        type=float,
        default=0.25,
        help="Per-sample probability of creating visual mismatch supervision during training",
    )
    parser.add_argument(
        "--lambda_mismatch_rank",
        type=float,
        default=1.0,
        help="Weight for relative gate ranking loss on mismatched visual inputs",
    )
    parser.add_argument(
        "--mismatch_rank_margin",
        type=float,
        default=0.1,
        help="Required gate gap margin: g_clean - g_mismatch > margin",
    )
    parser.add_argument(
        "--visual_mismatch_eval_prob",
        type=float,
        default=1.0,
        help="Mismatch probability used when measuring validation reliability metrics",
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
    else:
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
