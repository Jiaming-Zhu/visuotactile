import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_fusion_standard_ablation as base


TASKS = base.TASKS
tqdm = base.tqdm


def compute_ogmge_scales(
    visual_strength: float,
    tactile_strength: float,
    alpha: float = 1.0,
    min_scale: float = 0.05,
    eps: float = 1e-8,
) -> tuple[float, float]:
    visual_value = float(max(visual_strength, eps))
    tactile_value = float(max(tactile_strength, eps))
    if abs(visual_value - tactile_value) <= eps:
        return 1.0, 1.0

    if visual_value > tactile_value:
        ratio = tactile_value / visual_value
        return float(max(min_scale, ratio**alpha)), 1.0

    ratio = visual_value / tactile_value
    return 1.0, float(max(min_scale, ratio**alpha))


def collect_branch_parameter_names(model: nn.Module) -> Dict[str, List[str]]:
    visual_prefixes = (
        "vis_backbone.",
        "vis_proj.",
        "visual_proxy_head_",
    )
    tactile_prefixes = (
        "tac_encoder.",
        "tactile_proxy_head_",
    )

    groups = {"visual": [], "tactile": [], "shared": []}
    for name, _ in model.named_parameters():
        if name.startswith(visual_prefixes):
            groups["visual"].append(name)
        elif name.startswith(tactile_prefixes):
            groups["tactile"].append(name)
        else:
            groups["shared"].append(name)
    return groups


def apply_branch_gradient_modulation(
    named_params: Dict[str, nn.Parameter],
    name_groups: Dict[str, List[str]],
    visual_scale: float,
    tactile_scale: float,
    noise_std: float = 0.0,
) -> None:
    for name in name_groups["visual"]:
        param = named_params[name]
        if param.grad is None:
            continue
        param.grad.mul_(visual_scale)
        if noise_std > 0.0:
            param.grad.add_(torch.randn_like(param.grad) * noise_std)

    for name in name_groups["tactile"]:
        param = named_params[name]
        if param.grad is None:
            continue
        param.grad.mul_(tactile_scale)
        if noise_std > 0.0:
            param.grad.add_(torch.randn_like(param.grad) * noise_std)


class OGMGEFusionModel(base.FusionModel):
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
        proxy_hidden_dim: int = 64,
    ) -> None:
        super().__init__(
            fusion_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=num_layers,
            freeze_visual=freeze_visual,
            mass_classes=mass_classes,
            stiffness_classes=stiffness_classes,
            material_classes=material_classes,
            enable_tactile_aux=False,
            enable_supcon=False,
        )
        self.visual_proxy_head_mass = nn.Sequential(
            nn.Linear(fusion_dim, proxy_hidden_dim),
            nn.GELU(),
            nn.Linear(proxy_hidden_dim, mass_classes),
        )
        self.visual_proxy_head_stiffness = nn.Sequential(
            nn.Linear(fusion_dim, proxy_hidden_dim),
            nn.GELU(),
            nn.Linear(proxy_hidden_dim, stiffness_classes),
        )
        self.visual_proxy_head_material = nn.Sequential(
            nn.Linear(fusion_dim, proxy_hidden_dim),
            nn.GELU(),
            nn.Linear(proxy_hidden_dim, material_classes),
        )
        self.tactile_proxy_head_mass = nn.Sequential(
            nn.Linear(fusion_dim, proxy_hidden_dim),
            nn.GELU(),
            nn.Linear(proxy_hidden_dim, mass_classes),
        )
        self.tactile_proxy_head_stiffness = nn.Sequential(
            nn.Linear(fusion_dim, proxy_hidden_dim),
            nn.GELU(),
            nn.Linear(proxy_hidden_dim, stiffness_classes),
        )
        self.tactile_proxy_head_material = nn.Sequential(
            nn.Linear(fusion_dim, proxy_hidden_dim),
            nn.GELU(),
            nn.Linear(proxy_hidden_dim, material_classes),
        )

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
        v_global = v_tokens.mean(dim=1)
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
        return {
            "mass": self.head_mass(cls_out),
            "stiffness": self.head_stiffness(cls_out),
            "material": self.head_material(cls_out),
            "visual_proxy_mass": self.visual_proxy_head_mass(v_global),
            "visual_proxy_stiffness": self.visual_proxy_head_stiffness(v_global),
            "visual_proxy_material": self.visual_proxy_head_material(v_global),
            "tactile_proxy_mass": self.tactile_proxy_head_mass(t_global),
            "tactile_proxy_stiffness": self.tactile_proxy_head_stiffness(t_global),
            "tactile_proxy_material": self.tactile_proxy_head_material(t_global),
        }


def proxy_task_loss(outputs: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor], criterion: nn.Module, prefix: str) -> torch.Tensor:
    return (
        criterion(outputs[f"{prefix}_mass"], labels["mass"])
        + criterion(outputs[f"{prefix}_stiffness"], labels["stiffness"])
        + criterion(outputs[f"{prefix}_material"], labels["material"])
    )


def compute_metrics(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    train_mode: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None,
    fixed_prefix_ratio: Optional[float] = None,
    branch_name_groups: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, float]:
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_visual_proxy = 0.0
    total_tactile_proxy = 0.0
    total_samples = 0
    total_visual_scale = 0.0
    total_tactile_scale = 0.0
    correct = {task: 0 for task in TASKS}

    iterator = loader
    if tqdm is not None:
        desc = "train" if train_mode else ("prefix_eval" if fixed_prefix_ratio is not None else "eval")
        iterator = tqdm(loader, leave=False, desc=desc)

    for batch_idx, batch in enumerate(iterator, start=1):
        images = batch["image"].to(device, non_blocking=True)
        tactile = batch["tactile"].to(device, non_blocking=True)
        padding_mask = batch["padding_mask"].to(device, non_blocking=True)
        labels = {task: batch[task].to(device, non_blocking=True) for task in TASKS}

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
            if optimizer is None:
                raise ValueError("optimizer is required for train_mode=True")
            if branch_name_groups is None:
                raise ValueError("branch_name_groups are required for train_mode=True")

            optimizer.zero_grad()
            outputs = model(images, tactile, padding_mask=prefix_mask)
            ce_loss = (
                criterion(outputs["mass"], labels["mass"])
                + criterion(outputs["stiffness"], labels["stiffness"])
                + criterion(outputs["material"], labels["material"])
            )
            visual_proxy_loss = proxy_task_loss(outputs, labels, criterion, prefix="visual_proxy")
            tactile_proxy_loss = proxy_task_loss(outputs, labels, criterion, prefix="tactile_proxy")
            loss = ce_loss + args.lambda_proxy * (visual_proxy_loss + tactile_proxy_loss)
            loss.backward()

            visual_strength = 1.0 / max(float(visual_proxy_loss.detach().item()), 1e-8)
            tactile_strength = 1.0 / max(float(tactile_proxy_loss.detach().item()), 1e-8)
            visual_scale, tactile_scale = compute_ogmge_scales(
                visual_strength=visual_strength,
                tactile_strength=tactile_strength,
                alpha=args.ogmge_alpha,
                min_scale=args.ogmge_min_scale,
            )
            apply_branch_gradient_modulation(
                named_params=dict(model.named_parameters()),
                name_groups=branch_name_groups,
                visual_scale=visual_scale,
                tactile_scale=tactile_scale,
                noise_std=args.ogmge_noise_std,
            )
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
                visual_proxy_loss = proxy_task_loss(outputs, labels, criterion, prefix="visual_proxy")
                tactile_proxy_loss = proxy_task_loss(outputs, labels, criterion, prefix="tactile_proxy")
                loss = ce_loss + args.lambda_proxy * (visual_proxy_loss + tactile_proxy_loss)
                visual_strength = 1.0 / max(float(visual_proxy_loss.detach().item()), 1e-8)
                tactile_strength = 1.0 / max(float(tactile_proxy_loss.detach().item()), 1e-8)
                visual_scale, tactile_scale = compute_ogmge_scales(
                    visual_strength=visual_strength,
                    tactile_strength=tactile_strength,
                    alpha=args.ogmge_alpha,
                    min_scale=args.ogmge_min_scale,
                )

        bsz = images.size(0)
        total_loss += loss.item() * bsz
        total_ce += ce_loss.item() * bsz
        total_visual_proxy += visual_proxy_loss.item() * bsz
        total_tactile_proxy += tactile_proxy_loss.item() * bsz
        total_samples += bsz
        total_visual_scale += visual_scale * bsz
        total_tactile_scale += tactile_scale * bsz
        for task in TASKS:
            correct[task] += int((outputs[task].argmax(dim=1) == labels[task]).sum().item())

        if tqdm is not None and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                {
                    "loss": f"{total_loss / max(1, total_samples):.4f}",
                    "mass": f"{correct['mass'] / max(1, total_samples):.2%}",
                    "stiff": f"{correct['stiffness'] / max(1, total_samples):.2%}",
                    "mat": f"{correct['material'] / max(1, total_samples):.2%}",
                    "v_scale": f"{total_visual_scale / max(1, total_samples):.2f}",
                    "t_scale": f"{total_tactile_scale / max(1, total_samples):.2f}",
                    "step": batch_idx,
                }
            )

    result = {
        "loss": total_loss / max(1, total_samples),
        "ce_loss": total_ce / max(1, total_samples),
        "proxy_visual_loss": total_visual_proxy / max(1, total_samples),
        "proxy_tactile_loss": total_tactile_proxy / max(1, total_samples),
        "ogmge_visual_scale": total_visual_scale / max(1, total_samples),
        "ogmge_tactile_scale": total_tactile_scale / max(1, total_samples),
        "mass": correct["mass"] / max(1, total_samples),
        "stiffness": correct["stiffness"] / max(1, total_samples),
        "material": correct["material"] / max(1, total_samples),
    }
    result["average_accuracy"] = float(np.mean([result["mass"], result["stiffness"], result["material"]]))
    return result


def build_model_from_config(cfg: Dict[str, object], dataset: base.RoboticGraspDataset) -> OGMGEFusionModel:
    return OGMGEFusionModel(
        fusion_dim=int(cfg.get("fusion_dim", 256)),
        num_heads=int(cfg.get("num_heads", 8)),
        dropout=float(cfg.get("dropout", 0.1)),
        num_layers=int(cfg.get("num_layers", 4)),
        freeze_visual=True,
        mass_classes=len(dataset.mass_to_idx),
        stiffness_classes=len(dataset.stiffness_to_idx),
        material_classes=len(dataset.material_to_idx),
        proxy_hidden_dim=int(cfg.get("proxy_hidden_dim", 64)),
    )


def build_loader_for_eval(args: argparse.Namespace, split_dir: Path):
    return base.build_loader(
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

    device = base.resolve_device(args.device)
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
        "tasks": {task: {"accuracy": float(base_metrics[task])} for task in TASKS},
        "loss": float(base_metrics["loss"]),
        "ce_loss": float(base_metrics["ce_loss"]),
        "proxy_visual_loss": float(base_metrics["proxy_visual_loss"]),
        "proxy_tactile_loss": float(base_metrics["proxy_tactile_loss"]),
        "ogmge_visual_scale": float(base_metrics["ogmge_visual_scale"]),
        "ogmge_tactile_scale": float(base_metrics["ogmge_tactile_scale"]),
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

    device = base.resolve_device(args.device)
    loader = build_loader_for_eval(args, split_dir)
    dataset = loader.dataset
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})
    model = build_model_from_config(cfg, dataset).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss()

    prefix_results = []
    for prefix_ratio in base.parse_prefix_ratios(args.online_eval_prefixes):
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
    base.set_seed(args.seed)
    device = base.resolve_device(args.device)

    train_loader = base.build_loader(
        split_dir=train_dir,
        batch_size=args.batch_size,
        max_tactile_len=args.max_tactile_len,
        num_workers=args.num_workers,
        shuffle=True,
        augment_policy=args.augment_policy,
    )
    val_loader = base.build_loader(
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

    model = OGMGEFusionModel(
        fusion_dim=args.fusion_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        num_layers=args.num_layers,
        freeze_visual=args.freeze_visual,
        mass_classes=mass_classes,
        stiffness_classes=stiffness_classes,
        material_classes=material_classes,
        proxy_hidden_dim=args.proxy_hidden_dim,
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
    branch_name_groups = collect_branch_parameter_names(model)

    print(
        f"variant={args.variant_name} | lambda_proxy={args.lambda_proxy} | "
        f"ogmge_alpha={args.ogmge_alpha} | ogmge_min_scale={args.ogmge_min_scale} | "
        f"ogmge_noise_std={args.ogmge_noise_std}"
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
            branch_name_groups=branch_name_groups,
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
            f"val_mat={val_metrics['material']:.2%} val_avg={avg_val:.2%} | "
            f"train_v_scale={train_metrics['ogmge_visual_scale']:.2f} "
            f"train_t_scale={train_metrics['ogmge_tactile_scale']:.2f}"
        )

        if avg_val > best_val_acc:
            best_val_acc = avg_val
            best_epoch = epoch
            base.save_checkpoint(
                path=save_dir / "best_model.pth",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                val_metrics=val_metrics,
            )

        if epoch % args.save_every == 0:
            base.save_checkpoint(
                path=save_dir / f"checkpoint_epoch_{epoch}.pth",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                val_metrics=val_metrics,
            )

    (save_dir / "training_history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False))
    base.plot_training_curves(history, save_dir)
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
        description="Fairness-first OGM-GE baseline on top of the current standard fusion backbone."
    )
    parser.add_argument("--mode", choices=["train", "eval", "online_eval"], default="train")
    parser.add_argument("--data_root", type=str, default="/home/jiaming/Y3_Project/Plaintextdataset")
    parser.add_argument("--save_dir", type=str, default="/home/jiaming/Y3_Project/visuotactile/outputs/fusion_standard_ogmge")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--eval_split", choices=["val", "test", "ood_test"], default="test")
    parser.add_argument("--output_dir", type=str, default="")

    parser.add_argument("--variant_name", type=str, default="standard_ogmge")
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
    parser.add_argument("--proxy_hidden_dim", type=int, default=64)
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

    parser.add_argument("--augment_policy", choices=["none", "classical"], default="none")
    parser.add_argument("--lambda_proxy", type=float, default=0.5)
    parser.add_argument("--ogmge_alpha", type=float, default=1.0)
    parser.add_argument("--ogmge_min_scale", type=float, default=0.05)
    parser.add_argument("--ogmge_noise_std", type=float, default=0.0)
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
