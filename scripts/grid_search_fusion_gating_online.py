import argparse
import csv
import itertools
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_SCORE_WEIGHTS = {
    "test": 0.10,
    "ood": 0.45,
    "p0.2": 0.15,
    "p0.4": 0.30,
}

FAST_SCORE_WEIGHTS = {
    "ood": 0.50,
    "p0.2": 0.20,
    "p0.4": 0.30,
}


def parse_float_list(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_score_weights(raw: str) -> Dict[str, float]:
    weights = dict(DEFAULT_SCORE_WEIGHTS)
    if not raw.strip():
        return weights
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        key, value = item.split("=", 1)
        weights[key.strip()] = float(value.strip())
    return weights


def slugify_value(value) -> str:
    if isinstance(value, float):
        text = f"{value:.4f}".rstrip("0").rstrip(".")
    else:
        text = str(value)
    return text.replace("-", "m").replace(".", "p")


def combo_slug(combo: Dict[str, object]) -> str:
    parts = []
    for key in sorted(combo):
        short = {
            "online_train_prob": "otp",
            "online_min_prefix_ratio": "mpr",
            "min_prefix_len": "mpl",
            "lambda_reg": "lreg",
            "visual_drop_prob": "vdrop",
        }.get(key, key)
        parts.append(f"{short}{slugify_value(combo[key])}")
    return "_".join(parts)


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(cmd: List[str], dry_run: bool = False) -> None:
    printable = " ".join(cmd)
    print(f"$ {printable}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def find_candidate_checkpoints(trial_dir: Path, candidate_epochs: List[int]) -> List[Tuple[str, Path]]:
    candidates: List[Tuple[str, Path]] = []
    best_model = trial_dir / "best_model.pth"
    if best_model.exists():
        candidates.append(("best_model", best_model))

    if candidate_epochs:
        for epoch in candidate_epochs:
            ckpt = trial_dir / f"checkpoint_epoch_{epoch}.pth"
            if ckpt.exists():
                candidates.append((f"checkpoint_epoch_{epoch}", ckpt))
    else:
        for ckpt in sorted(trial_dir.glob("checkpoint_epoch_*.pth")):
            candidates.append((ckpt.stem, ckpt))

    deduped: List[Tuple[str, Path]] = []
    seen = set()
    for label, path in candidates:
        if path in seen:
            continue
        deduped.append((label, path))
        seen.add(path)
    return deduped


def build_train_command(args: argparse.Namespace, trial_dir: Path, combo: Dict[str, object]) -> List[str]:
    cmd = [
        sys.executable,
        str(args.train_script),
        "--mode",
        "train",
        "--data_root",
        args.data_root,
        "--save_dir",
        str(trial_dir),
        "--device",
        args.device,
        "--batch_size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--warmup_epochs",
        str(args.warmup_epochs),
        "--save_every",
        str(args.save_every),
        "--num_workers",
        str(args.num_workers),
        "--max_tactile_len",
        str(args.max_tactile_len),
        "--seed",
        str(args.seed),
        "--fusion_dim",
        str(args.fusion_dim),
        "--num_heads",
        str(args.num_heads),
        "--dropout",
        str(args.dropout),
        "--num_layers",
        str(args.num_layers),
        "--lambda_aux",
        str(args.lambda_aux),
        "--reg_type",
        args.reg_type,
        "--gate_target_mean",
        str(args.gate_target_mean),
        "--gate_entropy_eps",
        str(args.gate_entropy_eps),
        "--gate_reg_warmup_epochs",
        str(args.gate_reg_warmup_epochs),
        "--gate_reg_ramp_epochs",
        str(args.gate_reg_ramp_epochs),
        "--early_stop_patience",
        str(args.early_stop_patience),
        "--early_stop_acc",
        str(args.early_stop_acc),
        "--early_stop_min_epoch",
        str(args.early_stop_min_epoch),
        "--prefix_ratios",
        args.online_eval_ratios,
        "--online_train_prob",
        str(combo["online_train_prob"]),
        "--online_min_prefix_ratio",
        str(combo["online_min_prefix_ratio"]),
        "--min_prefix_len",
        str(combo["min_prefix_len"]),
        "--lambda_reg",
        str(combo["lambda_reg"]),
        "--visual_drop_prob",
        str(combo["visual_drop_prob"]),
        "--tactile_drop_prob",
        str(args.tactile_drop_prob),
    ]
    if args.freeze_visual:
        cmd.append("--freeze_visual")
    else:
        cmd.append("--unfreeze_visual")
    if args.live_plot:
        cmd.append("--live_plot")
    else:
        cmd.append("--no_live_plot")
    if args.block_modality != "none":
        cmd.extend(["--block_modality", args.block_modality])
    return cmd


def build_eval_command(
    args: argparse.Namespace,
    checkpoint_path: Path,
    split_name: str,
    output_dir: Path,
    block_modality: str = "none",
    mode: str = "eval",
) -> List[str]:
    cmd = [
        sys.executable,
        str(args.train_script),
        "--mode",
        mode,
        "--data_root",
        args.data_root,
        "--checkpoint",
        str(checkpoint_path),
        "--eval_split",
        split_name,
        "--device",
        args.device,
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--max_tactile_len",
        str(args.max_tactile_len),
        "--output_dir",
        str(output_dir),
        "--block_modality",
        block_modality,
    ]
    if mode == "online_eval":
        cmd.extend(["--prefix_ratios", args.online_eval_ratios])
    return cmd


def ensure_trial_evaluated(
    args: argparse.Namespace,
    checkpoint_label: str,
    checkpoint_path: Path,
    trial_dir: Path,
    dry_run: bool = False,
) -> Dict[str, object]:
    eval_root = trial_dir / "grid_eval" / checkpoint_label
    test_out = eval_root / "eval_test"
    ood_out = eval_root / "eval_ood_test"
    online_out = eval_root / "online_eval_ood_test"

    test_json = test_out / "evaluation_results.json"
    ood_json = ood_out / "evaluation_results.json"
    online_json = online_out / "online_evaluation_results.json"

    if not args.skip_test_eval:
        if not test_json.exists() or not args.resume:
            run_command(
                build_eval_command(args, checkpoint_path, "test", test_out, mode="eval"),
                dry_run=dry_run,
            )
    if not ood_json.exists() or not args.resume:
        run_command(
            build_eval_command(args, checkpoint_path, "ood_test", ood_out, mode="eval"),
            dry_run=dry_run,
        )
    if not online_json.exists() or not args.resume:
        run_command(
            build_eval_command(args, checkpoint_path, "ood_test", online_out, mode="online_eval"),
            dry_run=dry_run,
        )

    if dry_run:
        return {}

    test_data = load_json(test_json) if (not args.skip_test_eval and test_json.exists()) else None
    ood_data = load_json(ood_json)
    online_data = load_json(online_json)

    curve_map = {float(item["prefix_ratio"]): item for item in online_data.get("prefix_curves", [])}
    result = {
        "checkpoint_label": checkpoint_label,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": (test_data or ood_data).get("checkpoint_epoch"),
        "test_avg": float(test_data["summary"]["average_accuracy"]) if test_data else 0.0,
        "ood_avg": float(ood_data["summary"]["average_accuracy"]),
        "test_gate": float(test_data.get("avg_gate_score", 0.0)) if test_data else 0.0,
        "ood_gate": float(ood_data.get("avg_gate_score", 0.0)),
        "prefix_metrics": {
            f"{ratio:.1f}": {
                "average_accuracy": float(curve_map.get(ratio, {}).get("average_accuracy", 0.0)),
                "gate_score": float(curve_map.get(ratio, {}).get("gate_score", 0.0)),
            }
            for ratio in parse_float_list(args.online_eval_ratios)
        },
    }
    result["score"] = compute_score(result, args.score_weights)
    return result


def compute_score(metrics: Dict[str, object], weights: Dict[str, float]) -> float:
    score = 0.0
    for key, weight in weights.items():
        if key == "test":
            score += weight * float(metrics["test_avg"])
        elif key == "ood":
            score += weight * float(metrics["ood_avg"])
        elif key.startswith("p"):
            ratio = key[1:]
            value = float(metrics["prefix_metrics"].get(ratio, {}).get("average_accuracy", 0.0))
            score += weight * value
    return float(score)


def summarize_trial(
    args: argparse.Namespace,
    trial_index: int,
    combo: Dict[str, object],
    trial_dir: Path,
    dry_run: bool = False,
) -> Dict[str, object]:
    trial_summary_path = trial_dir / "grid_search_trial_summary.json"
    if args.resume and trial_summary_path.exists() and not dry_run:
        return load_json(trial_summary_path)

    best_model = trial_dir / "best_model.pth"
    if not best_model.exists() or not args.resume:
        run_command(build_train_command(args, trial_dir, combo), dry_run=dry_run)

    if dry_run:
        return {
            "trial_index": trial_index,
            "trial_dir": str(trial_dir),
            "combo": combo,
            "score": None,
            "best_checkpoint": None,
            "candidates": [],
        }

    candidates = find_candidate_checkpoints(trial_dir, args.candidate_epochs)
    if not candidates:
        raise FileNotFoundError(f"No candidate checkpoints found in {trial_dir}")

    candidate_results = []
    for label, ckpt_path in candidates:
        print(f"[trial {trial_index:03d}] evaluating {label}")
        candidate_results.append(
            ensure_trial_evaluated(
                args=args,
                checkpoint_label=label,
                checkpoint_path=ckpt_path,
                trial_dir=trial_dir,
                dry_run=dry_run,
            )
        )

    best_candidate = max(candidate_results, key=lambda item: item["score"])
    summary = {
        "trial_index": trial_index,
        "trial_dir": str(trial_dir),
        "combo": combo,
        "score": float(best_candidate["score"]),
        "best_checkpoint": best_candidate,
        "candidates": candidate_results,
    }
    trial_summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def build_grid(args: argparse.Namespace) -> List[Dict[str, object]]:
    keys = [
        "online_train_prob",
        "online_min_prefix_ratio",
        "min_prefix_len",
        "lambda_reg",
        "visual_drop_prob",
    ]
    values = [
        parse_float_list(args.online_train_prob_grid),
        parse_float_list(args.online_min_prefix_ratio_grid),
        parse_int_list(args.min_prefix_len_grid),
        parse_float_list(args.lambda_reg_grid),
        parse_float_list(args.visual_drop_prob_grid),
    ]
    combos = []
    for combo_values in itertools.product(*values):
        combo = dict(zip(keys, combo_values))
        combos.append(combo)
    return combos


def write_summary_files(output_base: Path, results: List[Dict[str, object]]) -> None:
    summary_json = output_base / "grid_search_summary.json"
    summary_csv = output_base / "grid_search_summary.csv"
    best_json = output_base / "grid_search_best.json"

    summary_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = [
        "rank",
        "trial_index",
        "score",
        "trial_dir",
        "checkpoint_label",
        "checkpoint_epoch",
        "test_avg",
        "ood_avg",
        "test_gate",
        "ood_gate",
        "p0.1",
        "p0.2",
        "p0.4",
        "p0.6",
        "p1.0",
        "online_train_prob",
        "online_min_prefix_ratio",
        "min_prefix_len",
        "lambda_reg",
        "visual_drop_prob",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, item in enumerate(results, start=1):
            best = item["best_checkpoint"]
            row = {
                "rank": rank,
                "trial_index": item["trial_index"],
                "score": item["score"],
                "trial_dir": item["trial_dir"],
                "checkpoint_label": best["checkpoint_label"],
                "checkpoint_epoch": best["checkpoint_epoch"],
                "test_avg": best["test_avg"],
                "ood_avg": best["ood_avg"],
                "test_gate": best["test_gate"],
                "ood_gate": best["ood_gate"],
                "p0.1": best["prefix_metrics"].get("0.1", {}).get("average_accuracy", 0.0),
                "p0.2": best["prefix_metrics"].get("0.2", {}).get("average_accuracy", 0.0),
                "p0.4": best["prefix_metrics"].get("0.4", {}).get("average_accuracy", 0.0),
                "p0.6": best["prefix_metrics"].get("0.6", {}).get("average_accuracy", 0.0),
                "p1.0": best["prefix_metrics"].get("1.0", {}).get("average_accuracy", 0.0),
                **item["combo"],
            }
            writer.writerow(row)

    if results:
        best_json.write_text(json.dumps(results[0], indent=2, ensure_ascii=False), encoding="utf-8")


def print_top_results(results: List[Dict[str, object]], top_k: int) -> None:
    print("\n===============================================================")
    print("Top Grid Search Results")
    print("===============================================================")
    for rank, item in enumerate(results[:top_k], start=1):
        best = item["best_checkpoint"]
        p02 = best["prefix_metrics"].get("0.2", {}).get("average_accuracy", 0.0)
        p04 = best["prefix_metrics"].get("0.4", {}).get("average_accuracy", 0.0)
        print(
            f"{rank:>2}. score={item['score']:.4f} | "
            f"ood={best['ood_avg']:.2%} | test={best['test_avg']:.2%} | "
            f"p0.2={p02:.2%} | p0.4={p04:.2%} | "
            f"gate(ood)={best['ood_gate']:.3f} | "
            f"ckpt={best['checkpoint_label']} | combo={item['combo']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid search wrapper for train_fusion_gating_online.py with post-hoc checkpoint ranking"
    )
    parser.add_argument(
        "--train_script",
        type=Path,
        default=Path("/home/martina/Y3_Project/visuotactile/scripts/train_fusion_gating_online.py"),
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/martina/Y3_Project/Plaintextdataset",
    )
    parser.add_argument(
        "--output_base",
        type=Path,
        default=Path("/home/martina/Y3_Project/visuotactile/outputs/grid_search_fusion_gating_online"),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_trials", type=int, default=0, help="0 means run all combinations")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--shuffle_grid",
        action="store_true",
        help="Shuffle the full grid before applying --max_trials",
    )
    parser.add_argument(
        "--fast_preset",
        action="store_true",
        help="Use a much cheaper first-pass search: shorter training, fewer checkpoints, fewer ratios, skip test eval",
    )
    parser.add_argument(
        "--skip_test_eval",
        action="store_true",
        help="Skip eval_test during search and score only on OOD + online OOD metrics",
    )

    parser.set_defaults(resume=True)
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no_resume", dest="resume", action="store_false")

    parser.set_defaults(live_plot=False)
    parser.add_argument("--live_plot", dest="live_plot", action="store_true")
    parser.add_argument("--no_live_plot", dest="live_plot", action="store_false")

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_tactile_len", type=int, default=3000)

    parser.add_argument("--fusion_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.set_defaults(freeze_visual=True)
    parser.add_argument("--freeze_visual", dest="freeze_visual", action="store_true")
    parser.add_argument("--unfreeze_visual", dest="freeze_visual", action="store_false")
    parser.add_argument("--tactile_drop_prob", type=float, default=0.0)
    parser.add_argument("--block_modality", type=str, default="none", choices=["none", "visual", "tactile"])

    parser.add_argument("--lambda_aux", type=float, default=0.5)
    parser.add_argument("--reg_type", type=str, default="entropy")
    parser.add_argument("--gate_target_mean", type=float, default=0.5)
    parser.add_argument("--gate_entropy_eps", type=float, default=1e-6)
    parser.add_argument("--gate_reg_warmup_epochs", type=int, default=5)
    parser.add_argument("--gate_reg_ramp_epochs", type=int, default=10)

    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--early_stop_acc", type=float, default=1.0)
    parser.add_argument("--early_stop_min_epoch", type=int, default=0)

    parser.add_argument("--online_train_prob_grid", type=str, default="0.6,0.8,1.0")
    parser.add_argument("--online_min_prefix_ratio_grid", type=str, default="0.1,0.2,0.3,0.4")
    parser.add_argument("--min_prefix_len_grid", type=str, default="32,64")
    parser.add_argument("--lambda_reg_grid", type=str, default="0.05,0.1,0.2")
    parser.add_argument("--visual_drop_prob_grid", type=str, default="0.05,0.1,0.2")

    parser.add_argument("--candidate_epochs", type=str, default="10,20,40,60")
    parser.add_argument("--online_eval_ratios", type=str, default="0.1,0.2,0.4,0.6,0.8,1.0")
    parser.add_argument(
        "--score_weights",
        type=str,
        default="",
        help="Comma-separated weights such as test=0.1,ood=0.45,p0.2=0.15,p0.4=0.3",
    )
    args = parser.parse_args()
    raw_candidate_epochs = args.candidate_epochs
    raw_online_eval_ratios = args.online_eval_ratios
    raw_score_weights = args.score_weights
    args.candidate_epochs = parse_int_list(args.candidate_epochs)
    args.score_weights = parse_score_weights(args.score_weights)
    if args.fast_preset:
        if args.epochs == 60:
            args.epochs = 20
        if args.save_every == 10:
            args.save_every = 20
        if raw_candidate_epochs.strip() == "10,20,40,60":
            args.candidate_epochs = [20]
        if raw_online_eval_ratios.strip() == "0.1,0.2,0.4,0.6,0.8,1.0":
            args.online_eval_ratios = "0.2,0.4,1.0"
        if not raw_score_weights.strip():
            args.score_weights = dict(FAST_SCORE_WEIGHTS)
        args.skip_test_eval = True
        args.shuffle_grid = True
    return args


def main() -> None:
    args = parse_args()
    output_base = args.output_base
    output_base.mkdir(parents=True, exist_ok=True)

    combos = build_grid(args)
    if args.shuffle_grid:
        rng = random.Random(args.seed)
        rng.shuffle(combos)
    if args.max_trials > 0:
        combos = combos[: args.max_trials]

    print("===============================================================")
    print("Grid Search: Fusion Gating Online")
    print(f"train_script: {args.train_script}")
    print(f"output_base : {output_base}")
    print(f"num_trials  : {len(combos)}")
    print(f"score_wts   : {args.score_weights}")
    print("===============================================================")

    results: List[Dict[str, object]] = []
    for idx, combo in enumerate(combos, start=1):
        trial_dir = output_base / f"trial_{idx:03d}__{combo_slug(combo)}"
        print(f"\n[trial {idx:03d}/{len(combos):03d}] {combo}")
        summary = summarize_trial(
            args=args,
            trial_index=idx,
            combo=combo,
            trial_dir=trial_dir,
            dry_run=args.dry_run,
        )
        results.append(summary)

    if args.dry_run:
        return

    results = sorted(results, key=lambda item: item["score"], reverse=True)
    write_summary_files(output_base, results)
    print_top_results(results, top_k=args.top_k)
    print(f"\nSummary JSON: {output_base / 'grid_search_summary.json'}")
    print(f"Summary CSV : {output_base / 'grid_search_summary.csv'}")
    print(f"Best config  : {output_base / 'grid_search_best.json'}")


if __name__ == "__main__":
    main()
