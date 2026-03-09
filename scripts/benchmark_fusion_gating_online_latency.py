import argparse
import json
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

try:
    from train_fusion_gating_online import (
        build_loader,
        build_model,
        build_prefix_padding_mask,
        fixed_prefix_lengths,
        parse_prefix_ratios,
        resolve_device,
        set_seed,
    )
except ImportError:  # pragma: no cover
    from visuotactile.scripts.train_fusion_gating_online import (
        build_loader,
        build_model,
        build_prefix_padding_mask,
        fixed_prefix_lengths,
        parse_prefix_ratios,
        resolve_device,
        set_seed,
    )


@dataclass
class CachedBatch:
    images: torch.Tensor
    tactile: torch.Tensor
    padding_mask: torch.Tensor


def downsample_tactile_padding_mask(padding_mask: torch.Tensor) -> torch.Tensor:
    tac_mask = padding_mask.float().unsqueeze(1)
    tac_mask = F.max_pool1d(tac_mask, kernel_size=2, stride=2)
    tac_mask = F.max_pool1d(tac_mask, kernel_size=2, stride=2)
    tac_mask = F.max_pool1d(tac_mask, kernel_size=2, stride=2)
    return tac_mask.squeeze(1) > 0.5


def summarize_timings(times_ms: List[float], control_hz: float) -> Dict[str, float]:
    arr = np.asarray(times_ms, dtype=np.float64)
    mean_ms = float(np.mean(arr))
    control_budget_ms = 1000.0 / max(control_hz, 1e-8)
    return {
        "mean_ms": mean_ms,
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "max_ms": float(np.max(arr)),
        "hz": float(1000.0 / mean_ms) if mean_ms > 0 else float("inf"),
        "control_budget_ms": control_budget_ms,
        "control_margin_x": float(control_budget_ms / mean_ms) if mean_ms > 0 else float("inf"),
    }


def load_model_and_batches(args: argparse.Namespace):
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})

    split_dir = Path(args.data_root) / args.eval_split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split not found: {split_dir}")

    loader = build_loader(
        split_dir,
        batch_size=args.batch_size,
        max_tactile_len=cfg.get("max_tactile_len", args.max_tactile_len),
        num_workers=0,
        shuffle=False,
    )
    dataset = loader.dataset
    model = build_model(
        cfg,
        args,
        mass_classes=len(dataset.mass_to_idx),
        stiffness_classes=len(dataset.stiffness_to_idx),
        material_classes=len(dataset.material_to_idx),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    cached_batches: List[CachedBatch] = []
    non_blocking = device.type == "cuda"
    for batch in loader:
        cached_batches.append(
            CachedBatch(
                images=batch["image"].to(device, non_blocking=non_blocking),
                tactile=batch["tactile"].to(device, non_blocking=non_blocking),
                padding_mask=batch["padding_mask"].to(device, non_blocking=non_blocking),
            )
        )
        if len(cached_batches) >= args.cached_batches:
            break
    if not cached_batches:
        raise RuntimeError(f"No batches loaded from {split_dir}")

    return device, cfg, checkpoint, model, dataset, cached_batches


def build_autocast_context(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def forward_prefix(
    model: torch.nn.Module,
    batch: CachedBatch,
    prefix_ratio: float,
    min_prefix_len: int,
    amp_enabled: bool,
) -> Dict[str, torch.Tensor]:
    prefix_lengths = fixed_prefix_lengths(batch.padding_mask, prefix_ratio, min_prefix_len)
    prefix_mask = build_prefix_padding_mask(batch.padding_mask, prefix_lengths)
    amp_context = build_autocast_context(batch.images.device, amp_enabled)
    with torch.inference_mode():
        with amp_context:
            return model(batch.images, batch.tactile, padding_mask=prefix_mask)


def time_single_call(
    model: torch.nn.Module,
    batch: CachedBatch,
    prefix_ratio: float,
    min_prefix_len: int,
    amp_enabled: bool,
) -> float:
    device = batch.images.device
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        outputs = forward_prefix(model, batch, prefix_ratio, min_prefix_len, amp_enabled)
        _ = outputs["gate_score"]
        end_event.record()
        torch.cuda.synchronize(device)
        return float(start_event.elapsed_time(end_event))

    start_time = time.perf_counter()
    outputs = forward_prefix(model, batch, prefix_ratio, min_prefix_len, amp_enabled)
    _ = outputs["gate_score"]
    return (time.perf_counter() - start_time) * 1000.0


def measure_prefix(
    model: torch.nn.Module,
    batches: List[CachedBatch],
    prefix_ratio: float,
    min_prefix_len: int,
    warmup_iters: int,
    measure_iters: int,
    control_hz: float,
    amp_enabled: bool,
) -> Dict[str, float]:
    for i in range(warmup_iters):
        batch = batches[i % len(batches)]
        _ = time_single_call(model, batch, prefix_ratio, min_prefix_len, amp_enabled)

    times_ms = []
    for i in range(measure_iters):
        batch = batches[i % len(batches)]
        times_ms.append(time_single_call(model, batch, prefix_ratio, min_prefix_len, amp_enabled))

    valid_lengths = []
    token_lengths = []
    for batch in batches:
        prefix_lengths = fixed_prefix_lengths(batch.padding_mask, prefix_ratio, min_prefix_len)
        prefix_mask = build_prefix_padding_mask(batch.padding_mask, prefix_lengths)
        token_mask = downsample_tactile_padding_mask(prefix_mask)
        valid_lengths.extend(prefix_lengths.tolist())
        token_lengths.extend((~token_mask).sum(dim=1).tolist())

    metrics = summarize_timings(times_ms, control_hz)
    metrics.update(
        {
            "prefix_ratio": prefix_ratio,
            "raw_prefix_len_mean": float(np.mean(valid_lengths)),
            "raw_prefix_len_min": int(np.min(valid_lengths)),
            "raw_prefix_len_max": int(np.max(valid_lengths)),
            "token_prefix_len_mean": float(np.mean(token_lengths)),
            "token_prefix_len_min": int(np.min(token_lengths)),
            "token_prefix_len_max": int(np.max(token_lengths)),
        }
    )
    return metrics


def measure_schedule(
    model: torch.nn.Module,
    batches: List[CachedBatch],
    prefix_ratios: List[float],
    min_prefix_len: int,
    warmup_iters: int,
    measure_iters: int,
    control_hz: float,
    amp_enabled: bool,
) -> Dict[str, float]:
    device = batches[0].images.device

    def run_schedule(batch: CachedBatch) -> None:
        for ratio in prefix_ratios:
            outputs = forward_prefix(model, batch, ratio, min_prefix_len, amp_enabled)
            _ = outputs["gate_score"]

    for i in range(warmup_iters):
        run_schedule(batches[i % len(batches)])
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    times_ms = []
    for i in range(measure_iters):
        batch = batches[i % len(batches)]
        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            run_schedule(batch)
            end_event.record()
            torch.cuda.synchronize(device)
            times_ms.append(float(start_event.elapsed_time(end_event)))
        else:
            start_time = time.perf_counter()
            run_schedule(batch)
            times_ms.append((time.perf_counter() - start_time) * 1000.0)

    metrics = summarize_timings(times_ms, control_hz)
    metrics["num_prefix_calls"] = len(prefix_ratios)
    metrics["avg_ms_per_call"] = metrics["mean_ms"] / max(len(prefix_ratios), 1)
    metrics["avg_hz_per_call_equivalent"] = float(1000.0 / metrics["avg_ms_per_call"]) if metrics["avg_ms_per_call"] > 0 else float("inf")
    return metrics


def build_markdown_report(result: Dict) -> str:
    device_name = result["device_name"]
    single_sentence = result["paper_ready"]["single_call_sentence"]
    schedule_sentence = result["paper_ready"]["schedule_sentence"]

    lines = [
        "# `fusion_gating_online_v2` Inference Latency Benchmark",
        "",
        f"- checkpoint: `{result['checkpoint']}`",
        f"- device: `{device_name}`",
        f"- eval split: `{result['eval_split']}`",
        f"- batch size: `{result['batch_size']}`",
        f"- cached batches: `{result['cached_batches']}`",
        f"- warmup / measure iters: `{result['warmup_iters']} / {result['measure_iters']}`",
        f"- control target: `{result['control_hz']:.1f} Hz` (`{1000.0 / result['control_hz']:.2f} ms` budget)",
        f"- precision: `{'AMP fp16' if result['amp_enabled'] else 'FP32'}`",
        "",
        "## Prefix Latency",
        "",
        "| Prefix | Raw Prefix Len (mean) | Token Len (mean) | Mean ms | P50 ms | P90 ms | Hz | Margin vs 100Hz |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for item in result["per_prefix"]:
        lines.append(
            f"| `{item['prefix_ratio']:.1f}` | `{item['raw_prefix_len_mean']:.1f}` | "
            f"`{item['token_prefix_len_mean']:.1f}` | `{item['mean_ms']:.3f}` | "
            f"`{item['p50_ms']:.3f}` | `{item['p90_ms']:.3f}` | `{item['hz']:.1f}` | "
            f"`{item['control_margin_x']:.2f}x` |"
        )

    schedule = result["prefix_schedule"]
    lines.extend(
        [
            "",
            "## Full Re-computation Schedule",
            "",
            f"- prefix schedule: `{result['prefix_ratios']}`",
            f"- total mean latency across the full prefix schedule: `{schedule['mean_ms']:.3f} ms`",
            f"- average per-call latency inside the schedule: `{schedule['avg_ms_per_call']:.3f} ms`",
            f"- equivalent per-call frequency: `{schedule['avg_hz_per_call_equivalent']:.1f} Hz`",
            "",
            "## Paper-ready Sentences",
            "",
            f"- {single_sentence}",
            f"- {schedule_sentence}",
            "",
            "## Notes",
            "",
            "- This benchmark excludes disk I/O and dataloader time; batches are preloaded to the target device.",
            "- The measured latency includes prefix-mask construction plus the full model forward pass.",
            "- `fusion_gating_online_v2` is a prefix re-computation model, so the relevant quantity is the per-query latency rather than KV-cache update cost.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark prefix re-computation latency for fusion_gating_online_v2"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/martina/Y3_Project/visuotactile/outputs/fusion_gating_online_v2/best_model.pth"),
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/martina/Y3_Project/Plaintextdataset",
    )
    parser.add_argument("--eval_split", choices=["val", "test", "ood_test"], default="ood_test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--cached_batches", type=int, default=8)
    parser.add_argument("--warmup_iters", type=int, default=30)
    parser.add_argument("--measure_iters", type=int, default=100)
    parser.add_argument("--prefix_ratios", type=str, default="0.1,0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--control_hz", type=float, default=100.0)
    parser.add_argument("--max_tactile_len", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fusion_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--freeze_visual", action="store_true", default=True)
    parser.add_argument("--amp", action="store_true", help="Use fp16 autocast on CUDA during inference")
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed, args.device)
    prefix_ratios = parse_prefix_ratios(args.prefix_ratios)

    device, cfg, checkpoint, model, dataset, batches = load_model_and_batches(args)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True

    per_prefix = []
    for ratio in prefix_ratios:
        per_prefix.append(
            measure_prefix(
                model=model,
                batches=batches,
                prefix_ratio=ratio,
                min_prefix_len=cfg.get("min_prefix_len", 64),
                warmup_iters=args.warmup_iters,
                measure_iters=args.measure_iters,
                control_hz=args.control_hz,
                amp_enabled=args.amp,
            )
        )

    prefix_schedule = measure_schedule(
        model=model,
        batches=batches,
        prefix_ratios=prefix_ratios,
        min_prefix_len=cfg.get("min_prefix_len", 64),
        warmup_iters=max(5, args.warmup_iters // 3),
        measure_iters=args.measure_iters,
        control_hz=args.control_hz,
        amp_enabled=args.amp,
    )

    worst_case = max(per_prefix, key=lambda item: item["mean_ms"])
    single_sentence = (
        f"On {torch.cuda.get_device_name(device) if device.type == 'cuda' else str(device)}, "
        f"`fusion_gating_online_v2` runs a single prefix query (batch={args.batch_size}, "
        f"{'AMP fp16' if args.amp else 'FP32'}) in {worst_case['mean_ms']:.3f} ms on average "
        f"at the worst tested prefix, i.e. {worst_case['hz']:.1f} Hz, which is "
        f"{worst_case['control_margin_x']:.2f}x faster than a {args.control_hz:.0f} Hz control loop."
    )
    schedule_sentence = (
        f"Recomputing the full prefix schedule {prefix_ratios} takes {prefix_schedule['mean_ms']:.3f} ms "
        f"in total on average, corresponding to {prefix_schedule['avg_ms_per_call']:.3f} ms per query."
    )

    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "eval_split": args.eval_split,
        "batch_size": args.batch_size,
        "cached_batches": len(batches),
        "warmup_iters": args.warmup_iters,
        "measure_iters": args.measure_iters,
        "control_hz": args.control_hz,
        "amp_enabled": args.amp,
        "prefix_ratios": prefix_ratios,
        "model_config": {
            "fusion_dim": cfg.get("fusion_dim", args.fusion_dim),
            "num_heads": cfg.get("num_heads"),
            "num_layers": cfg.get("num_layers"),
            "dropout": cfg.get("dropout"),
            "freeze_visual": cfg.get("freeze_visual"),
            "online_train_prob": cfg.get("online_train_prob"),
            "online_min_prefix_ratio": cfg.get("online_min_prefix_ratio"),
            "min_prefix_len": cfg.get("min_prefix_len"),
            "max_tactile_len": cfg.get("max_tactile_len"),
        },
        "num_samples_in_split": len(dataset),
        "per_prefix": per_prefix,
        "prefix_schedule": prefix_schedule,
        "paper_ready": {
            "single_call_sentence": single_sentence,
            "schedule_sentence": schedule_sentence,
        },
    }

    output_dir = args.output_dir or (args.checkpoint.parent / "latency_benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "latency_results.json"
    md_path = output_dir / "latency_report.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(build_markdown_report(result), encoding="utf-8")

    print("=" * 70)
    print("fusion_gating_online_v2 latency benchmark")
    print("=" * 70)
    for item in per_prefix:
        print(
            f"prefix={item['prefix_ratio']:.1f} | mean={item['mean_ms']:.3f} ms | "
            f"p90={item['p90_ms']:.3f} ms | hz={item['hz']:.1f} | "
            f"margin_vs_{args.control_hz:.0f}Hz={item['control_margin_x']:.2f}x"
        )
    print("-" * 70)
    print(
        f"schedule={prefix_ratios} | total_mean={prefix_schedule['mean_ms']:.3f} ms | "
        f"per_call_mean={prefix_schedule['avg_ms_per_call']:.3f} ms"
    )
    print("-" * 70)
    print(single_sentence)
    print(schedule_sentence)
    print(f"JSON: {json_path}")
    print(f"MD  : {md_path}")


if __name__ == "__main__":
    main()
