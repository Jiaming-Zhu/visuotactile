import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    from train_fusion_gating2 import (
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
    from visuotactile.scripts.train_fusion_gating2 import (
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
VISUAL_TOKENS = 49


def parse_ratio_list(raw: str) -> List[float]:
    values: List[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value <= 0.0 or value > 1.0:
            raise ValueError(f"ratio must be in (0, 1], got {value}")
        values.append(value)
    if not values:
        raise ValueError("at least one ratio is required")
    return sorted(set(values))


def resolve_chunk_size(args: argparse.Namespace) -> int:
    chunk_size = int(args.chunk_size)
    if args.window_size > 0 and args.window_size != chunk_size:
        raise ValueError("--window_size must equal --chunk_size in this v1 streaming implementation")
    if args.step_size > 0 and args.step_size != chunk_size:
        raise ValueError("--step_size must equal --chunk_size in this v1 streaming implementation")
    args.window_size = chunk_size
    args.step_size = chunk_size
    return chunk_size


def compute_valid_lengths(padding_mask: torch.Tensor) -> torch.Tensor:
    return (~padding_mask).sum(dim=1).long()


def masked_mean(tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    valid = valid_mask.unsqueeze(-1).float()
    denom = valid.sum(dim=1).clamp(min=1.0)
    return (tokens * valid).sum(dim=1) / denom


def downsample_padding_mask(padding_mask: torch.Tensor, stages: int = 3) -> torch.Tensor:
    mask = padding_mask.float().unsqueeze(1)
    for _ in range(stages):
        mask = F.max_pool1d(mask, kernel_size=2, stride=2)
    return mask.squeeze(1) > 0.5


def build_chunk_valid_len(valid_lengths: torch.Tensor, chunk_index: int, chunk_size: int) -> torch.Tensor:
    start = chunk_index * chunk_size
    remaining = valid_lengths - start
    return remaining.clamp(min=0, max=chunk_size)


def zero_logits(batch_size: int, num_classes: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros(batch_size, num_classes, device=device, dtype=dtype)


def gather_valid_step_positions(step_valid_mask: torch.Tensor) -> List[torch.Tensor]:
    positions: List[torch.Tensor] = []
    for sample_mask in step_valid_mask:
        positions.append(sample_mask.nonzero(as_tuple=False).squeeze(-1))
    return positions


def ratio_to_step_index(ratio: float, num_valid_steps: int) -> int:
    return int(math.ceil(ratio * num_valid_steps))


def build_early_indices(num_valid_steps: int, ratios: List[float]) -> List[int]:
    unique_indices = set()
    for ratio in ratios:
        idx = ratio_to_step_index(ratio, num_valid_steps)
        if 1 <= idx < num_valid_steps:
            unique_indices.add(idx)
    return sorted(unique_indices)


class TactileChunkTokenizer(nn.Module):
    def __init__(self, fusion_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
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
        self.pool_stages = 3

    @staticmethod
    def _conv_output_length(length: int, kernel_size: int, stride: int, padding: int) -> int:
        return math.floor((length + 2 * padding - (kernel_size - 1) - 1) / stride + 1)

    @classmethod
    def output_length(cls, chunk_size: int) -> int:
        length = chunk_size
        length = cls._conv_output_length(length, kernel_size=7, stride=2, padding=3)
        length = cls._conv_output_length(length, kernel_size=5, stride=2, padding=2)
        length = cls._conv_output_length(length, kernel_size=3, stride=2, padding=1)
        return max(1, length)

    def forward(
        self,
        tactile_chunk: torch.Tensor,
        chunk_valid_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.encoder(tactile_chunk).transpose(1, 2)
        positions = torch.arange(tactile_chunk.size(-1), device=tactile_chunk.device).unsqueeze(0)
        raw_padding_mask = positions >= chunk_valid_len.unsqueeze(1)
        token_padding_mask = downsample_padding_mask(raw_padding_mask, stages=self.pool_stages)
        token_padding_mask = token_padding_mask[:, : tokens.size(1)]
        token_valid_mask = ~token_padding_mask
        token_valid_len = token_valid_mask.sum(dim=1).long()
        tokens = tokens * token_valid_mask.unsqueeze(-1)
        summary = masked_mean(tokens, token_valid_mask)
        return tokens, token_valid_len, token_valid_mask, summary


class StreamMultiheadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def project_kv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._reshape(self.k_proj(x)), self._reshape(self.v_proj(x))

    def forward(
        self,
        query_input: torch.Tensor,
        visual_tokens: torch.Tensor,
        past_key: torch.Tensor,
        past_value: torch.Tensor,
        past_valid_len: torch.Tensor,
        current_context: torch.Tensor,
        current_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = query_input.size(0)
        q = self._reshape(self.q_proj(query_input))
        k_visual, v_visual = self.project_kv(visual_tokens)
        k_current, v_current = self.project_kv(current_context)

        max_past = int(past_valid_len.max().item()) if past_valid_len.numel() > 0 else 0
        if max_past > 0:
            k_past = past_key[:, :, :max_past, :]
            v_past = past_value[:, :, :max_past, :]
            past_positions = torch.arange(max_past, device=past_valid_len.device).unsqueeze(0)
            past_padding_mask = past_positions >= past_valid_len.unsqueeze(1)
        else:
            k_past = k_visual.new_zeros(batch_size, self.num_heads, 0, self.head_dim)
            v_past = v_visual.new_zeros(batch_size, self.num_heads, 0, self.head_dim)
            past_padding_mask = torch.zeros(batch_size, 0, device=query_input.device, dtype=torch.bool)

        visual_padding_mask = torch.zeros(
            batch_size,
            visual_tokens.size(1),
            device=query_input.device,
            dtype=torch.bool,
        )
        current_padding_mask = ~current_valid_mask
        key_padding_mask = torch.cat(
            [visual_padding_mask, past_padding_mask, current_padding_mask],
            dim=1,
        )

        k_full = torch.cat([k_visual, k_past, k_current], dim=2)
        v_full = torch.cat([v_visual, v_past, v_current], dim=2)

        attn_scores = torch.matmul(q, k_full.transpose(-2, -1)) * self.scale
        attn_scores = attn_scores.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(1),
            torch.finfo(attn_scores.dtype).min,
        )
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v_full)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, query_input.size(1), self.d_model)
        return self.out_proj(attn_output)


class CausalStreamTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        max_cache_tokens: int,
    ) -> None:
        super().__init__()
        self.max_cache_tokens = max_cache_tokens
        self.attn = StreamMultiheadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.query_norm = nn.LayerNorm(d_model)
        self.visual_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.cache_norm = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def _append_cache(
        self,
        cache_entry: Dict[str, torch.Tensor],
        key_to_append: torch.Tensor,
        value_to_append: torch.Tensor,
        token_valid_mask: torch.Tensor,
        sample_active: torch.Tensor,
    ) -> None:
        for batch_idx in range(token_valid_mask.size(0)):
            if not bool(sample_active[batch_idx].item()):
                continue
            valid_idx = token_valid_mask[batch_idx].nonzero(as_tuple=False).squeeze(-1)
            if valid_idx.numel() == 0:
                continue
            start = int(cache_entry["valid_len"][batch_idx].item())
            end = start + int(valid_idx.numel())
            if end > self.max_cache_tokens:
                raise RuntimeError(
                    f"tactile cache overflow: need {end} tokens but cache capacity is {self.max_cache_tokens}"
                )
            cache_entry["key"][batch_idx, :, start:end, :] = key_to_append[batch_idx, :, valid_idx, :]
            cache_entry["value"][batch_idx, :, start:end, :] = value_to_append[batch_idx, :, valid_idx, :]
            cache_entry["valid_len"][batch_idx] = end

    def forward(
        self,
        prev_cls_state: torch.Tensor,
        current_tokens: torch.Tensor,
        token_valid_mask: torch.Tensor,
        visual_tokens: torch.Tensor,
        cache_entry: Dict[str, torch.Tensor],
        sample_active: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([prev_cls_state.unsqueeze(1), current_tokens], dim=1)
        query_input = self.query_norm(combined)
        attn_output = self.attn(
            query_input=query_input,
            visual_tokens=self.visual_norm(visual_tokens),
            past_key=cache_entry["key"],
            past_value=cache_entry["value"],
            past_valid_len=cache_entry["valid_len"],
            current_context=query_input[:, 1:, :],
            current_valid_mask=token_valid_mask,
        )
        combined = combined + self.attn_dropout(attn_output)
        combined = combined + self.ffn(self.ffn_norm(combined))

        next_cls_state = combined[:, 0, :]
        next_tokens = combined[:, 1:, :] * token_valid_mask.unsqueeze(-1)
        cache_tokens = self.cache_norm(next_tokens)
        key_to_append, value_to_append = self.attn.project_kv(cache_tokens)
        self._append_cache(
            cache_entry=cache_entry,
            key_to_append=key_to_append,
            value_to_append=value_to_append,
            token_valid_mask=token_valid_mask,
            sample_active=sample_active,
        )
        next_cls_state = torch.where(sample_active.unsqueeze(-1), next_cls_state, prev_cls_state)
        return next_cls_state, next_tokens


class StreamingFusionGatingTransformer(nn.Module):
    def __init__(
        self,
        fusion_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_layers: int = 4,
        freeze_visual: bool = True,
        pretrained_visual: bool = True,
        mass_classes: int = 4,
        stiffness_classes: int = 4,
        material_classes: int = 5,
        chunk_size: int = 256,
        max_tactile_len: int = 3000,
    ) -> None:
        super().__init__()
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.chunk_size = chunk_size
        self.max_tactile_len = max_tactile_len
        self.mass_classes = mass_classes
        self.stiffness_classes = stiffness_classes
        self.material_classes = material_classes

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained_visual else None
        resnet = models.resnet18(weights=weights)
        self.vis_backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.vis_proj = nn.Conv2d(512, fusion_dim, kernel_size=1)
        if freeze_visual:
            for param in self.vis_backbone.parameters():
                param.requires_grad = False

        self.tactile_tokenizer = TactileChunkTokenizer(fusion_dim=fusion_dim)
        self.max_chunk_tokens = self.tactile_tokenizer.output_length(chunk_size)
        self.max_steps = math.ceil(max_tactile_len / chunk_size)
        self.max_cache_tokens = self.max_steps * self.max_chunk_tokens

        self.cls_init = nn.Parameter(torch.randn(1, fusion_dim))
        self.visual_pos_emb = nn.Parameter(torch.randn(1, VISUAL_TOKENS, fusion_dim))
        self.tactile_pos_emb = nn.Parameter(torch.randn(1, self.max_chunk_tokens, fusion_dim))
        self.step_emb = nn.Embedding(self.max_steps + 1, fusion_dim)
        self.t_null = nn.Parameter(torch.randn(1, 1, fusion_dim))

        self.gate_mlp = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, 1),
            nn.Sigmoid(),
        )

        self.layers = nn.ModuleList(
            [
                CausalStreamTransformerBlock(
                    d_model=fusion_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    max_cache_tokens=self.max_cache_tokens,
                )
                for _ in range(num_layers)
            ]
        )

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
        self.aux_head_mass = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.GELU(),
            nn.Linear(64, mass_classes),
        )
        self.aux_head_stiffness = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.GELU(),
            nn.Linear(64, stiffness_classes),
        )
        self.aux_head_material = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.GELU(),
            nn.Linear(64, material_classes),
        )

    def encode_visual(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        visual = self.vis_backbone(image)
        visual = self.vis_proj(visual)
        visual_tokens = visual.flatten(2).transpose(1, 2)
        visual_tokens = visual_tokens + self.visual_pos_emb[:, : visual_tokens.size(1), :]
        visual_summary = visual_tokens.mean(dim=1)
        return {"visual_tokens": visual_tokens, "visual_summary": visual_summary}

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        visual_cache: Dict[str, torch.Tensor],
    ) -> Dict[str, object]:
        dtype = visual_cache["visual_tokens"].dtype
        cache = []
        for _ in range(len(self.layers)):
            cache.append(
                {
                    "key": torch.zeros(
                        batch_size,
                        self.num_heads,
                        self.max_cache_tokens,
                        self.fusion_dim // self.num_heads,
                        device=device,
                        dtype=dtype,
                    ),
                    "value": torch.zeros(
                        batch_size,
                        self.num_heads,
                        self.max_cache_tokens,
                        self.fusion_dim // self.num_heads,
                        device=device,
                        dtype=dtype,
                    ),
                    "valid_len": torch.zeros(batch_size, device=device, dtype=torch.long),
                }
            )

        last_outputs = {
            "mass": zero_logits(batch_size, self.mass_classes, device, dtype),
            "stiffness": zero_logits(batch_size, self.stiffness_classes, device, dtype),
            "material": zero_logits(batch_size, self.material_classes, device, dtype),
        }
        last_aux_outputs = {
            "aux_mass": zero_logits(batch_size, self.mass_classes, device, dtype),
            "aux_stiffness": zero_logits(batch_size, self.stiffness_classes, device, dtype),
            "aux_material": zero_logits(batch_size, self.material_classes, device, dtype),
        }
        return {
            "visual_tokens": visual_cache["visual_tokens"],
            "visual_summary": visual_cache["visual_summary"],
            "cls_state": self.cls_init.expand(batch_size, -1).clone(),
            "step_index": torch.zeros(batch_size, device=device, dtype=torch.long),
            "tactile_kv_cache": cache,
            "last_outputs": last_outputs,
            "last_aux_outputs": last_aux_outputs,
            "last_gate_score": torch.zeros(batch_size, device=device, dtype=dtype),
        }

    def reset_state(self, state: Dict[str, object]) -> Dict[str, object]:
        batch_size = state["cls_state"].size(0)
        device = state["cls_state"].device
        visual_cache = {
            "visual_tokens": state["visual_tokens"],
            "visual_summary": state["visual_summary"],
        }
        return self.init_state(batch_size=batch_size, device=device, visual_cache=visual_cache)

    def _pad_chunk(self, tactile_chunk: torch.Tensor) -> torch.Tensor:
        if tactile_chunk.size(-1) == self.chunk_size:
            return tactile_chunk
        if tactile_chunk.size(-1) > self.chunk_size:
            raise ValueError(
                f"tactile_chunk length {tactile_chunk.size(-1)} exceeds configured chunk_size {self.chunk_size}"
            )
        pad_len = self.chunk_size - tactile_chunk.size(-1)
        return F.pad(tactile_chunk, (0, pad_len))

    def forward_step(
        self,
        tactile_chunk: torch.Tensor,
        chunk_valid_len: torch.Tensor,
        state: Dict[str, object],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, object]]:
        tactile_chunk = self._pad_chunk(tactile_chunk)
        chunk_valid_len = chunk_valid_len.long().clamp(min=0, max=self.chunk_size)
        token_input, token_valid_len, token_valid_mask, current_summary = self.tactile_tokenizer(
            tactile_chunk=tactile_chunk,
            chunk_valid_len=chunk_valid_len,
        )
        sample_active = (chunk_valid_len > 0) & (token_valid_len > 0)

        step_ids = state["step_index"].clamp(max=self.max_steps)
        step_embed = self.step_emb(step_ids)
        current_tokens = token_input + self.tactile_pos_emb[:, : token_input.size(1), :] + step_embed.unsqueeze(1)
        current_tokens = current_tokens * token_valid_mask.unsqueeze(-1)

        gate_input = torch.cat(
            [state["visual_summary"], state["cls_state"], current_summary],
            dim=-1,
        )
        current_gate = self.gate_mlp(gate_input).squeeze(-1)
        gate_for_attention = torch.where(sample_active, current_gate, state["last_gate_score"])
        gated_visual = (
            gate_for_attention.unsqueeze(-1).unsqueeze(-1) * state["visual_tokens"]
            + (1.0 - gate_for_attention).unsqueeze(-1).unsqueeze(-1) * self.t_null
        )

        current_cls = state["cls_state"] + step_embed
        for layer_idx, layer in enumerate(self.layers):
            current_cls, current_tokens = layer(
                prev_cls_state=current_cls,
                current_tokens=current_tokens,
                token_valid_mask=token_valid_mask,
                visual_tokens=gated_visual,
                cache_entry=state["tactile_kv_cache"][layer_idx],
                sample_active=sample_active,
            )

        new_outputs = {
            "mass": self.head_mass(current_cls),
            "stiffness": self.head_stiffness(current_cls),
            "material": self.head_material(current_cls),
        }
        new_aux_outputs = {
            "aux_mass": self.aux_head_mass(current_summary),
            "aux_stiffness": self.aux_head_stiffness(current_summary),
            "aux_material": self.aux_head_material(current_summary),
        }

        output_valid_mask = sample_active
        outputs = {}
        for task in TASKS:
            outputs[task] = torch.where(
                output_valid_mask.unsqueeze(-1),
                new_outputs[task],
                state["last_outputs"][task],
            )
        for aux_task in ["aux_mass", "aux_stiffness", "aux_material"]:
            outputs[aux_task] = torch.where(
                output_valid_mask.unsqueeze(-1),
                new_aux_outputs[aux_task],
                state["last_aux_outputs"][aux_task],
            )
        outputs["gate_score"] = torch.where(output_valid_mask, current_gate, state["last_gate_score"])

        state["cls_state"] = torch.where(output_valid_mask.unsqueeze(-1), current_cls, state["cls_state"])
        state["step_index"] = state["step_index"] + output_valid_mask.long()
        for task in TASKS:
            state["last_outputs"][task] = outputs[task]
        for aux_task in ["aux_mass", "aux_stiffness", "aux_material"]:
            state["last_aux_outputs"][aux_task] = outputs[aux_task]
        state["last_gate_score"] = outputs["gate_score"]

        return outputs, output_valid_mask, state

    def forward(
        self,
        image: torch.Tensor,
        tactile: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_step_outputs: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if padding_mask is None:
            padding_mask = torch.zeros(
                tactile.size(0),
                tactile.size(-1),
                device=tactile.device,
                dtype=torch.bool,
            )
        valid_lengths = compute_valid_lengths(padding_mask)
        visual_cache = self.encode_visual(image)
        state = self.init_state(batch_size=image.size(0), device=image.device, visual_cache=visual_cache)

        num_steps = math.ceil(tactile.size(-1) / self.chunk_size)
        step_outputs: Dict[str, List[torch.Tensor]] = {task: [] for task in TASKS}
        step_aux_outputs: Dict[str, List[torch.Tensor]] = {
            "aux_mass": [],
            "aux_stiffness": [],
            "aux_material": [],
        }
        step_gate_scores: List[torch.Tensor] = []
        step_valid_masks: List[torch.Tensor] = []

        for step in range(num_steps):
            start = step * self.chunk_size
            end = min((step + 1) * self.chunk_size, tactile.size(-1))
            chunk = tactile[:, :, start:end]
            chunk_valid_len = build_chunk_valid_len(valid_lengths, chunk_index=step, chunk_size=self.chunk_size)
            outputs, output_valid_mask, state = self.forward_step(
                tactile_chunk=chunk,
                chunk_valid_len=chunk_valid_len,
                state=state,
            )
            if return_step_outputs:
                for task in TASKS:
                    step_outputs[task].append(outputs[task])
                for aux_task in step_aux_outputs:
                    step_aux_outputs[aux_task].append(outputs[aux_task])
                step_gate_scores.append(outputs["gate_score"])
                step_valid_masks.append(output_valid_mask)

        final_outputs = {
            "mass": state["last_outputs"]["mass"],
            "stiffness": state["last_outputs"]["stiffness"],
            "material": state["last_outputs"]["material"],
            "aux_mass": state["last_aux_outputs"]["aux_mass"],
            "aux_stiffness": state["last_aux_outputs"]["aux_stiffness"],
            "aux_material": state["last_aux_outputs"]["aux_material"],
            "gate_score": state["last_gate_score"],
            "num_valid_steps": state["step_index"].clone(),
        }
        if return_step_outputs:
            final_outputs["step_outputs"] = {
                task: torch.stack(step_outputs[task], dim=1) for task in TASKS
            }
            final_outputs["step_aux_outputs"] = {
                aux_task: torch.stack(step_aux_outputs[aux_task], dim=1)
                for aux_task in step_aux_outputs
            }
            final_outputs["step_gate_scores"] = torch.stack(step_gate_scores, dim=1)
            final_outputs["step_valid_mask"] = torch.stack(step_valid_masks, dim=1)
        return final_outputs


def build_model(
    cfg: Dict[str, object],
    args: argparse.Namespace,
    mass_classes: int,
    stiffness_classes: int,
    material_classes: int,
) -> StreamingFusionGatingTransformer:
    return StreamingFusionGatingTransformer(
        fusion_dim=int(cfg.get("fusion_dim", args.fusion_dim)),
        num_heads=int(cfg.get("num_heads", args.num_heads)),
        dropout=float(cfg.get("dropout", args.dropout)),
        num_layers=int(cfg.get("num_layers", args.num_layers)),
        freeze_visual=bool(cfg.get("freeze_visual", args.freeze_visual)),
        pretrained_visual=bool(cfg.get("pretrained_visual", args.pretrained_visual)),
        mass_classes=mass_classes,
        stiffness_classes=stiffness_classes,
        material_classes=material_classes,
        chunk_size=int(cfg.get("chunk_size", args.chunk_size)),
        max_tactile_len=int(cfg.get("max_tactile_len", args.max_tactile_len)),
    )


def compute_gate_regularization(
    step_gate_scores: torch.Tensor,
    step_valid_mask: torch.Tensor,
    reg_type: str,
    gate_target_mean: float,
    gate_entropy_eps: float,
) -> torch.Tensor:
    sample_terms = []
    for sample_idx in range(step_gate_scores.size(0)):
        valid_idx = step_valid_mask[sample_idx].nonzero(as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            continue
        g = step_gate_scores[sample_idx, valid_idx]
        if reg_type == "polarization":
            term = (g * (1.0 - g)).mean()
        elif reg_type == "sparsity":
            term = g.mean()
        elif reg_type == "mean":
            term = (g.mean() - gate_target_mean).pow(2)
        elif reg_type == "center":
            term = (g - 0.5).pow(2).mean()
        elif reg_type == "entropy":
            g_clamped = torch.clamp(g, gate_entropy_eps, 1.0 - gate_entropy_eps)
            entropy = -(
                g_clamped * torch.log(g_clamped)
                + (1.0 - g_clamped) * torch.log(1.0 - g_clamped)
            ).mean()
            term = math.log(2.0) - entropy
        else:
            term = torch.zeros((), device=step_gate_scores.device, dtype=step_gate_scores.dtype)
        sample_terms.append(term)
    if not sample_terms:
        return torch.zeros((), device=step_gate_scores.device, dtype=step_gate_scores.dtype)
    return torch.stack(sample_terms).mean()


def compute_multistep_loss(
    outputs: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    lambda_reg: float,
    lambda_aux: float,
    reg_type: str,
    gate_target_mean: float,
    gate_entropy_eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    step_valid_mask = outputs["step_valid_mask"]
    step_outputs = outputs["step_outputs"]
    sample_losses = []
    for sample_idx in range(step_valid_mask.size(0)):
        valid_idx = step_valid_mask[sample_idx].nonzero(as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            continue
        weights = torch.arange(
            1,
            valid_idx.numel() + 1,
            device=step_valid_mask.device,
            dtype=outputs["mass"].dtype,
        )
        weights = weights / weights.sum()
        task_loss = torch.zeros((), device=step_valid_mask.device, dtype=outputs["mass"].dtype)
        for task in TASKS:
            logits = step_outputs[task][sample_idx, valid_idx]
            repeated_labels = labels[task][sample_idx].repeat(valid_idx.numel())
            ce = F.cross_entropy(logits, repeated_labels, reduction="none")
            task_loss = task_loss + torch.sum(weights * ce)
        sample_losses.append(task_loss)
    if sample_losses:
        ce_loss = torch.stack(sample_losses).mean()
    else:
        ce_loss = outputs["mass"].sum() * 0.0

    aux_loss = (
        F.cross_entropy(outputs["aux_mass"], labels["mass"])
        + F.cross_entropy(outputs["aux_stiffness"], labels["stiffness"])
        + F.cross_entropy(outputs["aux_material"], labels["material"])
    )
    reg_loss = compute_gate_regularization(
        step_gate_scores=outputs["step_gate_scores"],
        step_valid_mask=step_valid_mask,
        reg_type=reg_type,
        gate_target_mean=gate_target_mean,
        gate_entropy_eps=gate_entropy_eps,
    )
    total_loss = ce_loss + lambda_aux * aux_loss + lambda_reg * reg_loss
    return total_loss, reg_loss, aux_loss


def compute_early_online_auc(
    outputs: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    ratios: List[float],
) -> Tuple[float, int]:
    step_valid_mask = outputs["step_valid_mask"]
    valid_positions = gather_valid_step_positions(step_valid_mask)
    sample_scores: List[float] = []
    for sample_idx, sample_positions in enumerate(valid_positions):
        num_valid_steps = int(sample_positions.numel())
        early_indices = build_early_indices(num_valid_steps, ratios)
        if len(early_indices) < 2:
            continue
        correctness = []
        for early_idx in early_indices:
            raw_step = int(sample_positions[early_idx - 1].item())
            for task in TASKS:
                pred = int(outputs["step_outputs"][task][sample_idx, raw_step].argmax(dim=-1).item())
                correctness.append(float(pred == int(labels[task][sample_idx].item())))
        sample_scores.append(float(np.mean(correctness)))
    if not sample_scores:
        return 0.0, 0
    return float(np.mean(sample_scores)), len(sample_scores)


def update_progress_stats(
    progress_stats: Dict[float, Dict[str, float]],
    outputs: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    ratios: List[float],
) -> None:
    step_valid_mask = outputs["step_valid_mask"]
    valid_positions = gather_valid_step_positions(step_valid_mask)
    for sample_idx, sample_positions in enumerate(valid_positions):
        num_valid_steps = int(sample_positions.numel())
        if num_valid_steps == 0:
            continue
        for ratio in ratios:
            idx = ratio_to_step_index(ratio, num_valid_steps)
            idx = max(1, min(idx, num_valid_steps))
            raw_step = int(sample_positions[idx - 1].item())
            progress_stats[ratio]["num_samples"] += 1
            progress_stats[ratio]["gate_sum"] += float(outputs["step_gate_scores"][sample_idx, raw_step].item())
            for task in TASKS:
                pred = int(outputs["step_outputs"][task][sample_idx, raw_step].argmax(dim=-1).item())
                target = int(labels[task][sample_idx].item())
                progress_stats[ratio][f"{task}_correct"] += float(pred == target)


def finalize_progress_stats(progress_stats: Dict[float, Dict[str, float]]) -> Dict[float, Dict[str, float]]:
    finalized = {}
    for ratio, stats in progress_stats.items():
        num_samples = max(1.0, stats["num_samples"])
        task_metrics = {
            task: float(stats[f"{task}_correct"] / num_samples)
            for task in TASKS
        }
        finalized[ratio] = {
            **task_metrics,
            "gate_score": float(stats["gate_sum"] / num_samples),
            "num_samples": int(stats["num_samples"]),
            "average_accuracy": float(np.mean([task_metrics[task] for task in TASKS])),
        }
    return finalized


def run_epoch(
    model: StreamingFusionGatingTransformer,
    loader,
    device: torch.device,
    args: argparse.Namespace,
    train_mode: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lambda_reg: float = 0.0,
    selection_ratios: Optional[List[float]] = None,
    progress_ratios: Optional[List[float]] = None,
    collect_predictions: bool = False,
) -> Dict[str, object]:
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_reg = 0.0
    total_aux = 0.0
    total_gate = 0.0
    total_samples = 0
    correct = {task: 0 for task in TASKS}
    early_score_sum = 0.0
    early_score_count = 0

    all_preds = {task: [] for task in TASKS}
    all_labels = {task: [] for task in TASKS}
    all_gate_scores: List[float] = []

    progress_stats: Dict[float, Dict[str, float]] = {}
    if progress_ratios is not None:
        for ratio in progress_ratios:
            progress_stats[ratio] = {
                "num_samples": 0.0,
                "gate_sum": 0.0,
                "mass_correct": 0.0,
                "stiffness_correct": 0.0,
                "material_correct": 0.0,
            }

    iterator = loader
    if tqdm is not None:
        desc = "train" if train_mode else "eval"
        iterator = tqdm(loader, leave=False, desc=desc)

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
            outputs = model(images, tactile, padding_mask=padding_mask, return_step_outputs=True)
            loss, reg_loss, aux_loss = compute_multistep_loss(
                outputs=outputs,
                labels=labels,
                lambda_reg=lambda_reg,
                lambda_aux=args.lambda_aux,
                reg_type=args.reg_type,
                gate_target_mean=args.gate_target_mean,
                gate_entropy_eps=args.gate_entropy_eps,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            optimizer.step()
        else:
            with torch.no_grad():
                outputs = model(images, tactile, padding_mask=padding_mask, return_step_outputs=True)
                loss, reg_loss, aux_loss = compute_multistep_loss(
                    outputs=outputs,
                    labels=labels,
                    lambda_reg=lambda_reg,
                    lambda_aux=args.lambda_aux,
                    reg_type=args.reg_type,
                    gate_target_mean=args.gate_target_mean,
                    gate_entropy_eps=args.gate_entropy_eps,
                )

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_reg += float(reg_loss.item()) * batch_size
        total_aux += float(aux_loss.item()) * batch_size
        total_gate += float(outputs["gate_score"].sum().item())
        total_samples += batch_size

        for task in TASKS:
            correct[task] += int((outputs[task].argmax(dim=1) == labels[task]).sum().item())

        if selection_ratios is not None:
            batch_score, batch_count = compute_early_online_auc(outputs=outputs, labels=labels, ratios=selection_ratios)
            early_score_sum += batch_score * batch_count
            early_score_count += batch_count

        if progress_ratios is not None:
            update_progress_stats(progress_stats=progress_stats, outputs=outputs, labels=labels, ratios=progress_ratios)

        if collect_predictions:
            for task in TASKS:
                all_preds[task].extend(outputs[task].argmax(dim=1).cpu().tolist())
                all_labels[task].extend(batch[task].tolist())
            all_gate_scores.extend(outputs["gate_score"].detach().cpu().tolist())

        if tqdm is not None and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                {
                    "loss": f"{total_loss / max(1, total_samples):.4f}",
                    "mass": f"{correct['mass'] / max(1, total_samples):.2%}",
                    "stiff": f"{correct['stiffness'] / max(1, total_samples):.2%}",
                    "mat": f"{correct['material'] / max(1, total_samples):.2%}",
                    "g": f"{total_gate / max(1, total_samples):.3f}",
                    "step": batch_idx,
                }
            )

    metrics: Dict[str, object] = {
        "loss": total_loss / max(1, total_samples),
        "reg_loss": total_reg / max(1, total_samples),
        "aux_loss": total_aux / max(1, total_samples),
        "gate_score": total_gate / max(1, total_samples),
        "mass": correct["mass"] / max(1, total_samples),
        "stiffness": correct["stiffness"] / max(1, total_samples),
        "material": correct["material"] / max(1, total_samples),
        "average_accuracy": float(np.mean([correct[task] / max(1, total_samples) for task in TASKS])),
        "early_online_auc": early_score_sum / max(1, early_score_count),
        "early_online_samples": early_score_count,
    }
    if progress_ratios is not None:
        metrics["progress_metrics"] = finalize_progress_stats(progress_stats)
    if collect_predictions:
        metrics["predictions"] = all_preds
        metrics["labels"] = all_labels
        metrics["gate_scores"] = all_gate_scores
    return metrics


def gating_lambda_for_epoch(args: argparse.Namespace, epoch: int) -> float:
    if args.reg_type == "none":
        return 0.0
    if epoch <= args.gate_reg_warmup_epochs:
        return 0.0
    if args.gate_reg_ramp_epochs > 0:
        progress = (epoch - args.gate_reg_warmup_epochs) / max(1, args.gate_reg_ramp_epochs)
        return float(args.lambda_reg) * float(min(1.0, max(0.0, progress)))
    return float(args.lambda_reg)


def lr_schedule_lambda(args: argparse.Namespace, epoch_idx: int) -> float:
    if epoch_idx < args.warmup_epochs:
        return (epoch_idx + 1) / max(1, args.warmup_epochs)
    progress = (epoch_idx - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
    return 0.5 * (1 + np.cos(np.pi * progress))


def build_selection_score(metrics: Dict[str, object]) -> float:
    return 0.6 * float(metrics["average_accuracy"]) + 0.4 * float(metrics["early_online_auc"])


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
    selection_ratios = parse_ratio_list(args.selection_ratios)
    online_eval_ratios = parse_ratio_list(args.online_eval_ratios)
    print(f"device: {device}")
    print(f"block_modality: {args.block_modality}")
    print(
        f"streaming: chunk_size={args.chunk_size}, max_tactile_len={args.max_tactile_len}, "
        f"selection_ratios={selection_ratios}"
    )
    print(
        f"gating regularization: {args.reg_type} with weight {args.lambda_reg} "
        f"(warmup={args.gate_reg_warmup_epochs}, ramp={args.gate_reg_ramp_epochs}, target_mean={args.gate_target_mean})"
    )

    train_loader = build_loader(train_dir, args.batch_size, args.max_tactile_len, args.num_workers, shuffle=True)
    val_loader = build_loader(val_dir, args.batch_size, args.max_tactile_len, args.num_workers, shuffle=False)
    print(f"train samples: {len(train_loader.dataset)} | val samples: {len(val_loader.dataset)}")

    args.mass_classes = len(train_loader.dataset.mass_to_idx)
    args.stiffness_classes = len(train_loader.dataset.stiffness_to_idx)
    args.material_classes = len(train_loader.dataset.material_to_idx)

    model = build_model(
        cfg=vars(args),
        args=args,
        mass_classes=args.mass_classes,
        stiffness_classes=args.stiffness_classes,
        material_classes=args.material_classes,
    ).to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch_idx: lr_schedule_lambda(args, epoch_idx),
    )

    history = []
    best_score = -1.0
    best_epoch = -1
    early_stop_streak = 0
    live_plotter = None
    if args.live_plot:
        try:
            live_plotter = LiveTrainingPlotter(save_dir)
        except Exception as exc:
            print(f"live plot disabled: {exc}")

    for epoch in range(1, args.epochs + 1):
        lambda_reg_eff = gating_lambda_for_epoch(args, epoch)
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            args=args,
            train_mode=True,
            optimizer=optimizer,
            lambda_reg=lambda_reg_eff,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            args=args,
            train_mode=False,
            lambda_reg=lambda_reg_eff,
            selection_ratios=selection_ratios,
            progress_ratios=online_eval_ratios,
        )
        scheduler.step()

        selection_score = build_selection_score(val_metrics)
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "selection_score": selection_score,
            }
        )
        if live_plotter is not None:
            live_plotter.update(epoch, train_metrics, val_metrics)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} | "
            f"val_mass={val_metrics['mass']:.2%} val_stiff={val_metrics['stiffness']:.2%} "
            f"val_mat={val_metrics['material']:.2%} | "
            f"val_early_auc={val_metrics['early_online_auc']:.2%} score={selection_score:.4f}"
        )

        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": vars(args),
                    "val_metrics": val_metrics,
                    "selection_score": selection_score,
                },
                save_dir / "best_model.pth",
            )

        if args.early_stop_patience > 0:
            if selection_score >= args.early_stop_score:
                early_stop_streak += 1
            else:
                early_stop_streak = 0
            if epoch >= args.early_stop_min_epoch and early_stop_streak >= args.early_stop_patience:
                print(
                    f"early stop: selection_score={selection_score:.4f} for {early_stop_streak} epochs "
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
    print(f"best epoch: {best_epoch} | best selection score: {best_score:.4f}")

    for split_name in ["test", "ood_test"]:
        split_dir = data_root / split_name
        if split_dir.is_dir():
            metrics = eval_split(args, split_name=split_name, checkpoint_path=save_dir / "best_model.pth")
            print(
                f"{split_name}: loss={metrics['loss']:.4f}, "
                f"mass={metrics['mass']:.2%}, stiffness={metrics['stiffness']:.2%}, "
                f"material={metrics['material']:.2%}, avg_g={metrics['avg_gate_score']:.3f}"
            )


def load_model_for_eval(
    args: argparse.Namespace,
    split_dir: Path,
    checkpoint_path: Path,
) -> Tuple[StreamingFusionGatingTransformer, Dict[str, object], object, torch.device]:
    device = resolve_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})
    loader = build_loader(split_dir, args.batch_size, cfg.get("max_tactile_len", args.max_tactile_len), args.num_workers, shuffle=False)
    dataset = loader.dataset

    model = build_model(
        cfg=cfg,
        args=args,
        mass_classes=len(dataset.mass_to_idx),
        stiffness_classes=len(dataset.stiffness_to_idx),
        material_classes=len(dataset.material_to_idx),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint, loader, device


def eval_split(args: argparse.Namespace, split_name: str, checkpoint_path: Optional[Path] = None) -> Dict[str, object]:
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

    model, checkpoint, loader, device = load_model_for_eval(args, split_dir, ckpt_path)
    dataset = loader.dataset
    cfg = checkpoint.get("config", {})
    metrics = run_epoch(
        model=model,
        loader=loader,
        device=device,
        args=args,
        train_mode=False,
        lambda_reg=float(cfg.get("lambda_reg", args.lambda_reg)),
        collect_predictions=True,
    )

    label_names = {
        "mass": list(dataset.mass_to_idx.keys()),
        "stiffness": list(dataset.stiffness_to_idx.keys()),
        "material": list(dataset.material_to_idx.keys()),
    }
    results = {}
    for task in TASKS:
        preds = metrics["predictions"][task]
        labels = metrics["labels"][task]
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
        _plot_confusion_matrix(metrics["labels"][task], metrics["predictions"][task], label_names[task], task, output_dir)
    _plot_summary(results, output_dir)

    full_result = {
        "split": split_name,
        "block_modality": args.block_modality,
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "selection_score": checkpoint.get("selection_score"),
        "num_samples": len(dataset),
        "loss": float(metrics["loss"]),
        "mass": float(metrics["mass"]),
        "stiffness": float(metrics["stiffness"]),
        "material": float(metrics["material"]),
        "avg_gate_score": float(np.mean(metrics["gate_scores"])) if metrics["gate_scores"] else 0.0,
        "summary": {
            "average_accuracy": avg_acc,
            "average_macro_f1": avg_macro_f1,
            "average_weighted_f1": avg_weighted_f1,
        },
        "gate_scores": metrics["gate_scores"],
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

    ratios = parse_ratio_list(args.online_eval_ratios)
    model, checkpoint, loader, device = load_model_for_eval(args, split_dir, ckpt_path)
    cfg = checkpoint.get("config", {})
    metrics = run_epoch(
        model=model,
        loader=loader,
        device=device,
        args=args,
        train_mode=False,
        lambda_reg=float(cfg.get("lambda_reg", args.lambda_reg)),
        progress_ratios=ratios,
    )

    curves = []
    for ratio in ratios:
        progress = metrics["progress_metrics"][ratio]
        curves.append(
            {
                "prefix_ratio": float(ratio),
                "gate_score": float(progress["gate_score"]),
                "mass": float(progress["mass"]),
                "stiffness": float(progress["stiffness"]),
                "material": float(progress["material"]),
                "average_accuracy": float(progress["average_accuracy"]),
                "num_samples": int(progress["num_samples"]),
            }
        )

    result = {
        "split": split_name,
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "selection_score": checkpoint.get("selection_score"),
        "num_samples": len(loader.dataset),
        "prefix_curves": curves,
    }

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        suffix = "" if args.block_modality == "none" else f"_block_{args.block_modality}"
        output_dir = ckpt_path.parent / f"online_eval_{split_name}{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "online_evaluation_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict streaming gating transformer for visuotactile fusion")
    parser.add_argument("--mode", choices=["train", "eval", "online_eval"], default="train")
    parser.add_argument("--data_root", type=str, default="/home/martina/Y3_Project/Plaintextdataset")
    parser.add_argument("--save_dir", type=str, default="outputs/fusion_gating_stream_transformer")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--eval_split", choices=["val", "test", "ood_test"], default="test")
    parser.add_argument("--output_dir", type=str, default="")
    parser.set_defaults(live_plot=True)
    parser.add_argument("--live_plot", dest="live_plot", action="store_true")
    parser.add_argument("--no_live_plot", dest="live_plot", action="store_false")
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
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--window_size", type=int, default=0)
    parser.add_argument("--step_size", type=int, default=0)
    parser.set_defaults(freeze_visual=True)
    parser.add_argument("--freeze_visual", dest="freeze_visual", action="store_true")
    parser.add_argument("--unfreeze_visual", dest="freeze_visual", action="store_false")
    parser.set_defaults(pretrained_visual=True)
    parser.add_argument("--pretrained_visual", dest="pretrained_visual", action="store_true")
    parser.add_argument("--no_pretrained_visual", dest="pretrained_visual", action="store_false")
    parser.add_argument("--visual_drop_prob", type=float, default=0.0)
    parser.add_argument("--tactile_drop_prob", type=float, default=0.0)
    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--early_stop_score", type=float, default=1.0)
    parser.add_argument("--early_stop_min_epoch", type=int, default=0)

    parser.add_argument("--lambda_reg", type=float, default=0.1)
    parser.add_argument("--lambda_aux", type=float, default=0.5)
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

    parser.add_argument(
        "--selection_ratios",
        type=str,
        default="0.25,0.5,0.75",
        help="Progress ratios used for early online checkpoint selection",
    )
    parser.add_argument(
        "--online_eval_ratios",
        type=str,
        default="0.25,0.5,0.75,1.0",
        help="Progress ratios reported by online_eval",
    )

    args = parser.parse_args()
    args.chunk_size = resolve_chunk_size(args)
    return args


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
        print(json.dumps(result, indent=2, ensure_ascii=False))
