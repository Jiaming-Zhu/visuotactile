"""
视触觉融合门控网络在线训练脚本 - 带可靠检查点选择和视觉错配门控监督

该脚本实现了一个多模态融合模型（视觉+触觉），用于预测物体的物理属性（质量、刚度、材质）。
核心创新点：
1. 在线前缀训练：模拟不完整触觉序列的推理场景
2. 门控机制：学习动态融合视觉和触觉信息的权重
3. 可靠检查点选择：在验证集上选择泛化能力最好的模型
4. 视觉错配监督：增强模型对视觉干扰的鲁棒性
"""

import argparse
import json
import math
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
    # tqdm用于显示训练进度条，如果不可用则设为None
    tqdm = None

# 尝试从相对路径导入（开发模式），失败则从包导入（安装模式）
try:
    from train_fusion_gating_online import (
        TASKS,  # 任务列表：["mass", "stiffness", "material"]
        SupervisedContrastiveLoss,  # 有监督对比损失
        build_model,  # 模型构建函数
        effective_padding_mask,  # 计算有效前缀掩码
        eval_split,  # 评估分割数据集
        online_eval_split,  # 在线评估（不同前缀长度）
    )
    from train_fusion_gating2 import (
        LiveTrainingPlotter,  # 实时训练曲线绘制器
        _plot_training_curves,  # 绘制训练曲线
        apply_modality_block,  # 应用模态阻塞（训练时随机丢弃某模态）
        apply_modality_dropout,  # 应用模态Dropout
        build_loader,  # 构建数据加载器
        resolve_device,  # 解析设备（cuda/cpu）
        set_seed,  # 设置随机种子
    )
except ImportError:  # pragma: no cover
    from visuotactile.scripts.train_fusion_gating_online import (  # type: ignore
        TASKS,
        SupervisedContrastiveLoss,
        build_model,
        effective_padding_mask,
        eval_split,
        online_eval_split,
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


def compute_gate_regularization(
    gate_score: torch.Tensor,
    reg_type: str,
    gate_target_mean: float,
    gate_entropy_eps: float,
) -> torch.Tensor:
    """
    计算门控分数的正则化损失

    门控分数gate_score是一个(0,1)区间内的值，表示触觉信息在融合中的权重。
    正则化的目的是引导门控分数达到期望的分布状态。

    参数:
        gate_score: 门控分数，形状为 (batch_size,)，每个样本一个分数
        reg_type: 正则化类型，不同类型鼓励不同的门控行为
        gate_target_mean: 目标均值（用于mean类型）
        gate_entropy_eps: 熵计算中的clamping epsilon，防止log(0)

    返回:
        正则化损失标量

    正则化类型解释:
        - polarization（极化）: 鼓励gate_score趋向0或1（极端决策），通过g*(1-g)实现
          原理：g*(1-g)在g=0.5时最大，在g=0或1时最小
        - sparsity（稀疏性）: 鼓励整体较低的gate分数，即更依赖视觉
        - mean（均值）: 鼓励gate分数的均值接近目标值
        - center（居中）: 鼓励gate分数接近0.5
        - entropy（熵）: 鼓励门控的不确定性最大，即均匀分布
    """
    if reg_type == "polarization":
        # g*(1-g) 在 g=0.5 时最大(0.25)，g=0或1时最小(0)
        # 最小化这个值意味着鼓励极端化（0或1）
        return (gate_score * (1.0 - gate_score)).mean()

    if reg_type == "sparsity":
        # 直接惩罚gate分数的均值，鼓励稀疏（低值）
        return gate_score.mean()

    if reg_type == "mean":
        # 惩罚与目标均值的偏差平方
        return (gate_score.mean() - gate_target_mean).pow(2)

    if reg_type == "center":
        # 鼓励gate分数接近0.5（平衡点）
        return (gate_score - 0.5).pow(2).mean()

    if reg_type == "entropy":
        # 最大熵正则化：鼓励门控分布接近均匀分布
        # H(g) = -g*log(g) - (1-g)*log(1-g)，最大值在g=0.5时
        # log(2) - entropy 使得损失始终为正
        g_clamped = torch.clamp(gate_score, gate_entropy_eps, 1.0 - gate_entropy_eps)
        entropy = -(
            g_clamped * torch.log(g_clamped)
            + (1.0 - g_clamped) * torch.log(1.0 - g_clamped)
        ).mean()
        return gate_score.new_tensor(math.log(2.0)) - entropy

    # "none"类型，返回零损失
    return gate_score.new_zeros(())


def compute_clean_losses(
    outputs: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    criterion: nn.Module,
    supcon_criterion: Optional[nn.Module],
    lambda_reg: float,
    lambda_aux: float,
    lambda_supcon: float,
    supcon_task: str,
    reg_type: str,
    gate_target_mean: float,
    gate_entropy_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算所有损失分量并合并为总损失

    损失组成:
    1. ce_loss（主损失）: 三个任务的交叉熵损失之和
    2. reg_loss（门控正则化）: 鼓励门控分数达到期望分布
    3. aux_loss（辅助损失）: 辅助头（aux_*）的损失，提供正则化效果
    4. supcon_loss（对比损失）: 有监督对比学习损失，同类样本在embedding空间接近

    参数:
        outputs: 模型输出字典，包含主输出和辅助输出
        labels: 标签字典
        criterion: 交叉熵损失函数
        supcon_criterion: 对比损失函数（可选）
        lambda_*: 各损失分量的权重

    返回:
        (total_loss, reg_loss, supcon_loss, aux_loss) 元组
    """
    # 主任务损失：对三个任务（质量、刚度、材质）分别计算交叉熵并求和
    ce_loss = (
        criterion(outputs["mass"], labels["mass"])
        + criterion(outputs["stiffness"], labels["stiffness"])
        + criterion(outputs["material"], labels["material"])
    )

    # 门控正则化损失
    reg_loss = compute_gate_regularization(
        gate_score=outputs["gate_score"],
        reg_type=reg_type,
        gate_target_mean=gate_target_mean,
        gate_entropy_eps=gate_entropy_eps,
    )

    # 辅助损失：辅助头预测相同任务，提供额外的学习信号
    # 这是一种类似知识蒸馏的技巧，帮助主任务学习
    aux_loss = (
        criterion(outputs["aux_mass"], labels["mass"])
        + criterion(outputs["aux_stiffness"], labels["stiffness"])
        + criterion(outputs["aux_material"], labels["material"])
    )

    # 有监督对比损失：同类别样本的embedding应该接近
    # 仅当启用且权重>0时计算
    if supcon_criterion is not None and lambda_supcon > 0.0:
        supcon_loss = supcon_criterion(outputs["contrastive_embedding"], labels[supcon_task])
    else:
        supcon_loss = ce_loss.new_zeros(())

    # 总损失 = 主损失 + λ_reg * 正则化 + λ_aux * 辅助损失 + λ_supcon * 对比损失
    total_loss = ce_loss + lambda_reg * reg_loss + lambda_aux * aux_loss + lambda_supcon * supcon_loss
    return total_loss, reg_loss, supcon_loss, aux_loss


def build_label_signature(labels: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    构建标签签名：将多个标签任务合并为一个唯一标识符

    原理：将三个任务的标签组合成一个唯一的整数签名
    例如：mass=1, stiffness=2, material=3 -> 1*100 + 2*10 + 3 = 123

    用途：在视觉错配批构建中，用于快速判断两个样本是否属于同一"类别组合"
    这样可以确保错配样本一定来自不同的真实标签组合

    参数:
        labels: 标签字典

    返回:
        形状为 (batch_size,) 的签名张量
    """
    return labels["mass"] * 100 + labels["stiffness"] * 10 + labels["material"]


def build_visual_mismatch_batch(
    images: torch.Tensor,
    labels: Dict[str, torch.Tensor],
    mismatch_prob: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    构建视觉错配批次：创建图像与标签不匹配的样本

    核心思想：通过故意错配图像和标签，来训练模型识别视觉信息的可靠性。
    当图像被替换为不匹配的图像时，门控应该降低对视觉的依赖（降低gate_score）。

    原理：
    1. 随机选择一批样本的图像进行替换
    2. 替换时从不同标签组合的样本中选择图像
    3. 返回修改后的图像批次和错配掩码

    参数:
        images: 原始图像张量 (batch_size, C, H, W)
        labels: 标签字典
        mismatch_prob: 每个样本被错配的概率

    返回:
        (mismatch_images, mismatch_mask): 错配后的图像和掩码
        mismatch_mask[i]=True 表示第i个样本的图像被替换了
    """
    batch_size = images.size(0)
    device = images.device

    # 边界情况处理
    if mismatch_prob <= 0.0 or batch_size < 2:
        # 不进行任何错配
        return images, torch.zeros(batch_size, dtype=torch.bool, device=device)

    # 全错配情况（测试用）
    if mismatch_prob >= 1.0:
        mismatch_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    else:
        # 伯努利采样决定每个样本是否错配
        mismatch_mask = torch.rand(batch_size, device=device) < mismatch_prob
        # 确保至少有一个样本被选中，否则返回原图像
        if not mismatch_mask.any():
            return images, mismatch_mask

    # 构建标签签名用于快速匹配判断
    signatures = build_label_signature(labels).detach().cpu().tolist()
    target_indices = mismatch_mask.nonzero(as_tuple=False).flatten().tolist()
    source_indices = []  # 存储提供图像的源索引
    valid_targets = []   # 存储真正需要错配的目标索引
    all_indices = list(range(batch_size))

    # 为每个需要错配的目标样本找一个源样本
    for target_idx in target_indices:
        # 优先选择标签组合不同的候选（确保真正的错配）
        diff_candidates = [idx for idx in all_indices if idx != target_idx and signatures[idx] != signatures[target_idx]]
        if not diff_candidates:
            # 如果没有不同签名的，退而求其次选择任意不同样本
            diff_candidates = [idx for idx in all_indices if idx != target_idx]
        if not diff_candidates:
            continue
        # 随机选择一个源样本
        rand_pos = torch.randint(len(diff_candidates), (1,), device=device).item()
        valid_targets.append(target_idx)
        source_indices.append(diff_candidates[rand_pos])

    # 没有任何有效错配对
    if not source_indices:
        return images, torch.zeros(batch_size, dtype=torch.bool, device=device)

    # 执行错配：克隆原图像，然后用源图像替换目标位置
    mismatch_images = images.clone()
    target_tensor = torch.tensor(valid_targets, dtype=torch.long, device=device)
    source_tensor = torch.tensor(source_indices, dtype=torch.long, device=device)
    mismatch_images[target_tensor] = images[source_tensor]

    # 构建有效掩码：只有真正进行了错配的位置为True
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
    enable_mismatch_supervision: bool = False,
) -> Dict[str, float]:
    """
    计算训练或验证指标

    该函数是训练/验证的核心循环，完成：
    1. 数据加载与预处理
    2. 前缀掩码计算（在线训练的关键）
    3. 模态阻塞和Dropout（训练增强）
    4. 前向传播与损失计算
    5. 视觉错配监督（可选）
    6. 梯度反向传播与参数更新（仅训练模式）
    7. 指标累积与返回

    参数:
        model: 融合门控模型
        loader: 数据加载器
        criterion: 损失函数
        device: 计算设备
        args: 命令行参数
        train_mode: 是否为训练模式（决定是否更新参数）
        optimizer: 优化器（训练模式必需）
        lambda_reg: 门控正则化权重（动态调整）
        fixed_prefix_ratio: 固定前缀比例（用于评估）
        enable_mismatch_supervision: 是否启用视觉错配监督

    返回:
        包含所有指标的字典
    """
    # 设置模型模式：训练模式启用dropout和batchnorm更新
    if train_mode:
        model.train()
    else:
        model.eval()

    # 初始化累积变量
    total_loss = 0.0
    total_reg = 0.0
    total_supcon = 0.0
    total_aux = 0.0
    total_gate = 0.0
    total_samples = 0
    correct = {task: 0 for task in TASKS}

    # 视觉错配相关统计
    total_mismatch_gate_loss = 0.0
    total_mismatch_gate = 0.0
    total_clean_gate_on_mismatch = 0.0
    total_mismatch_samples = 0

    # 有监督对比损失构建器（仅当启用时）
    supcon_criterion = None
    if args.lambda_supcon > 0.0:
        supcon_criterion = SupervisedContrastiveLoss(temperature=args.supcon_temperature)

    # 迭代器设置（可选tqdm进度条）
    iterator = loader
    if tqdm is not None:
        desc = "train" if train_mode else ("prefix_eval" if fixed_prefix_ratio is not None else "eval")
        iterator = tqdm(loader, leave=False, desc=desc)

    # 批次迭代
    for batch_idx, batch in enumerate(iterator, start=1):
        # 数据移到设备
        images = batch["image"].to(device)
        tactile = batch["tactile"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        labels = {task: batch[task].to(device) for task in TASKS}

        # 计算有效前缀掩码
        # 这是实现在线训练的核心：随机截断触觉序列，模拟不完整观测
        prefix_mask = effective_padding_mask(
            padding_mask=padding_mask,
            train_mode=train_mode,
            online_train_prob=args.online_train_prob,  # 随机前缀的概率
            online_min_prefix_ratio=args.online_min_prefix_ratio,  # 最小前缀比例
            min_prefix_len=args.min_prefix_len,  # 最小前缀长度（原始时间步）
            fixed_ratio=fixed_prefix_ratio,  # 固定比例（评估时使用）
        )

        # 模态阻塞：在某些批次中完全丢弃某一模态，增强鲁棒性
        images, tactile = apply_modality_block(images, tactile, args.block_modality)

        # 模态Dropout：独立地随机丢弃某些样本的视觉或触觉信息
        if train_mode:
            images, tactile = apply_modality_dropout(
                images,
                tactile,
                visual_drop_prob=args.visual_drop_prob,
                tactile_drop_prob=args.tactile_drop_prob,
            )

        # 初始化错配相关变量
        mismatch_gate_loss = images.new_zeros(())
        mismatch_gate_score_sum = 0.0
        clean_gate_on_mismatch_sum = 0.0
        mismatch_sample_count = 0

        # ==================== 训练模式 ====================
        if train_mode:
            if optimizer is None:
                raise ValueError("optimizer is required when train_mode=True")

            optimizer.zero_grad()  # 清除上一步的梯度

            # 前向传播
            outputs = model(images, tactile, padding_mask=prefix_mask)

            # 计算干净样本的损失
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

            # 视觉错配监督：增强模型对视觉错配的识别能力
            # 原理：当视觉信息不可靠时，门控应该自动降低对视觉的依赖
            if enable_mismatch_supervision and args.lambda_mismatch_gate > 0.0 and args.visual_mismatch_prob > 0.0:
                # 创建错配批次
                mismatch_images, mismatch_mask = build_visual_mismatch_batch(
                    images=images,
                    labels=labels,
                    mismatch_prob=args.visual_mismatch_prob,
                )
                if mismatch_mask.any():
                    # 对错配图像进行前向传播
                    mismatch_outputs = model(mismatch_images, tactile, padding_mask=prefix_mask)
                    # 错配样本的门控分数应该接近0（完全不依赖视觉）
                    mismatch_gate = mismatch_outputs["gate_score"][mismatch_mask]
                    mismatch_gate_loss = F.binary_cross_entropy(mismatch_gate, torch.zeros_like(mismatch_gate))
                    # 将错配损失加入到总目标
                    total_objective = total_objective + args.lambda_mismatch_gate * mismatch_gate_loss
                    mismatch_sample_count = int(mismatch_mask.sum().item())
                    mismatch_gate_score_sum = mismatch_gate.sum().item()
                    # 记录干净图像通过同一模型时的门控分数（作为对比）
                    clean_gate_on_mismatch_sum = outputs["gate_score"][mismatch_mask].sum().item()

            # 反向传播
            total_objective.backward()
            # 梯度裁剪：防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # 参数更新
            optimizer.step()
            loss = total_objective

        # ==================== 验证模式 ====================
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

        # 累积指标
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_reg += reg_loss.item() * batch_size
        total_supcon += supcon_loss.item() * batch_size
        total_aux += aux_loss.item() * batch_size
        total_gate += outputs["gate_score"].sum().item()
        total_samples += batch_size

        # 错配指标累积
        total_mismatch_gate_loss += mismatch_gate_loss.item() * max(1, mismatch_sample_count)
        total_mismatch_gate += mismatch_gate_score_sum
        total_clean_gate_on_mismatch += clean_gate_on_mismatch_sum
        total_mismatch_samples += mismatch_sample_count

        # 计算每个任务的准确率
        for task in TASKS:
            correct[task] += (outputs[task].argmax(dim=1) == labels[task]).sum().item()

        # 更新进度条显示
        if tqdm is not None and hasattr(iterator, "set_postfix"):
            postfix = {
                "loss": f"{total_loss / max(1, total_samples):.4f}",
                "mass": f"{correct['mass'] / max(1, total_samples):.2%}",
                "stiff": f"{correct['stiffness'] / max(1, total_samples):.2%}",
                "mat": f"{correct['material'] / max(1, total_samples):.2%}",
                "g": f"{total_gate / max(1, total_samples):.3f}",  # 平均门控分数
                "supcon": f"{total_supcon / max(1, total_samples):.4f}",
                "step": batch_idx,
            }
            if total_mismatch_samples > 0:
                postfix["g_mis"] = f"{total_mismatch_gate / total_mismatch_samples:.3f}"
            iterator.set_postfix(postfix)

    # 计算最终指标
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
        "mismatch_gate_loss": total_mismatch_gate_loss / max(1, total_mismatch_samples),
        "mismatch_gate_score": mismatch_gate_mean if total_mismatch_samples > 0 else 0.0,
        "clean_gate_on_mismatch": clean_gate_on_mismatch_mean if total_mismatch_samples > 0 else 0.0,
        # 差距：干净图像门控 - 错配图像门控，差距越大说明模型越能区分视觉可靠性
        "mismatch_gate_gap": (
            clean_gate_on_mismatch_mean - mismatch_gate_mean if total_mismatch_samples > 0 else 0.0
        ),
        "mismatch_samples": float(total_mismatch_samples),
    }


def compute_mismatch_gate_metrics(
    model: nn.Module,
    loader,
    device: torch.device,
    args: argparse.Namespace,
    fixed_prefix_ratio: Optional[float] = None,
) -> Dict[str, float]:
    """
    计算视觉错配门控指标

    与compute_metrics中的错配监督不同，该函数专门评估模型对视觉错配的敏感性：
    1. 对每个样本，随机替换其图像（确保来自不同标签组合）
    2. 计算错配图像的门控分数应该接近0
    3. 同时计算干净图像的门控分数作为基准

    这些指标用于衡量"可靠性"——模型能否正确识别视觉信息不可靠的情况。

    参数:
        model: 模型
        loader: 数据加载器
        device: 设备
        args: 参数
        fixed_prefix_ratio: 固定前缀比例

    返回:
        错配相关指标的字典
    """
    model.eval()
    total_gate_loss = 0.0
    total_mismatch_gate = 0.0
    total_clean_gate = 0.0
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

            # 计算前缀掩码（验证模式使用确定性设置）
            prefix_mask = effective_padding_mask(
                padding_mask=padding_mask,
                train_mode=False,
                online_train_prob=0.0,  # 验证时使用完整序列
                online_min_prefix_ratio=args.online_min_prefix_ratio,
                min_prefix_len=args.min_prefix_len,
                fixed_ratio=fixed_prefix_ratio,
            )

            images, tactile = apply_modality_block(images, tactile, args.block_modality)

            # 创建视觉错配批次
            mismatch_images, mismatch_mask = build_visual_mismatch_batch(
                images=images,
                labels=labels,
                mismatch_prob=args.visual_mismatch_eval_prob,  # 评估时使用更高概率
            )
            if not mismatch_mask.any():
                continue

            # 分别对干净图像和错配图像进行前向传播
            clean_outputs = model(images, tactile, padding_mask=prefix_mask)
            mismatch_outputs = model(mismatch_images, tactile, padding_mask=prefix_mask)

            # 提取错配掩码对应的门控分数
            clean_gate = clean_outputs["gate_score"][mismatch_mask]
            mismatch_gate = mismatch_outputs["gate_score"][mismatch_mask]

            # BCE损失：错配图像的门控分数应该接近0
            total_gate_loss += F.binary_cross_entropy(
                mismatch_gate,
                torch.zeros_like(mismatch_gate),
                reduction="sum",
            ).item()
            total_clean_gate += clean_gate.sum().item()
            total_mismatch_gate += mismatch_gate.sum().item()
            total_samples += mismatch_gate.numel()

    # 计算均值
    mismatch_gate_mean = total_mismatch_gate / max(1, total_samples)
    clean_gate_mean = total_clean_gate / max(1, total_samples)

    return {
        "gate_loss": total_gate_loss / max(1, total_samples),
        "gate_score": mismatch_gate_mean if total_samples > 0 else 0.0,
        "clean_gate_score": clean_gate_mean if total_samples > 0 else 0.0,
        # 干净门控 - 错配门控：正值越大表示模型越能识别视觉不可靠的情况
        "clean_minus_mismatch_gap": clean_gate_mean - mismatch_gate_mean if total_samples > 0 else 0.0,
        "num_samples": float(total_samples),
    }


def reliable_selection_start_epoch(args: argparse.Namespace) -> int:
    """
    确定可靠检查点选择的起始epoch

    原因：在训练早期，模型还在快速学习阶段，门控机制尚未稳定
    过早的检查点选择可能导致选择到泛化能力差的模型
    因此设置一个"冷静期"，等模型稳定后再开始选择

    规则：
    1. 如果手动指定了reliable_selection_start_epoch，使用该值
    2. 如果没有使用正则化（reg_type="none"），从epoch 1开始选择
    3. 否则，从warmup结束后开始选择
    """
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
    val_mismatch_metrics: Dict[str, float],
    avg_val_acc: float,
    lambda_reg_eff: float,
    role: str,
) -> None:
    """
    保存模型检查点

    检查点包含：
    - epoch: 当前训练的epoch
    - model_state_dict: 模型参数
    - optimizer_state_dict: 优化器状态（用于恢复训练）
    - scheduler_state_dict: 学习率调度器状态
    - config: 训练配置（命令行参数）
    - val_metrics: 验证指标
    - val_mismatch_metrics: 错配验证指标
    - avg_val_acc: 平均验证准确率
    - lambda_reg_eff: 有效的正则化权重
    - checkpoint_role: 检查点角色（"best_acc" 或 "best_reliable"）
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": vars(args),
            "val_metrics": val_metrics,
            "val_mismatch_metrics": val_mismatch_metrics,
            "avg_val_acc": avg_val_acc,
            "lambda_reg_eff": lambda_reg_eff,
            "checkpoint_role": role,
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    """
    主训练函数

    训练流程：
    1. 数据加载器构建
    2. 模型、优化器、学习率调度器初始化
    3. 训练循环（多个epoch）：
       a. 计算当前有效的正则化权重（warmup + ramp）
       b. 训练集前向传播+反向传播
       c. 验证集评估（包括错配评估）
       d. 学习率更新
       e. 保存最佳检查点
       f. 早停检查
    4. 训练后处理：保存最终模型、绘制曲线
    5. 在测试集和OOD测试集上评估
    """
    data_root = Path(args.data_root)
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(f"Expected both train/ and val/ under {data_root}")

    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed, args.device)

    device = resolve_device(args.device)
    reliable_epoch_start = reliable_selection_start_epoch(args)

    # 打印训练配置
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
        f"mismatch supervision: prob={args.visual_mismatch_prob}, "
        f"lambda={args.lambda_mismatch_gate}, eval_prob={args.visual_mismatch_eval_prob}"
    )
    print(
        f"reliable checkpoint selection starts at epoch {reliable_epoch_start}, "
        f"primary={args.primary_checkpoint}"
    )
    print(
        f"supcon: weight={args.lambda_supcon}, task={args.supcon_task}, "
        f"temperature={args.supcon_temperature}"
    )

    # 构建数据加载器
    train_loader = build_loader(train_dir, args.batch_size, args.max_tactile_len, args.num_workers, shuffle=True)
    val_loader = build_loader(val_dir, args.batch_size, args.max_tactile_len, args.num_workers, shuffle=False)
    print(f"train samples: {len(train_loader.dataset)} | val samples: {len(val_loader.dataset)}")

    # 从数据集获取类别数量
    args.mass_classes = len(train_loader.dataset.mass_to_idx)
    args.stiffness_classes = len(train_loader.dataset.stiffness_to_idx)
    args.material_classes = len(train_loader.dataset.material_to_idx)

    # 构建模型
    model = build_model(
        vars(args),
        args,
        args.mass_classes,
        args.stiffness_classes,
        args.material_classes,
    ).to(device)

    # 损失函数：交叉熵
    criterion = nn.CrossEntropyLoss()

    # 优化器：AdamW（带权重衰减的Adam）
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # 学习率调度器：Cosine Annealing with Warmup
    # 原理：先线性 warmup 让优化器稳定，然后余弦退火缓慢降低学习率
    def lr_lambda(epoch_idx: int) -> float:
        if epoch_idx < args.warmup_epochs:
            # Warmup阶段：线性增长
            return (epoch_idx + 1) / max(1, args.warmup_epochs)
        # 余弦退火阶段
        progress = (epoch_idx - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 训练历史记录
    history = []

    # 最佳指标跟踪
    best_acc_value = -1.0
    best_acc_epoch = -1
    best_reliable_score = None
    best_reliable_epoch = -1
    best_reliable_avg = -1.0
    early_stop_streak = 0

    # 实时绘图器（可选）
    live_plotter = None
    if args.live_plot:
        try:
            live_plotter = LiveTrainingPlotter(save_dir)
        except Exception as exc:
            print(f"live plot disabled: {exc}")

    # ==================== 训练循环 ====================
    for epoch in range(1, args.epochs + 1):
        # 计算有效的正则化权重
        # 目的：在warmup期间禁用正则化，让模型先学习；之后逐渐增加正则化强度
        if args.reg_type == "none":
            lambda_reg_eff = 0.0
        elif epoch <= args.gate_reg_warmup_epochs:
            # Warmup期间：关闭正则化
            lambda_reg_eff = 0.0
        elif args.gate_reg_ramp_epochs > 0:
            # Ramp期间：线性增加正则化权重
            progress = (epoch - args.gate_reg_warmup_epochs) / max(1, args.gate_reg_ramp_epochs)
            lambda_reg_eff = float(args.lambda_reg) * float(min(1.0, max(0.0, progress)))
        else:
            # Ramp结束后：使用完整权重
            lambda_reg_eff = float(args.lambda_reg)

        # 训练集评估（包含梯度更新）
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

        # 验证集评估（无梯度更新）
        val_metrics = compute_metrics(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            args=args,
            train_mode=False,
            lambda_reg=lambda_reg_eff,
        )

        # 验证集错配评估
        val_mismatch_metrics = compute_mismatch_gate_metrics(
            model=model,
            loader=val_loader,
            device=device,
            args=args,
        )

        # 学习率调度器更新
        scheduler.step()

        # 计算平均验证准确率
        avg_val = float(np.mean([val_metrics[task] for task in TASKS]))
        best_acc_updated = False
        best_reliable_updated = False

        # ==================== 保存最佳准确率检查点 ====================
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

        # ==================== 保存最佳可靠性检查点 ====================
        # 可靠性评分是一个多元指标：
        # (avg_val准确率, -gate_loss错配损失, gap差距, -reg_loss正则化, -loss总损失, epoch)
        # 使用元组比较实现多目标优化
        reliable_candidate = epoch >= reliable_epoch_start and val_mismatch_metrics["num_samples"] > 0
        reliable_score = (
            avg_val,                                    # 越高越好
            -val_mismatch_metrics["gate_loss"],         # 错配损失越负越好（实际是负的，所以取反）
            val_mismatch_metrics["clean_minus_mismatch_gap"],  # 差距越大越好
            -val_metrics["reg_loss"],                   # 正则化损失越负越好
            -val_metrics["loss"],                       # 总损失越负越好
            epoch,                                      # 越新越好（作为tie-breaker）
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

        # 记录历史
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

        # 实时更新训练曲线（可选）
        if live_plotter is not None:
            live_plotter.update(epoch, train_metrics, val_metrics)

        # 打印epoch摘要
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} | "
            f"val_mass={val_metrics['mass']:.2%} val_stiff={val_metrics['stiffness']:.2%} "
            f"val_mat={val_metrics['material']:.2%} | val_g={val_metrics['gate_score']:.3f} "
            f"val_mis_g={val_mismatch_metrics['gate_score']:.3f} gap={val_mismatch_metrics['clean_minus_mismatch_gap']:.3f}"
        )

        # ==================== 早停检查 ====================
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

        # ==================== 定期保存检查点 ====================
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

    # ==================== 训练后处理 ====================
    # 保存训练历史
    (save_dir / "training_history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False))
    _plot_training_curves(history, save_dir)
    if live_plotter is not None:
        live_plotter.save()
        live_plotter.close()

    # 根据配置选择最终模型
    # "reliable"模式：优先选择best_reliable（泛化能力更好）
    # "acc"模式：选择best_acc（验证集准确率最高）
    preferred_name = "best_reliable.pth" if args.primary_checkpoint == "reliable" and best_reliable_epoch > 0 else "best_acc.pth"
    preferred_path = save_dir / preferred_name
    if not preferred_path.exists():
        raise FileNotFoundError(f"Preferred checkpoint missing: {preferred_path}")
    # 复制为best_model.pth作为最终模型
    shutil.copyfile(preferred_path, save_dir / "best_model.pth")

    # 保存检查点选择摘要
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

    # 打印最终摘要
    print(f"best acc epoch: {best_acc_epoch} | best val avg acc: {best_acc_value:.2%}")
    if best_reliable_epoch > 0:
        print(f"best reliable epoch: {best_reliable_epoch} | best reliable avg acc: {best_reliable_avg:.2%}")
    else:
        print("best reliable epoch: none")
    print(f"primary checkpoint: {preferred_name}")

    # 在测试集和OOD测试集上评估
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
    """
    解析命令行参数

    参数分类：
    1. 基本配置：模式、数据路径、保存路径、设备
    2. 训练配置：epoch数、batch大小、学习率、优化器参数
    3. 模型配置：融合维度、注意力头数、层数、Dropout率
    4. 门控正则化：正则化类型、权重、warmup/ramp配置
    5. 在线训练：前缀训练概率、最小前缀比例
    6. 视觉错配监督：错配概率、损失权重
    7. 可靠检查点选择：起始epoch、首选模型类型
    """
    parser = argparse.ArgumentParser(
        description="Online-prefix gating fusion with reliable checkpoint selection and visual mismatch gate supervision"
    )

    # 基本配置
    parser.add_argument("--mode", choices=["train", "eval", "online_eval"], default="train")
    parser.add_argument("--data_root", type=str, default="/home/martina/Y3_Project/Plaintextdataset")
    parser.add_argument("--save_dir", type=str, default="outputs/fusion_gating_online_reliable")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--eval_split", choices=["val", "test", "ood_test"], default="test")
    parser.add_argument("--output_dir", type=str, default="")

    # 实时绘图
    parser.set_defaults(live_plot=True)
    parser.add_argument("--live_plot", dest="live_plot", action="store_true", help="Enable live training plots")
    parser.add_argument("--no_live_plot", dest="live_plot", action="store_false", help="Disable live training plots")

    # 模态阻塞
    parser.add_argument(
        "--block_modality",
        type=str,
        default="none",
        choices=["none", "visual", "tactile"],
        help="Block specific modality: none | visual | tactile",
    )

    # 训练配置
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

    # 模型配置
    parser.add_argument("--fusion_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.set_defaults(separate_cls_tokens=False)
    parser.add_argument("--separate_cls_tokens", dest="separate_cls_tokens", action="store_true")
    parser.add_argument("--shared_cls_token", dest="separate_cls_tokens", action="store_false")
    parser.add_argument("--fixed_gate_value", type=float, default=None)
    parser.set_defaults(freeze_visual=True)
    parser.add_argument("--freeze_visual", dest="freeze_visual", action="store_true")
    parser.add_argument("--unfreeze_visual", dest="freeze_visual", action="store_false")

    # 模态Dropout
    parser.add_argument("--visual_drop_prob", type=float, default=0.0)
    parser.add_argument("--tactile_drop_prob", type=float, default=0.0)

    # 早停
    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--early_stop_acc", type=float, default=1.0)
    parser.add_argument("--early_stop_min_epoch", type=int, default=0)

    # 损失权重
    parser.add_argument("--lambda_reg", type=float, default=0.1)
    parser.add_argument("--lambda_aux", type=float, default=0.5)
    parser.add_argument("--lambda_supcon", type=float, default=0.0)
    parser.add_argument("--supcon_temperature", type=float, default=0.07)
    parser.add_argument("--supcon_task", type=str, default="material", choices=list(TASKS))

    # 门控正则化
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

    # 在线训练（前缀训练）
    parser.add_argument("--online_train_prob", type=float, default=1.0, help="Probability of random prefix training")
    parser.add_argument("--online_min_prefix_ratio", type=float, default=0.2, help="Minimum prefix ratio during training")
    parser.add_argument("--min_prefix_len", type=int, default=64, help="Minimum tactile prefix length in raw timesteps")
    parser.add_argument(
        "--prefix_ratios",
        type=str,
        default="0.1,0.2,0.4,0.6,0.8,1.0",
        help="Comma-separated ratios for online_eval",
    )

    # 视觉错配监督
    parser.add_argument(
        "--visual_mismatch_prob",
        type=float,
        default=0.25,
        help="Per-sample probability of creating visual mismatch supervision during training",
    )
    parser.add_argument(
        "--lambda_mismatch_gate",
        type=float,
        default=0.5,
        help="Weight for BCE(g_mismatch, 0) supervision on mismatched visual inputs",
    )
    parser.add_argument(
        "--visual_mismatch_eval_prob",
        type=float,
        default=1.0,
        help="Mismatch probability used when measuring validation reliability metrics",
    )

    # 可靠检查点选择
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

    # 根据模式执行不同操作
    if cli_args.mode == "train":
        # 训练模式：执行完整训练流程
        train(cli_args)
    elif cli_args.mode == "eval":
        # 评估模式：在指定数据集上评估单个检查点
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
        # 在线评估模式：评估不同前缀长度下的模型性能
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
