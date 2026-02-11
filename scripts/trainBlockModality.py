#!/usr/bin/env python3
"""
trainBlockModality.py
=====================

模态屏蔽消融实验脚本 (Modality Blocking Ablation Study)

使用与 train_fusion.py 相同的模型架构，但在训练时可以选择屏蔽某个模态，
用于测试模型在单一输入时的表现和鲁棒性。

Usage:
    # 基准线 (无屏蔽)
    python scripts/trainBlockModality.py --block_modality none
    
    # 纯触觉 (屏蔽视觉)
    python scripts/trainBlockModality.py --block_modality visual
    
    # 纯视觉 (屏蔽触觉)
    python scripts/trainBlockModality.py --block_modality tactile
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
import argparse
import pickle
import time
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ============================================================================
# 触觉数据归一化统计量 (与 train_fusion.py 保持一致)
# ============================================================================
TACTILE_STATS = {
    'joint_position': {'mean': 21.70, 'std': 38.13},
    'joint_load':     {'mean': 7.21,  'std': 14.03},
    'joint_current':  {'mean': 52.56, 'std': 133.43},
    'joint_velocity': {'mean': 0.13,  'std': 9.79},
}


# ============================================================================
# 模型定义 (与 train_fusion.py 完全一致)
# ============================================================================

class FusionModel(nn.Module):
    def __init__(self, fusion_dim=256, num_heads=8, dropout=0.1, num_layers=4, freeze_visual=True):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        self.dropout = dropout

        # Visual Encoder (ResNet18)
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.vis_backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.vis_proj = nn.Conv2d(512, fusion_dim, kernel_size=1)
        
        if freeze_visual:
            for param in self.vis_backbone.parameters():
                param.requires_grad = False
                
        # Tactile Encoder (1D-CNN)
        self.tac_encoder = nn.Sequential(
            nn.Conv1d(24, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, fusion_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(fusion_dim), nn.ReLU()
        )
        
        # Fusion Tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, fusion_dim))
        self.pos_emb = nn.Parameter(torch.randn(1, 425, fusion_dim))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim, 
            nhead=num_heads, 
            dim_feedforward=fusion_dim * 4,
            dropout=dropout, 
            batch_first=True, 
            activation="gelu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Multi-Head Output
        self.head_mass = nn.Sequential(
            nn.Linear(fusion_dim, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 4)
        )
        self.head_stiffness = nn.Sequential(
            nn.Linear(fusion_dim, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 4)
        )
        self.head_material = nn.Sequential(
            nn.Linear(fusion_dim, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 5)
        )

    def forward(self, img, tac, padding_mask=None):
        B = img.shape[0]
        device = img.device
        
        # --- Visual Features ---
        v = self.vis_backbone(img)
        v = self.vis_proj(v)
        v_tokens = v.flatten(2).transpose(1, 2)
        num_vis_tokens = v_tokens.shape[1]

        # --- Tactile Features ---
        t = self.tac_encoder(tac)
        t_tokens = t.transpose(1, 2)
        num_tac_tokens = t_tokens.shape[1]

        # --- Fusion ---
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, v_tokens, t_tokens], dim=1)
        seq_len = x.shape[1]
        x = x + self.pos_emb[:, :seq_len, :]

        # --- Attention Mask ---
        full_mask = None
        if padding_mask is not None:
            tac_mask = padding_mask.float().unsqueeze(1)
            tac_mask = torch.nn.functional.max_pool1d(tac_mask, kernel_size=2, stride=2)
            tac_mask = torch.nn.functional.max_pool1d(tac_mask, kernel_size=2, stride=2)
            tac_mask = torch.nn.functional.max_pool1d(tac_mask, kernel_size=2, stride=2)
            tac_mask = tac_mask.squeeze(1) > 0.5
            tac_mask = tac_mask[:, :num_tac_tokens]
            
            cls_vis_mask = torch.zeros(B, 1 + num_vis_tokens, dtype=torch.bool, device=device)
            full_mask = torch.cat([cls_vis_mask, tac_mask], dim=1)

        x = self.transformer_encoder(x, src_key_padding_mask=full_mask)
        cls_out = x[:, 0, :]
        
        return {
            'mass': self.head_mass(cls_out),
            'stiffness': self.head_stiffness(cls_out),
            'material': self.head_material(cls_out)
        }


# ============================================================================
# 数据集定义 (与 train_fusion.py 保持一致)
# ============================================================================

class RoboticGraspDataset(Dataset):
    def __init__(self, root_dir, properties_file=None, max_tactile_len=3000, transform=None, mode='train'):
        super().__init__()
        self.max_tactile_len = max_tactile_len
        self.mode = mode
        
        # Handle directory structure (train/val)
        mode_dir = os.path.join(root_dir, mode)
        if os.path.isdir(mode_dir):
            self.root_dir = mode_dir
            if properties_file is None:
                properties_file = os.path.join(mode_dir, 'physical_properties.json')
            print(f"[{mode.upper()}] 使用目录: {mode_dir}")
        else:
            self.root_dir = root_dir
            if properties_file is None:
                properties_file = os.path.join(root_dir, 'physical_properties.json')
            print(f"[{mode.upper()}] 使用旧目录结构: {root_dir}")
        
        with open(properties_file, 'r') as f:
            props_config = json.load(f)
        
        self.properties = props_config['properties']
        self.mass_to_idx = props_config['mass_to_idx']
        self.stiffness_to_idx = props_config['stiffness_to_idx']
        self.material_to_idx = props_config['material_to_idx']
        
        # 反向映射
        self.idx_to_mass = props_config['idx_to_mass']
        self.idx_to_stiffness = props_config['idx_to_stiffness']
        self.idx_to_material = props_config['idx_to_material']
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
            
        self.samples = self._collect_samples()
        print(f"[{mode.upper()}] 加载 {len(self.samples)} 个样本")

    def _collect_samples(self):
        samples = []
        for obj_class in os.listdir(self.root_dir):
            obj_class_path = os.path.join(self.root_dir, obj_class)
            if not os.path.isdir(obj_class_path) or obj_class.startswith('analysis'):
                continue
            
            labels = self._parse_labels(obj_class)
            if labels is None:
                continue
            
            for episode in os.listdir(obj_class_path):
                episode_path = os.path.join(obj_class_path, episode)
                if not os.path.isdir(episode_path):
                    continue
                
                img_path = os.path.join(episode_path, 'visual_anchor.jpg')
                tactile_path = os.path.join(episode_path, 'tactile_data.pkl')
                
                if os.path.exists(img_path) and os.path.exists(tactile_path):
                    samples.append({
                        'img_path': img_path,
                        'tactile_path': tactile_path,
                        'labels': labels,
                        'obj_class': obj_class
                    })
        return samples

    def _parse_labels(self, obj_class):
        if obj_class not in self.properties:
            return None
        props = self.properties[obj_class]
        return {
            'mass': self.mass_to_idx[props['mass']],
            'stiffness': self.stiffness_to_idx[props['stiffness']],
            'material': self.material_to_idx[props['material']]
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 1. Load Image
        try:
            img = Image.open(sample['img_path']).convert('RGB')
            img = self.transform(img)
        except Exception as e:
            print(f"Error loading image: {sample['img_path']}: {e}")
            img = torch.zeros(3, 224, 224)
            
        # 2. Load Tactile
        try:
            with open(sample['tactile_path'], 'rb') as f:
                data = pickle.load(f)
            
            timestamps = np.array(data['timestamps'])
            L = len(timestamps)
            
            def get_arr(key, expected_dim=6):
                arr = np.array(data.get(key, []))
                if arr.size == 0:
                    return np.zeros((L, expected_dim))
                if arr.ndim == 1:
                    return np.zeros((L, expected_dim))
                return arr

            pos = get_arr('joint_position_profile')
            load = get_arr('joint_load_profile')
            curr = get_arr('joint_current_profile')
            vel = get_arr('joint_velocity_profile')
            
            # Concatenate: (L, 24) -> Transpose to (24, L)
            tactile = np.concatenate([pos, load, curr, vel], axis=1).T.astype(np.float32)
            
            # Apply Z-Score normalization
            for i, key in enumerate(['joint_position', 'joint_load', 'joint_current', 'joint_velocity']):
                mean = TACTILE_STATS[key]['mean']
                std = TACTILE_STATS[key]['std']
                tactile[i*6:(i+1)*6] = (tactile[i*6:(i+1)*6] - mean) / (std + 1e-6)
                
        except Exception as e:
            print(f"Error loading tactile: {sample['tactile_path']}: {e}")
            tactile = np.zeros((24, 10), dtype=np.float32)

        # 3. Padding / Truncating
        C, T = tactile.shape
        if T > self.max_tactile_len:
            tactile = tactile[:, :self.max_tactile_len]
            padding_mask = np.zeros(self.max_tactile_len, dtype=bool)
        else:
            padding = np.zeros((C, self.max_tactile_len - T), dtype=np.float32)
            tactile = np.concatenate([tactile, padding], axis=1)
            padding_mask = np.zeros(self.max_tactile_len, dtype=bool)
            padding_mask[T:] = True
            
        return {
            'image': img,
            'tactile': torch.FloatTensor(tactile),
            'padding_mask': torch.BoolTensor(padding_mask),
            'mass': torch.tensor(sample['labels']['mass'], dtype=torch.long),
            'stiffness': torch.tensor(sample['labels']['stiffness'], dtype=torch.long),
            'material': torch.tensor(sample['labels']['material'], dtype=torch.long)
        }


# ============================================================================
# 模态屏蔽逻辑 (核心功能)
# ============================================================================

def apply_modality_block(images, tactile, block_mode):
    """
    根据 block_mode 将对应的模态输入置零
    
    Args:
        images: (B, 3, 224, 224) 视觉输入
        tactile: (B, 24, T) 触觉输入
        block_mode: 'none' | 'visual' | 'tactile'
        
    Returns:
        处理后的 (images, tactile)
    """
    if block_mode == 'visual':
        # 屏蔽视觉：将图像全置为 0 (黑色图像)
        return torch.zeros_like(images), tactile
    elif block_mode == 'tactile':
        # 屏蔽触觉：将触觉信号全置为 0
        return images, torch.zeros_like(tactile)
    else:
        # 不屏蔽
        return images, tactile


# ============================================================================
# 训练函数
# ============================================================================

def train(config, block_mode='none'):
    device = torch.device(config['device'])
    
    print("\n" + "=" * 60)
    print(f"🚀 模态屏蔽消融实验")
    print("=" * 60)
    print(f"  屏蔽模态: {block_mode.upper()}")
    if block_mode == 'visual':
        print(f"  → 模型仅使用触觉输入 (视觉为全零)")
    elif block_mode == 'tactile':
        print(f"  → 模型仅使用视觉输入 (触觉为全零)")
    else:
        print(f"  → 模型使用完整的视觉+触觉输入")
    print(f"  设备: {device}")
    print(f"  数据集: {config['data_root']}")
    print(f"  保存目录: {config['save_dir']}")
    print("=" * 60)
    
    # Dataset & Loader
    train_dataset = RoboticGraspDataset(
        config['data_root'], 
        mode='train', 
        max_tactile_len=config['max_tactile_len']
    )
    val_dataset = RoboticGraspDataset(
        config['data_root'], 
        mode='val', 
        max_tactile_len=config['max_tactile_len']
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    # Model
    model = FusionModel(
        fusion_dim=config['fusion_dim'],
        num_heads=config.get('num_heads', 8),
        dropout=config.get('dropout', 0.1),
        num_layers=config.get('num_layers', 4),
        freeze_visual=True
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=config['weight_decay']
    )
    criterion = nn.CrossEntropyLoss()
    
    # Training History
    history = {
        'train_loss': [],
        'val_acc_mass': [],
        'val_acc_stiffness': [],
        'val_acc_material': [],
        'val_acc_avg': []
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    start_time = time.time()
    
    print(f"\n开始训练 ({config['epochs']} epochs)...")
    print("-" * 60)
    
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            images = batch['image'].to(device)
            tactile = batch['tactile'].to(device)
            mask = batch['padding_mask'].to(device)
            labels = {k: v.to(device) for k, v in batch.items() if k in ['mass', 'stiffness', 'material']}
            
            # === 应用模态屏蔽 ===
            images, tactile = apply_modality_block(images, tactile, block_mode)
            
            optimizer.zero_grad()
            outputs = model(images, tactile, padding_mask=mask)
            
            loss = (criterion(outputs['mass'], labels['mass']) + 
                    criterion(outputs['stiffness'], labels['stiffness']) + 
                    criterion(outputs['material'], labels['material'])) / 3.0
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        val_results = evaluate(model, val_loader, device, block_mode)
        avg_loss = train_loss / len(train_loader)
        
        # Record history
        history['train_loss'].append(avg_loss)
        history['val_acc_mass'].append(val_results['mass'])
        history['val_acc_stiffness'].append(val_results['stiffness'])
        history['val_acc_material'].append(val_results['material'])
        history['val_acc_avg'].append(val_results['avg'])
        
        print(f"Epoch {epoch+1:02d}/{config['epochs']} | "
              f"Loss: {avg_loss:.4f} | "
              f"Val Acc - Mass: {val_results['mass']:.2%}, "
              f"Stiff: {val_results['stiffness']:.2%}, "
              f"Mat: {val_results['material']:.2%}, "
              f"Avg: {val_results['avg']:.2%}")
        
        if val_results['avg'] > best_val_acc:
            best_val_acc = val_results['avg']
            best_epoch = epoch + 1
            save_name = f"best_model_block_{block_mode}.pth"
            save_path = os.path.join(config['save_dir'], save_name)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_val_acc,
                'block_mode': block_mode,
                'config': config
            }, save_path)
            print(f"  --> ⭐ Best Model Saved: {save_path}")

    # Training Complete
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("✅ 训练完成!")
    print("=" * 60)
    print(f"  总用时: {total_time/60:.1f} 分钟")
    print(f"  最佳验证准确率: {best_val_acc:.2%} (Epoch {best_epoch})")
    print(f"  屏蔽模态: {block_mode}")
    
    # Save history
    history_path = os.path.join(config['save_dir'], f'history_block_{block_mode}.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  训练历史: {history_path}")
    
    # === 保存训练曲线图 ===
    curve_path = os.path.join(config['save_dir'], f'training_curves_block_{block_mode}.png')
    plot_training_curves(history, curve_path, block_mode)
    
    # === 保存混淆矩阵 ===
    # 加载最佳模型
    best_model_path = os.path.join(config['save_dir'], f'best_model_block_{block_mode}.pth')
    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 获取预测结果
    all_preds, all_labels = evaluate_with_predictions(model, val_loader, device, block_mode)
    
    # 获取标签名称 (使用 train 目录的 properties 文件)
    with open(os.path.join(config['data_root'], 'train', 'physical_properties.json'), 'r') as f:
        props = json.load(f)
    
    label_names = {
        'mass': [props['idx_to_mass'][str(i)] for i in range(len(props['idx_to_mass']))],
        'stiffness': [props['idx_to_stiffness'][str(i)] for i in range(len(props['idx_to_stiffness']))],
        'material': [props['idx_to_material'][str(i)] for i in range(len(props['idx_to_material']))]
    }
    
    cm_path = os.path.join(config['save_dir'], f'confusion_matrix_block_{block_mode}.png')
    plot_confusion_matrices(all_preds, all_labels, label_names, cm_path, block_mode)
    
    print("=" * 60)
    
    return model, history


def evaluate(model, loader, device, block_mode):
    """评估模型并返回各任务的准确率"""
    model.eval()
    
    correct = {'mass': 0, 'stiffness': 0, 'material': 0}
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            tactile = batch['tactile'].to(device)
            mask = batch['padding_mask'].to(device)
            labels = {k: v.to(device) for k, v in batch.items() if k in ['mass', 'stiffness', 'material']}
            
            # === 应用模态屏蔽 ===
            images, tactile = apply_modality_block(images, tactile, block_mode)
            
            outputs = model(images, tactile, padding_mask=mask)
            
            for task in ['mass', 'stiffness', 'material']:
                preds = outputs[task].argmax(dim=1)
                correct[task] += (preds == labels[task]).sum().item()
            
            total += images.size(0)
    
    results = {
        'mass': correct['mass'] / total,
        'stiffness': correct['stiffness'] / total,
        'material': correct['material'] / total,
    }
    results['avg'] = (results['mass'] + results['stiffness'] + results['material']) / 3
    
    return results


def evaluate_with_predictions(model, loader, device, block_mode):
    """评估模型并返回预测结果和真实标签（用于混淆矩阵）"""
    model.eval()
    
    all_preds = {'mass': [], 'stiffness': [], 'material': []}
    all_labels = {'mass': [], 'stiffness': [], 'material': []}
    
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            tactile = batch['tactile'].to(device)
            mask = batch['padding_mask'].to(device)
            labels = {k: v.to(device) for k, v in batch.items() if k in ['mass', 'stiffness', 'material']}
            
            # === 应用模态屏蔽 ===
            images, tactile = apply_modality_block(images, tactile, block_mode)
            
            outputs = model(images, tactile, padding_mask=mask)
            
            for task in ['mass', 'stiffness', 'material']:
                preds = outputs[task].argmax(dim=1)
                all_preds[task].extend(preds.cpu().tolist())
                all_labels[task].extend(labels[task].cpu().tolist())
    
    return all_preds, all_labels


def plot_training_curves(history, save_path, block_mode):
    """绘制训练曲线图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curve
    axes[0].plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'Training Loss (Block: {block_mode})', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[1].plot(epochs, history['val_acc_mass'], 'r-', linewidth=2, label='Mass')
    axes[1].plot(epochs, history['val_acc_stiffness'], 'g-', linewidth=2, label='Stiffness')
    axes[1].plot(epochs, history['val_acc_material'], 'b-', linewidth=2, label='Material')
    axes[1].plot(epochs, history['val_acc_avg'], 'k--', linewidth=2, label='Average')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title(f'Validation Accuracy (Block: {block_mode})', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  训练曲线图: {save_path}")


def plot_confusion_matrices(all_preds, all_labels, label_names, save_path, block_mode):
    """绘制混淆矩阵"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    tasks = ['mass', 'stiffness', 'material']
    
    for idx, task in enumerate(tasks):
        preds = all_preds[task]
        labels = all_labels[task]
        names = label_names[task]
        
        # 确保包含所有类别
        all_class_labels = list(range(len(names)))
        cm = confusion_matrix(labels, preds, labels=all_class_labels)
        
        # 归一化
        cm_sum = cm.sum(axis=1, keepdims=True)
        cm_sum[cm_sum == 0] = 1  # 避免除零
        cm_normalized = cm.astype('float') / cm_sum
        
        # 创建注释（数量 + 百分比）
        annot = []
        for i in range(cm.shape[0]):
            row = []
            for j in range(cm.shape[1]):
                count = cm[i, j]
                pct = cm_normalized[i, j]
                row.append(f'{count}\n({pct:.0%})')
            annot.append(row)
        annot = np.array(annot)
        
        # 绘制热力图
        sns.heatmap(
            cm_normalized,
            annot=annot,
            fmt='',
            cmap='Blues',
            xticklabels=names,
            yticklabels=names,
            ax=axes[idx],
            cbar=True,
            square=True,
            annot_kws={'fontsize': 9}
        )
        
        axes[idx].set_title(f'{task.upper()} (Block: {block_mode})', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted', fontsize=10)
        axes[idx].set_ylabel('True', fontsize=10)
        axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  混淆矩阵图: {save_path}")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='模态屏蔽消融实验 (Modality Blocking Ablation Study)')
    parser.add_argument('--block_modality', type=str, default='none', 
                        choices=['none', 'visual', 'tactile'],
                        help='要屏蔽的模态: none(完整输入), visual(仅触觉), tactile(仅视觉)')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--save_dir', type=str, default='outputs/ablation_study', help='模型保存目录')
    parser.add_argument('--device', type=str, default='cuda', help='设备 (cuda/cpu)')
    parser.add_argument('--eval_only', action='store_true', help='仅评估模式：跳过训练，直接加载已有模型生成混淆矩阵')
    
    args = parser.parse_args()
    
    config = {
        'data_root': '/home/martina/Y3_Project/Plaintextdataset',
        'device': args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu',
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': 0.01,
        'epochs': args.epochs,
        'fusion_dim': 256,
        'num_heads': 8,
        'dropout': 0.1,
        'num_layers': 4,
        'save_dir': args.save_dir,
        'max_tactile_len': 3000,
        'num_workers': 4
    }
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
    if args.eval_only:
        # === 仅评估模式 ===
        block_mode = args.block_modality
        device = config['device']
        
        print("=" * 60)
        print("🔍 仅评估模式 (Eval Only)")
        print("=" * 60)
        print(f"  屏蔽模态: {block_mode}")
        print(f"  设备: {device}")
        
        # 加载验证集
        val_dataset = RoboticGraspDataset(
            config['data_root'], 
            mode='val', 
            max_tactile_len=config['max_tactile_len']
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config['batch_size'], 
            shuffle=False, 
            num_workers=config.get('num_workers', 4),
            pin_memory=True
        )
        
        # 创建模型并加载权重
        model = FusionModel(
            fusion_dim=config['fusion_dim'],
            num_heads=config['num_heads'],
            dropout=config['dropout'],
            num_layers=config['num_layers'],
            freeze_visual=True
        ).to(device)
        
        best_model_path = os.path.join(config['save_dir'], f'best_model_block_{block_mode}.pth')
        print(f"  加载模型: {best_model_path}")
        
        checkpoint = torch.load(best_model_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # 评估准确率
        val_results = evaluate(model, val_loader, device, block_mode)
        print(f"\n📊 验证集准确率:")
        print(f"    Mass: {val_results['mass']:.2%}")
        print(f"    Stiffness: {val_results['stiffness']:.2%}")
        print(f"    Material: {val_results['material']:.2%}")
        print(f"    Average: {val_results['avg']:.2%}")
        
        # 生成混淆矩阵
        all_preds, all_labels = evaluate_with_predictions(model, val_loader, device, block_mode)
        
        # 获取标签名称
        with open(os.path.join(config['data_root'], 'train', 'physical_properties.json'), 'r') as f:
            props = json.load(f)
        
        label_names = {
            'mass': [props['idx_to_mass'][str(i)] for i in range(len(props['idx_to_mass']))],
            'stiffness': [props['idx_to_stiffness'][str(i)] for i in range(len(props['idx_to_stiffness']))],
            'material': [props['idx_to_material'][str(i)] for i in range(len(props['idx_to_material']))]
        }
        
        cm_path = os.path.join(config['save_dir'], f'confusion_matrix_block_{block_mode}.png')
        plot_confusion_matrices(all_preds, all_labels, label_names, cm_path, block_mode)
        
        print("=" * 60)
        print("✅ 评估完成!")
        print("=" * 60)
    else:
        # === 训练模式 ===
        train(config, block_mode=args.block_modality)
