import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import json
import pickle
from PIL import Image
from torchvision import transforms


# ============================================================================
# 触觉数据归一化统计量 (基于全数据集统计)
# 用于 Z-Score 标准化: (x - mean) / std
# ============================================================================
TACTILE_STATS = {
    'joint_position': {'mean': 21.70, 'std': 38.13},
    'joint_load':     {'mean': 7.21,  'std': 14.03},
    'joint_current':  {'mean': 52.56, 'std': 133.43},
    'joint_velocity': {'mean': 0.13,  'std': 9.79},
}


class FusionModel(nn.Module):
    def __init__(self,fusion_dim=256,num_heads=8,dropout=0.1,num_layers=4, freeze_visual=True):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        self.dropout = dropout

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        #去掉最后的平均池化层和全连接层
        self.vis_backbone = nn.Sequential(*list(resnet.children())[:-2])
        #将512维的特征图投影到fusion_dim维
        self.vis_proj = nn.Conv2d(512,fusion_dim,kernel_size=1)
        #冻结视觉主干
        if freeze_visual:
            for param in self.vis_backbone.parameters():
                param.requires_grad = False
        self.tac_encoder = nn.Sequential(
            #layer1
            nn.Conv1d(24,64,kernel_size=7,stride=2,padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            #layer2
            nn.Conv1d(64,128,kernel_size=5,stride=2,padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            #layer3
            nn.Conv1d(128,fusion_dim,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU()
        )
        #经过三次卷积，特征图大小变为原来的1/8

        #fusion core
        #CLS token
        self.cls_token = nn.Parameter(torch.randn(1,1,fusion_dim))

        #learnable positional embedding
        #最大长度为 cls_token+vis_token+tac_token
        self.pos_emb = nn.Parameter(torch.randn(1,425,fusion_dim))

        #transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=num_heads,
            dim_feedforward=fusion_dim*4,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,num_layers=num_layers)


        #multi-head output layer
        self.head_mass = nn.Sequential(
            nn.Linear(fusion_dim,128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128,4)
        )
        self.head_stiffness = nn.Sequential(
            nn.Linear(fusion_dim,128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128,4)
        )
        self.head_material = nn.Sequential(
            nn.Linear(fusion_dim,128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128,5)
        )





    def forward(self, img, tac, padding_mask=None):
        '''
        Image Input: (B, 3, 224, 224)
        Tactile Input: (B, 24, T)   T是时间窗口长度, 基于当前数据集。 单个数据最大长度为3000
        padding_mask: (B, T) bool型, True=填充位置(被忽略), False=有效数据
        
        Output:
        - mass: (B, 4)
        - stiffness: (B, 4)
        - material: (B, 5)
        '''
        B = img.shape[0]
        device = img.device

        # --- A. Visual Tokenization ---
        # 1. CNN Features Extraction
        v = self.vis_backbone(img)  # [B,512,7,7]
        # 2. Project to fusion_dim [B,256,7,7]
        v = self.vis_proj(v)
        # 3. Flatten spatial [B,256,7,7] -> [B,49,256]
        v_tokens = v.flatten(2).transpose(1,2)
        num_vis_tokens = v_tokens.shape[1]  # 49


        # --- B. Tactile Tokenization ---
        # 1. 1D-CNN Extract: [B,256,Length/8] 
        t = self.tac_encoder(tac)
        # 2. Transpose [B,256,Length/8] -> [B,Length/8,256]
        t_tokens = t.transpose(1,2)
        num_tac_tokens = t_tokens.shape[1]  # T/8


        # --- C. Fusion ---
        # 1. Expand CLS token 
        cls_token = self.cls_token.expand(B,-1,-1)
        # 2. Concatenate [CLS,Visual,Tactile]
        # shape: [B,1+49+Length/8,256]
        x = torch.cat([cls_token, v_tokens, t_tokens], dim=1)

        # 3. Add positional embedding (按照当前实际长度截取)
        seq_len = x.shape[1]
        x = x + self.pos_emb[:, :seq_len, :]

        # --- D. 构建完整的 Attention Mask ---
        # Transformer 需要知道哪些位置是 padding (True=忽略, False=保留)
        full_mask = None
        if padding_mask is not None:
            # 原始 padding_mask: (B, T)
            # 触觉经过 3 层 stride=2 的卷积, 长度变为 T/8
            # 使用 max_pool1d 对 mask 进行下采样 (任何一个位置有 True 就是 True)
            # 转换为 float 进行池化，然后转回 bool
            tac_mask = padding_mask.float().unsqueeze(1)  # (B, 1, T)
            # 三次 stride=2 的下采样
            tac_mask = torch.nn.functional.max_pool1d(tac_mask, kernel_size=2, stride=2)
            tac_mask = torch.nn.functional.max_pool1d(tac_mask, kernel_size=2, stride=2)
            tac_mask = torch.nn.functional.max_pool1d(tac_mask, kernel_size=2, stride=2)
            tac_mask = tac_mask.squeeze(1) > 0.5  # (B, T/8) 转回 bool
            
            # 确保长度匹配 (可能因为 padding 导致长度不完全是 /8)
            tac_mask = tac_mask[:, :num_tac_tokens]
            
            # 构建完整 mask: [CLS (1个False) + Visual (49个False) + Tactile (T/8个)]
            cls_vis_mask = torch.zeros(B, 1 + num_vis_tokens, dtype=torch.bool, device=device)
            full_mask = torch.cat([cls_vis_mask, tac_mask], dim=1)  # (B, 1+49+T/8)

        # 4. Transformer forward (带 mask)
        x = self.transformer_encoder(x, src_key_padding_mask=full_mask)

        # 5. predict
        # extract CLS token
        cls_out = x[:, 0, :]

        # multi-head output
        pred_mass = self.head_mass(cls_out)
        pred_stiffness = self.head_stiffness(cls_out)
        pred_material = self.head_material(cls_out)

        return {
            'mass': pred_mass,
            'stiffness': pred_stiffness,
            'material': pred_material
        }

class RoboticGraspDataset(Dataset):
    """
    抓取物体属性识别数据集
    
    数据目录结构:
    root_dir/
        ObjectClass_Variant/
            episode_xxx/
                visual_anchor.jpg   (RGB图像)
                tactile_data.pkl    (触觉时序数据)
                metadata.json       (元信息)
    
    触觉输入: 24维 = joint_position(6) + joint_load(6) + joint_current(6) + joint_velocity(6)
    
    标签说明 (来自 physical_properties.json):
    - mass (4类):       very_low(0), low(1), medium(2), high(3)
    - stiffness (4类):  very_soft(0), soft(1), medium(2), rigid(3)
    - material (5类):   sponge(0), foam(1), wood(2), hollow_container(3), filled_container(4)
    """

    def __init__(self, root_dir, properties_file=None, max_tactile_len=3000, transform=None, 
                 mode='train', train_ratio=0.8, seed=42):
        """
        Args:
            root_dir: 数据集根目录路径 (包含 train/ 和 val/ 子目录)
            properties_file: 物理属性JSON配置文件路径 (可选，默认自动从 root_dir/{mode}/physical_properties.json 加载)
            max_tactile_len: 触觉序列最大长度，不足则padding，超过则截断
            transform: 图像变换（可选），默认使用ImageNet标准化
            mode: 'train' 或 'val'
            train_ratio: 训练集比例 (已弃用，现在使用预定义的 train/val 目录)
            seed: 随机种子 (已弃用)
        """
        super().__init__()
        self.max_tactile_len = max_tactile_len
        self.mode = mode
        
        # 新结构：root_dir/train/ 和 root_dir/val/
        mode_dir = os.path.join(root_dir, mode)
        if os.path.isdir(mode_dir):
            # 使用新的 train/val 目录结构
            self.root_dir = mode_dir
            # 自动加载对应目录下的 properties 文件
            if properties_file is None:
                properties_file = os.path.join(mode_dir, 'physical_properties.json')
            print(f"[{mode.upper()}] 使用目录: {mode_dir}")
        else:
            # 兼容旧结构：直接使用 root_dir
            self.root_dir = root_dir
            if properties_file is None:
                properties_file = os.path.join(root_dir, 'physical_properties.json')
            print(f"[{mode.upper()}] 使用旧目录结构: {root_dir}")
        
        # 加载物理属性配置
        with open(properties_file, 'r') as f:
            props_config = json.load(f)
        
        self.properties = props_config['properties']           # 物体类别 -> 属性字典
        self.mass_to_idx = props_config['mass_to_idx']         # mass字符串 -> 索引
        self.stiffness_to_idx = props_config['stiffness_to_idx']
        self.material_to_idx = props_config['material_to_idx']
        
        # 反向映射（用于打印/调试）
        self.idx_to_mass = props_config['idx_to_mass']
        self.idx_to_stiffness = props_config['idx_to_stiffness']
        self.idx_to_material = props_config['idx_to_material']
        
        # 默认图像变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
        
        # 收集所有有效样本 (不再需要随机划分，直接使用 train/val 目录下的所有数据)
        self.samples = self._collect_samples()
        
        print(f"[{mode.upper()}] 加载 {len(self.samples)} 个样本")

    def _collect_samples(self):
        """扫描数据目录，收集所有有效样本路径"""
        samples = []
        
        # 遍历所有物体类别目录
        for obj_class in os.listdir(self.root_dir):
            obj_class_path = os.path.join(self.root_dir, obj_class)
            if not os.path.isdir(obj_class_path) or obj_class.startswith('analysis'):
                continue
            
            # 解析标签
            labels = self._parse_labels(obj_class)
            if labels is None:
                print(f"[警告] 无法解析标签: {obj_class}")
                continue
            
            # 遍历该类别下的所有episode
            for episode in os.listdir(obj_class_path):
                episode_path = os.path.join(obj_class_path, episode)
                if not os.path.isdir(episode_path):
                    continue
                
                # 检查必需文件是否存在
                img_path = os.path.join(episode_path, 'visual_anchor.jpg')
                tactile_path = os.path.join(episode_path, 'tactile_data.pkl')
                
                if os.path.exists(img_path) and os.path.exists(tactile_path):
                    samples.append({
                        'img_path': img_path,
                        'tactile_path': tactile_path,
                        'labels': labels,
                        'obj_class': obj_class,
                    })
        
        return samples

    def _parse_labels(self, obj_class):
        """
        根据物体类别名称从配置文件中查找mass, stiffness, material标签
        
        Args:
            obj_class: 物体类别名称 (例如: WoodBlock_Native, CardboardBox_Hollow)
        
        Returns:
            dict: {'mass': int, 'stiffness': int, 'material': int} 或 None
        """
        # 直接从配置中查找
        if obj_class not in self.properties:
            return None
        
        props = self.properties[obj_class]
        
        # 转换为索引
        mass_idx = self.mass_to_idx.get(props['mass'])
        stiffness_idx = self.stiffness_to_idx.get(props['stiffness'])
        material_idx = self.material_to_idx.get(props['material'])
        
        if mass_idx is None or stiffness_idx is None or material_idx is None:
            return None
        
        return {
            'mass': mass_idx,
            'stiffness': stiffness_idx,
            'material': material_idx,
        }

    def _load_tactile(self, tactile_path):
        """
        加载并处理触觉数据 (带 Z-Score 归一化和 Padding Mask)
        
        Returns:
            tactile: torch.Tensor, shape (24, max_tactile_len) - 归一化后的触觉数据
            padding_mask: torch.Tensor, shape (max_tactile_len,) - 1=有效数据, 0=填充数据
        """
        with open(tactile_path, 'rb') as f:
            data = pickle.load(f)
        
        # ========== 1. 提取并进行 Z-Score 标准化 ==========
        # 将不同量级的物理量归一化到近似相同的范围，避免梯度被大数值主导
        def normalize(arr, stat_key):
            mean = TACTILE_STATS[stat_key]['mean']
            std = TACTILE_STATS[stat_key]['std']
            return (np.array(arr) - mean) / (std + 1e-8)  # 加小量防止除零
        
        joint_pos = normalize(data['joint_position_profile'], 'joint_position')     # (T, 6)
        joint_load = normalize(data['joint_load_profile'], 'joint_load')             # (T, 6)
        joint_current = normalize(data['joint_current_profile'], 'joint_current')   # (T, 6)
        joint_vel = normalize(data['joint_velocity_profile'], 'joint_velocity')     # (T, 6)
        
        # ========== 2. 拼接为 (T, 24) 并转置为 (24, T) ==========
        tactile = np.concatenate([joint_pos, joint_load, joint_current, joint_vel], axis=1)
        tactile = tactile.T  # (24, T)
        
        T = tactile.shape[1]
        
        # ========== 3. 创建输出张量和 Padding Mask ==========
        # padding_mask: 1 表示有效数据，0 表示填充 (Transformer 需要知道哪些是填充)
        tactile_tensor = torch.zeros((24, self.max_tactile_len), dtype=torch.float32)
        padding_mask = torch.zeros(self.max_tactile_len, dtype=torch.bool)  # False=有效, True=需mask
        
        valid_len = min(T, self.max_tactile_len)
        
        # 填充有效数据
        tactile_tensor[:, :valid_len] = torch.tensor(tactile[:, :valid_len], dtype=torch.float32)
        # 设置 padding mask (PyTorch Transformer 约定: True=忽略, False=保留)
        padding_mask[valid_len:] = True  # 填充部分设为 True (被忽略)
        
        return tactile_tensor, padding_mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # ========== 加载图像 ==========
        try:
            img = Image.open(sample['img_path']).convert('RGB')
            if self.transform:
                img = self.transform(img)
        except Exception as e:
            # 鲁棒性处理：如果图片损坏，生成全黑图
            print(f"[警告] 图像损坏: {sample['img_path']}, 使用全黑图替代")
            img = torch.zeros((3, 224, 224), dtype=torch.float32)
        
        # ========== 加载触觉数据 ==========
        tactile, padding_mask = self._load_tactile(sample['tactile_path'])
        
        # ========== 标签 ==========
        labels = sample['labels']
        
        return {
            'image': img,                                              # (3, 224, 224)
            'tactile': tactile,                                        # (24, max_tactile_len)
            'padding_mask': padding_mask,                              # (max_tactile_len,) bool
            'mass': torch.tensor(labels['mass'], dtype=torch.long),
            'stiffness': torch.tensor(labels['stiffness'], dtype=torch.long),
            'material': torch.tensor(labels['material'], dtype=torch.long),
        }


def get_dataloaders(root_dir, properties_file=None, batch_size=16, max_tactile_len=3000, num_workers=4):
    """
    创建训练和验证数据加载器
    
    Args:
        root_dir: 数据集根目录 (包含 train/ 和 val/ 子目录)
        properties_file: 物理属性JSON配置文件路径 (可选，默认自动从各子目录加载)
        batch_size: 批次大小
        max_tactile_len: 触觉序列最大长度
        num_workers: 数据加载线程数
    
    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader
    
    train_dataset = RoboticGraspDataset(
        root_dir=root_dir,
        properties_file=properties_file,
        max_tactile_len=max_tactile_len,
        mode='train'
    )
    
    val_dataset = RoboticGraspDataset(
        root_dir=root_dir,
        properties_file=properties_file,
        max_tactile_len=max_tactile_len,
        mode='val'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


# ============================================================================
# 实时可视化
# ============================================================================
import matplotlib
matplotlib.use('TkAgg')  # 使用交互式后端
import matplotlib.pyplot as plt


class FusionLivePlotter:
    """实时绘制视觉-触觉融合模型训练曲线"""
    
    def __init__(self, output_dir=None):
        self.output_dir = output_dir
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc_mass': [], 'val_acc_mass': [],
            'train_acc_stiffness': [], 'val_acc_stiffness': [],
            'train_acc_material': [], 'val_acc_material': [],
            'learning_rate': [],
        }
        
        plt.ion()  # 开启交互模式
        
        # 2x3 布局
        self.fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        self.fig.suptitle('Visual-Tactile Fusion Model Training', fontsize=14, fontweight='bold')
        
        # ===== Loss =====
        self.ax_loss = axes[0, 0]
        self.line_train_loss, = self.ax_loss.plot([], [], 'b-o', label='Train', markersize=3, linewidth=1.5)
        self.line_val_loss, = self.ax_loss.plot([], [], 'r-o', label='Val', markersize=3, linewidth=1.5)
        self.ax_loss.set_title('Total Loss')
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.legend(loc='upper right')
        self.ax_loss.grid(True, alpha=0.3)
        
        # ===== Mass Accuracy =====
        self.ax_mass = axes[0, 1]
        self.line_train_mass, = self.ax_mass.plot([], [], 'b-o', label='Train', markersize=3, linewidth=1.5)
        self.line_val_mass, = self.ax_mass.plot([], [], 'r-o', label='Val', markersize=3, linewidth=1.5)
        self.ax_mass.set_title('Mass Accuracy')
        self.ax_mass.set_xlabel('Epoch')
        self.ax_mass.set_ylabel('Accuracy')
        self.ax_mass.set_ylim(0, 1.05)
        self.ax_mass.legend(loc='lower right')
        self.ax_mass.grid(True, alpha=0.3)
        
        # ===== Stiffness Accuracy =====
        self.ax_stiffness = axes[0, 2]
        self.line_train_stiffness, = self.ax_stiffness.plot([], [], 'b-o', label='Train', markersize=3, linewidth=1.5)
        self.line_val_stiffness, = self.ax_stiffness.plot([], [], 'r-o', label='Val', markersize=3, linewidth=1.5)
        self.ax_stiffness.set_title('Stiffness Accuracy')
        self.ax_stiffness.set_xlabel('Epoch')
        self.ax_stiffness.set_ylabel('Accuracy')
        self.ax_stiffness.set_ylim(0, 1.05)
        self.ax_stiffness.legend(loc='lower right')
        self.ax_stiffness.grid(True, alpha=0.3)
        
        # ===== Material Accuracy =====
        self.ax_material = axes[1, 0]
        self.line_train_material, = self.ax_material.plot([], [], 'b-o', label='Train', markersize=3, linewidth=1.5)
        self.line_val_material, = self.ax_material.plot([], [], 'r-o', label='Val', markersize=3, linewidth=1.5)
        self.ax_material.set_title('Material Accuracy')
        self.ax_material.set_xlabel('Epoch')
        self.ax_material.set_ylabel('Accuracy')
        self.ax_material.set_ylim(0, 1.05)
        self.ax_material.legend(loc='lower right')
        self.ax_material.grid(True, alpha=0.3)
        
        # ===== Val Accuracy 对比 =====
        self.ax_compare = axes[1, 1]
        self.line_cmp_mass, = self.ax_compare.plot([], [], 's-', label='Mass', markersize=4, linewidth=1.5, color='#ff7f0e')
        self.line_cmp_stiffness, = self.ax_compare.plot([], [], '^-', label='Stiffness', markersize=4, linewidth=1.5, color='#2ca02c')
        self.line_cmp_material, = self.ax_compare.plot([], [], 'd-', label='Material', markersize=4, linewidth=1.5, color='#d62728')
        self.ax_compare.set_title('Val Accuracy Comparison')
        self.ax_compare.set_xlabel('Epoch')
        self.ax_compare.set_ylabel('Accuracy')
        self.ax_compare.set_ylim(0, 1.05)
        self.ax_compare.legend(loc='lower right')
        self.ax_compare.grid(True, alpha=0.3)
        
        # ===== Learning Rate =====
        self.ax_lr = axes[1, 2]
        self.line_lr, = self.ax_lr.plot([], [], 'g-o', markersize=3, linewidth=1.5)
        self.ax_lr.set_title('Learning Rate')
        self.ax_lr.set_xlabel('Epoch')
        self.ax_lr.set_ylabel('LR')
        self.ax_lr.set_yscale('log')
        self.ax_lr.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.show(block=False)
    
    def update(self, **kwargs):
        """更新所有图表"""
        for key, value in kwargs.items():
            if key in self.history:
                self.history[key].append(value)
        
        epochs = list(range(1, len(self.history['train_loss']) + 1))
        
        # 更新 Loss
        self.line_train_loss.set_data(epochs, self.history['train_loss'])
        self.line_val_loss.set_data(epochs, self.history['val_loss'])
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()
        
        # 更新 Mass Acc
        self.line_train_mass.set_data(epochs, self.history['train_acc_mass'])
        self.line_val_mass.set_data(epochs, self.history['val_acc_mass'])
        self.ax_mass.relim()
        self.ax_mass.autoscale_view()
        self.ax_mass.set_ylim(0, 1.05)
        
        # 更新 Stiffness Acc
        self.line_train_stiffness.set_data(epochs, self.history['train_acc_stiffness'])
        self.line_val_stiffness.set_data(epochs, self.history['val_acc_stiffness'])
        self.ax_stiffness.relim()
        self.ax_stiffness.autoscale_view()
        self.ax_stiffness.set_ylim(0, 1.05)
        
        # 更新 Material Acc
        self.line_train_material.set_data(epochs, self.history['train_acc_material'])
        self.line_val_material.set_data(epochs, self.history['val_acc_material'])
        self.ax_material.relim()
        self.ax_material.autoscale_view()
        self.ax_material.set_ylim(0, 1.05)
        
        # 更新对比图
        self.line_cmp_mass.set_data(epochs, self.history['val_acc_mass'])
        self.line_cmp_stiffness.set_data(epochs, self.history['val_acc_stiffness'])
        self.line_cmp_material.set_data(epochs, self.history['val_acc_material'])
        self.ax_compare.relim()
        self.ax_compare.autoscale_view()
        self.ax_compare.set_ylim(0, 1.05)
        
        # 更新 Learning Rate
        if self.history['learning_rate']:
            self.line_lr.set_data(epochs, self.history['learning_rate'])
            self.ax_lr.relim()
            self.ax_lr.autoscale_view()
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.05)
    
    def save(self, filename='fusion_training.png'):
        """保存训练曲线图"""
        if self.output_dir:
            save_path = os.path.join(self.output_dir, filename)
            self.fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 训练曲线已保存: {save_path}")
    
    def close(self):
        """关闭图表"""
        plt.ioff()
        plt.close(self.fig)
    
    def get_history(self):
        """获取训练历史（用于保存）"""
        return self.history


# ============================================================================
# 训练相关函数
# ============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    训练一个 epoch
    
    Returns:
        dict: 包含 loss 和各任务准确率的字典
    """
    model.train()
    
    total_loss = 0.0
    correct = {'mass': 0, 'stiffness': 0, 'material': 0}
    total_samples = 0
    
    from tqdm import tqdm
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    
    for batch in pbar:
        # 数据移动到设备
        images = batch['image'].to(device)
        tactile = batch['tactile'].to(device)
        padding_mask = batch['padding_mask'].to(device)
        mass_labels = batch['mass'].to(device)
        stiffness_labels = batch['stiffness'].to(device)
        material_labels = batch['material'].to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(images, tactile, padding_mask=padding_mask)
        
        # 计算多任务损失
        loss_mass = criterion(outputs['mass'], mass_labels)
        loss_stiffness = criterion(outputs['stiffness'], stiffness_labels)
        loss_material = criterion(outputs['material'], material_labels)
        
        # 总损失 (可以加权，这里简单平均)
        loss = loss_mass + loss_stiffness + loss_material
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪 (防止梯度爆炸)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 统计
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        
        # 计算准确率
        correct['mass'] += (outputs['mass'].argmax(dim=1) == mass_labels).sum().item()
        correct['stiffness'] += (outputs['stiffness'].argmax(dim=1) == stiffness_labels).sum().item()
        correct['material'] += (outputs['material'].argmax(dim=1) == material_labels).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mass_acc': f'{correct["mass"]/total_samples:.2%}',
        })
    
    # 计算平均指标
    avg_loss = total_loss / total_samples
    acc = {k: v / total_samples for k, v in correct.items()}
    
    return {'loss': avg_loss, **acc}


@torch.no_grad()
def evaluate(model, val_loader, criterion, device):
    """
    验证模型
    
    Returns:
        dict: 包含 loss 和各任务准确率的字典
    """
    model.eval()
    
    total_loss = 0.0
    correct = {'mass': 0, 'stiffness': 0, 'material': 0}
    total_samples = 0
    
    # 用于计算混淆矩阵 (可选)
    all_preds = {'mass': [], 'stiffness': [], 'material': []}
    all_labels = {'mass': [], 'stiffness': [], 'material': []}
    
    for batch in val_loader:
        images = batch['image'].to(device)
        tactile = batch['tactile'].to(device)
        padding_mask = batch['padding_mask'].to(device)
        mass_labels = batch['mass'].to(device)
        stiffness_labels = batch['stiffness'].to(device)
        material_labels = batch['material'].to(device)
        
        # 前向传播
        outputs = model(images, tactile, padding_mask=padding_mask)
        
        # 计算损失
        loss_mass = criterion(outputs['mass'], mass_labels)
        loss_stiffness = criterion(outputs['stiffness'], stiffness_labels)
        loss_material = criterion(outputs['material'], material_labels)
        loss = loss_mass + loss_stiffness + loss_material
        
        # 统计
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        
        # 准确率
        correct['mass'] += (outputs['mass'].argmax(dim=1) == mass_labels).sum().item()
        correct['stiffness'] += (outputs['stiffness'].argmax(dim=1) == stiffness_labels).sum().item()
        correct['material'] += (outputs['material'].argmax(dim=1) == material_labels).sum().item()
        
        # 收集预测结果
        all_preds['mass'].extend(outputs['mass'].argmax(dim=1).cpu().tolist())
        all_preds['stiffness'].extend(outputs['stiffness'].argmax(dim=1).cpu().tolist())
        all_preds['material'].extend(outputs['material'].argmax(dim=1).cpu().tolist())
        all_labels['mass'].extend(mass_labels.cpu().tolist())
        all_labels['stiffness'].extend(stiffness_labels.cpu().tolist())
        all_labels['material'].extend(material_labels.cpu().tolist())
    
    avg_loss = total_loss / total_samples
    acc = {k: v / total_samples for k, v in correct.items()}
    
    return {
        'loss': avg_loss,
        **acc,
        'preds': all_preds,
        'labels': all_labels
    }


def train(config):
    """
    主训练函数
    
    Args:
        config: 训练配置字典
    """
    import time
    from datetime import datetime
    
    # ========== 配置 ==========
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"\n使用设备: {device}")
    
    # ========== 数据加载 ==========
    print("\n" + "=" * 60)
    print("加载数据集...")
    print("=" * 60)
    
    train_loader, val_loader = get_dataloaders(
        root_dir=config['data_root'],
        properties_file=config['properties_file'],
        batch_size=config.get('batch_size', 16),
        max_tactile_len=config.get('max_tactile_len', 3000),
        num_workers=config.get('num_workers', 4)
    )
    
    # ========== 模型 ==========
    print("\n初始化模型...")
    model = FusionModel(
        fusion_dim=config.get('fusion_dim', 256),
        num_heads=config.get('num_heads', 8),
        dropout=config.get('dropout', 0.1),
        num_layers=config.get('num_layers', 4),
        freeze_visual=config.get('freeze_visual', True)
    ).to(device)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    
    # ========== 损失函数和优化器 ==========
    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.get('lr', 1e-4),
        weight_decay=config.get('weight_decay', 0.01)
    )
    
    # 学习率调度器 (Cosine Annealing with Warmup)
    num_epochs = config.get('epochs', 50)
    warmup_epochs = config.get('warmup_epochs', 5)
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # ========== 训练循环 ==========
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)
    
    best_val_acc = 0.0
    best_epoch = 0
    history = {'train': [], 'val': []}
    
    save_dir = config.get('save_dir', 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    
    # ========== 初始化实时绘图 ==========
    plotter = None
    if config.get('live_plot', True):
        try:
            plotter = FusionLivePlotter(output_dir=save_dir)
            print("📊 实时绘图已启用")
        except Exception as e:
            print(f"⚠️ 无法启用实时绘图: {e}")
            plotter = None
    
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        # 训练
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # 验证
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        epoch_time = time.time() - epoch_start
        
        # 记录历史
        history['train'].append(train_metrics)
        history['val'].append({k: v for k, v in val_metrics.items() if k not in ['preds', 'labels']})
        
        # 计算平均验证准确率
        avg_val_acc = (val_metrics['mass'] + val_metrics['stiffness'] + val_metrics['material']) / 3
        
        # 打印结果
        print(f"\nEpoch {epoch}/{num_epochs} ({epoch_time:.1f}s) | LR: {current_lr:.2e}")
        print(f"  Train Loss: {train_metrics['loss']:.4f} | "
              f"Mass: {train_metrics['mass']:.2%} | "
              f"Stiff: {train_metrics['stiffness']:.2%} | "
              f"Mat: {train_metrics['material']:.2%}")
        print(f"  Val   Loss: {val_metrics['loss']:.4f} | "
              f"Mass: {val_metrics['mass']:.2%} | "
              f"Stiff: {val_metrics['stiffness']:.2%} | "
              f"Mat: {val_metrics['material']:.2%}")
        
        # ========== 更新实时绘图 ==========
        if plotter is not None:
            plotter.update(
                train_loss=train_metrics['loss'],
                val_loss=val_metrics['loss'],
                train_acc_mass=train_metrics['mass'],
                val_acc_mass=val_metrics['mass'],
                train_acc_stiffness=train_metrics['stiffness'],
                val_acc_stiffness=val_metrics['stiffness'],
                train_acc_material=train_metrics['material'],
                val_acc_material=val_metrics['material'],
                learning_rate=current_lr,
            )
        
        # 保存最佳模型
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_epoch = epoch
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_metrics': val_metrics,
                'config': config,
                'history': plotter.get_history() if plotter else history,
            }
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
            print(f"  ✓ 保存最佳模型 (Avg Acc: {avg_val_acc:.2%})")
        
        # 定期保存 checkpoint
        if epoch % config.get('save_every', 10) == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': plotter.get_history() if plotter else history,
                'config': config
            }
            torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    # ========== 训练完成 ==========
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"  总用时: {total_time/60:.1f} 分钟")
    print(f"  最佳验证准确率: {best_val_acc:.2%} (Epoch {best_epoch})")
    print(f"  模型保存位置: {os.path.join(save_dir, 'best_model.pth')}")
    
    # 保存训练历史
    final_history = plotter.get_history() if plotter else history
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(final_history, f, indent=2)
    print(f"  训练历史保存位置: {history_path}")
    
    # 保存并关闭绘图
    if plotter is not None:
        plotter.save()
        plotter.close()
    
    return model, final_history


# ============================================================================
# 评估函数
# ============================================================================

def plot_confusion_matrices(all_preds, all_labels, label_names, output_dir, task_name):
    """
    绘制单个任务的混淆矩阵
    
    Args:
        all_preds: 预测列表
        all_labels: 真实标签列表
        label_names: 标签名称列表
        output_dir: 输出目录
        task_name: 任务名称 (mass/stiffness/material)
    """
    try:
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
    except ImportError:
        print("⚠️ 需要安装 sklearn 和 seaborn: pip install scikit-learn seaborn")
        return
    
    # 使用 labels 参数确保混淆矩阵包含所有类别，即使验证集中不存在
    all_class_labels = list(range(len(label_names)))
    cm = confusion_matrix(all_labels, all_preds, labels=all_class_labels)
    cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
    
    # 创建带数量和百分比的注释
    annot_labels = []
    for i in range(cm.shape[0]):
        row = []
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = cm_normalized[i, j]
            row.append(f'{count}\n({pct:.1%})')
        annot_labels.append(row)
    annot_labels = np.array(annot_labels)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=annot_labels,
        fmt='',
        cmap='Blues',
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
        cbar=True,
        square=True,
        annot_kws={'fontsize': 10},
    )
    
    ax.set_title(f'{task_name.upper()} Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'confusion_matrix_{task_name}.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  📊 混淆矩阵已保存: {save_path}")


def eval_model(config):
    """
    评估模型并生成详细报告
    
    Args:
        config: 评估配置字典
    """
    try:
        from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
    except ImportError:
        print("❌ 需要安装 sklearn: pip install scikit-learn")
        return
    
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"\n使用设备: {device}")
    
    # ========== 加载数据 ==========
    print("\n" + "=" * 60)
    print("加载验证数据集...")
    print("=" * 60)
    
    val_dataset = RoboticGraspDataset(
        root_dir=config['data_root'],
        properties_file=config['properties_file'],
        max_tactile_len=config.get('max_tactile_len', 3000),
        mode='val'
    )
    
    from torch.utils.data import DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 16),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    # ========== 加载模型 ==========
    print("\n加载模型...")
    checkpoint_path = config['checkpoint']
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 模型文件不存在: {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 从 checkpoint 获取模型配置
    saved_config = checkpoint.get('config', {})
    
    model = FusionModel(
        fusion_dim=saved_config.get('fusion_dim', 256),
        num_heads=saved_config.get('num_heads', 8),
        dropout=saved_config.get('dropout', 0.1),
        num_layers=saved_config.get('num_layers', 4),
        freeze_visual=True
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"  ✓ 模型加载成功 (来自 Epoch {checkpoint.get('epoch', 'N/A')})")
    
    # ========== 评估 ==========
    print("\n" + "=" * 60)
    print("开始评估...")
    print("=" * 60)
    
    all_preds = {'mass': [], 'stiffness': [], 'material': []}
    all_labels = {'mass': [], 'stiffness': [], 'material': []}
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            tactile = batch['tactile'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            
            outputs = model(images, tactile, padding_mask=padding_mask)
            
            # 收集预测和标签
            all_preds['mass'].extend(outputs['mass'].argmax(dim=1).cpu().tolist())
            all_preds['stiffness'].extend(outputs['stiffness'].argmax(dim=1).cpu().tolist())
            all_preds['material'].extend(outputs['material'].argmax(dim=1).cpu().tolist())
            
            all_labels['mass'].extend(batch['mass'].tolist())
            all_labels['stiffness'].extend(batch['stiffness'].tolist())
            all_labels['material'].extend(batch['material'].tolist())
    
    # ========== 计算指标 ==========
    output_dir = config.get('output_dir', os.path.dirname(checkpoint_path))
    os.makedirs(output_dir, exist_ok=True)
    
    # 标签名称映射
    label_names = {
        'mass': list(val_dataset.mass_to_idx.keys()),
        'stiffness': list(val_dataset.stiffness_to_idx.keys()),
        'material': list(val_dataset.material_to_idx.keys()),
    }
    
    results = {}
    
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    
    for task in ['mass', 'stiffness', 'material']:
        preds = all_preds[task]
        labels = all_labels[task]
        names = label_names[task]
        all_class_labels = list(range(len(names)))  # 确保覆盖所有可能的类别索引
        
        # 计算指标 (使用 labels 参数确保所有类别都被计算，即使验证集中不存在)
        acc = accuracy_score(labels, preds)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, preds, labels=all_class_labels, average=None, zero_division=0
        )
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, preds, labels=all_class_labels, average='macro', zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            labels, preds, labels=all_class_labels, average='weighted', zero_division=0
        )
        
        results[task] = {
            'accuracy': acc,
            'macro': {'precision': precision_macro, 'recall': recall_macro, 'f1': f1_macro},
            'weighted': {'precision': precision_weighted, 'recall': recall_weighted, 'f1': f1_weighted},
            'per_class': {
                names[i]: {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1': float(f1[i]),
                    'support': int(support[i])
                } for i in range(len(names))
            }
        }
        
        # 打印结果
        print(f"\n{'─' * 60}")
        print(f"📊 {task.upper()} 分类结果")
        print(f"{'─' * 60}")
        print(f"  Accuracy: {acc:.4f} ({acc:.2%})")
        print(f"\n  Macro Average:")
        print(f"    Precision: {precision_macro:.4f}")
        print(f"    Recall:    {recall_macro:.4f}")
        print(f"    F1-Score:  {f1_macro:.4f}")
        print(f"\n  Weighted Average:")
        print(f"    Precision: {precision_weighted:.4f}")
        print(f"    Recall:    {recall_weighted:.4f}")
        print(f"    F1-Score:  {f1_weighted:.4f}")
        
        # 详细分类报告
        print(f"\n  详细分类报告:")
        report = classification_report(labels, preds, labels=all_class_labels, target_names=names, digits=4, zero_division=0)
        for line in report.split('\n'):
            print(f"    {line}")
        
        # 绘制混淆矩阵
        plot_confusion_matrices(preds, labels, names, output_dir, task)
    
    # ========== 汇总 ==========
    print("\n" + "=" * 60)
    print("📈 总体性能汇总")
    print("=" * 60)
    
    avg_acc = np.mean([results[t]['accuracy'] for t in ['mass', 'stiffness', 'material']])
    avg_f1_macro = np.mean([results[t]['macro']['f1'] for t in ['mass', 'stiffness', 'material']])
    avg_f1_weighted = np.mean([results[t]['weighted']['f1'] for t in ['mass', 'stiffness', 'material']])
    
    print(f"\n  任务         Accuracy    Macro-F1    Weighted-F1")
    print(f"  {'─' * 50}")
    for task in ['mass', 'stiffness', 'material']:
        print(f"  {task:12} {results[task]['accuracy']:.4f}      "
              f"{results[task]['macro']['f1']:.4f}       "
              f"{results[task]['weighted']['f1']:.4f}")
    print(f"  {'─' * 50}")
    print(f"  {'平均':12} {avg_acc:.4f}      {avg_f1_macro:.4f}       {avg_f1_weighted:.4f}")
    
    # ========== 保存结果 ==========
    results['summary'] = {
        'average_accuracy': avg_acc,
        'average_macro_f1': avg_f1_macro,
        'average_weighted_f1': avg_f1_weighted,
        'checkpoint': checkpoint_path,
        'checkpoint_epoch': checkpoint.get('epoch', None),
        'num_samples': len(val_dataset),
    }
    
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 评估结果已保存: {results_path}")
    print(f"📊 混淆矩阵已保存到: {output_dir}/")
    
    # 绘制综合对比图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Model Evaluation Summary', fontsize=14, fontweight='bold')
    
    tasks = ['mass', 'stiffness', 'material']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    # Accuracy 对比
    accs = [results[t]['accuracy'] for t in tasks]
    axes[0].bar(tasks, accs, color=colors)
    axes[0].set_title('Accuracy by Task')
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel('Accuracy')
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')
    
    # F1-Score 对比
    f1s_macro = [results[t]['macro']['f1'] for t in tasks]
    f1s_weighted = [results[t]['weighted']['f1'] for t in tasks]
    x = np.arange(len(tasks))
    width = 0.35
    axes[1].bar(x - width/2, f1s_macro, width, label='Macro', color='#3498db')
    axes[1].bar(x + width/2, f1s_weighted, width, label='Weighted', color='#e74c3c')
    axes[1].set_title('F1-Score by Task')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tasks)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel('F1-Score')
    axes[1].legend()
    
    # Precision vs Recall
    for i, task in enumerate(tasks):
        axes[2].scatter(results[task]['macro']['recall'], 
                       results[task]['macro']['precision'],
                       s=200, c=colors[i], label=task, marker='o')
    axes[2].set_title('Precision vs Recall (Macro)')
    axes[2].set_xlabel('Recall')
    axes[2].set_ylabel('Precision')
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    summary_path = os.path.join(output_dir, 'evaluation_summary.png')
    fig.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"📊 汇总图已保存: {summary_path}")
    
    print("\n✅ 评估完成!")
    
    return results


# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='训练视觉-触觉融合模型')
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test', 'eval'],
                        help='运行模式: train(训练), test(测试数据集), eval(评估模型)')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--device', type=str, default='cuda', help='设备 (cuda/cpu)')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='模型保存目录')
    parser.add_argument('--checkpoint', type=str, default=None, help='模型检查点路径 (eval模式必需)')
    parser.add_argument('--output_dir', type=str, default=None, help='评估结果输出目录 (默认与checkpoint同目录)')
    parser.add_argument('--no-plot', action='store_true', help='禁用实时绘图')
    
    args = parser.parse_args()
    
    # 数据路径配置 (新结构: Plaintextdataset/train/ 和 Plaintextdataset/val/)
    dataset_path = '/home/martina/Y3_Project/Plaintextdataset'
    # properties_file 现在自动从 train/val 目录下加载，无需手动指定
    
    if args.mode == 'eval':
        # ========== 评估模式 ==========
        if args.checkpoint is None:
            print("❌ 评估模式需要指定 --checkpoint 参数")
            print("   例如: --checkpoint /path/to/best_model.pth")
            exit(1)
        
        config = {
            'data_root': dataset_path,
            'properties_file': None,  # 自动从 val/ 目录加载
            'batch_size': args.batch_size,
            'max_tactile_len': 3000,
            'num_workers': 4,
            'device': args.device,
            'checkpoint': args.checkpoint,
            'output_dir': args.output_dir if args.output_dir else os.path.dirname(args.checkpoint),
        }
        
        eval_model(config)
    
    elif args.mode == 'train':
        # ========== 训练模式 ==========
        config = {
            'data_root': dataset_path,
            'properties_file': None,  # 自动从 train/val 目录加载
            'batch_size': args.batch_size,
            'max_tactile_len': 3000,
            'num_workers': 4,
            
            # 模型配置
            'fusion_dim': 256,
            'num_heads': 8,
            'dropout': 0.1,
            'num_layers': 4,
            'freeze_visual': True,
            
            # 训练配置
            'epochs': args.epochs,
            'lr': args.lr,
            'weight_decay': 0.01,
            'warmup_epochs': 5,
            'device': args.device,
            'save_dir': args.save_dir,
            'save_every': 10,
            'live_plot': not getattr(args, 'no_plot', False),  # 实时绘图开关
        }
        train(config)
    
    elif args.mode == 'test':
        # ========== 测试模式 =========
        print("=" * 50)
        print("测试 RoboticGraspDataset")
        print("=" * 50)
        
        train_dataset = RoboticGraspDataset(
            root_dir=dataset_path,
            properties_file=None,  # 自动从 train/ 目录加载
            max_tactile_len=3000,
            mode='train'
        )
        
        val_dataset = RoboticGraspDataset(
            root_dir=dataset_path,
            properties_file=None,  # 自动从 val/ 目录加载
            max_tactile_len=3000,
            mode='val'
        )
        
        # 测试单个样本
        sample = train_dataset[0]
        print(f"\n样本数据 (训练集):")
        print(f"  图像形状: {sample['image'].shape}")
        print(f"  触觉形状: {sample['tactile'].shape}")
        print(f"  Padding Mask形状: {sample['padding_mask'].shape}")
        print(f"  Mass标签: {sample['mass'].item()} ({train_dataset.idx_to_mass[str(sample['mass'].item())]})")
        print(f"  Stiffness标签: {sample['stiffness'].item()} ({train_dataset.idx_to_stiffness[str(sample['stiffness'].item())]})")
        print(f"  Material标签: {sample['material'].item()} ({train_dataset.idx_to_material[str(sample['material'].item())]})")
        
        # 验证归一化效果
        tactile = sample['tactile']
        valid_mask = ~sample['padding_mask']  # 有效数据位置
        valid_len = valid_mask.sum().item()
        print(f"\n归一化验证:")
        print(f"  有效数据长度: {valid_len}")
        print(f"  触觉数据均值: {tactile[:, :valid_len].mean().item():.4f} (理想≈0)")
        print(f"  触觉数据标准差: {tactile[:, :valid_len].std().item():.4f} (理想≈1)")
        print(f"  触觉数据范围: [{tactile[:, :valid_len].min().item():.4f}, {tactile[:, :valid_len].max().item():.4f}]")
        
        # 打印标签映射
        print("\n标签映射 (训练集):")
        print(f"  Mass: {train_dataset.mass_to_idx}")
        print(f"  Stiffness: {train_dataset.stiffness_to_idx}")
        print(f"  Material: {train_dataset.material_to_idx}")
        
        # 测试DataLoader
        train_loader, val_loader = get_dataloaders(dataset_path, None, batch_size=4)
        
        batch = next(iter(train_loader))
        print(f"\n批次数据:")
        print(f"  图像批次形状: {batch['image'].shape}")
        print(f"  触觉批次形状: {batch['tactile'].shape}")
        print(f"  Padding Mask批次形状: {batch['padding_mask'].shape}")
        print(f"  Mass批次: {batch['mass']}")
        
        # 测试模型前向传播
        print("\n" + "=" * 50)
        print("测试 FusionModel 前向传播 (带 Padding Mask)")
        print("=" * 50)
        
        model = FusionModel()
        
        # 传入 padding_mask
        outputs = model(batch['image'], batch['tactile'], padding_mask=batch['padding_mask'])
        
        print(f"  Mass输出形状: {outputs['mass'].shape}")
        print(f"  Stiffness输出形状: {outputs['stiffness'].shape}")
        print(f"  Material输出形状: {outputs['material'].shape}")
        print("\n✓ 所有测试通过!")

