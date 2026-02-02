#!/usr/bin/env python3
"""
数据集统计分析脚本
遍历 Plaintextdataset 目录，生成详细的统计学信息报告
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image


def load_episode(episode_path: Path):
    """加载单个 episode 的所有数据"""
    metadata_path = episode_path / "metadata.json"
    tactile_path = episode_path / "tactile_data.pkl"
    image_path = episode_path / "visual_anchor.jpg"
    
    data = {}
    
    # 加载 metadata
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            data['metadata'] = json.load(f)
    
    # 加载触觉数据
    if tactile_path.exists():
        with open(tactile_path, 'rb') as f:
            data['tactile'] = pickle.load(f)
    
    # 获取图像信息
    if image_path.exists():
        img = Image.open(image_path)
        data['image_size'] = img.size
        data['image_mode'] = img.mode
    
    return data


def compute_array_stats(arr):
    """计算数组的统计信息"""
    arr = np.array(arr)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'median': float(np.median(arr)),
        'q25': float(np.percentile(arr, 25)),
        'q75': float(np.percentile(arr, 75)),
    }


def analyze_dataset(dataset_root: str, output_dir: str = None):
    """
    分析整个数据集
    
    Args:
        dataset_root: Plaintextdataset 目录路径
        output_dir: 输出目录，默认为 dataset_root/analysis_output
    """
    dataset_root = Path(dataset_root)
    if output_dir is None:
        output_dir = dataset_root / "analysis_output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 统计容器
    stats = {
        'summary': {
            'total_episodes': 0,
            'total_samples': 0,
            'total_duration_s': 0,
            'classes': {},
            'analysis_time': datetime.now().isoformat(),
        },
        'per_class': {},
        'global_tactile_stats': {},
    }
    
    # 用于全局统计的累积器
    all_durations = []
    all_num_samples = []
    all_sampling_rates = []
    
    # 每个类别的累积器
    class_data = defaultdict(lambda: {
        'episodes': [],
        'durations': [],
        'num_samples': [],
        'sampling_rates': [],
        'tactile_arrays': defaultdict(list),  # 用于存储所有触觉数据以计算全局统计
    })
    
    # 全局触觉数据累积器
    global_tactile = defaultdict(list)
    
    # 遍历所有类别目录
    class_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir() and not d.name.startswith('.')])
    
    print("=" * 70)
    print(f"数据集分析: {dataset_root}")
    print("=" * 70)
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        if class_name == 'analysis_output':
            continue
            
        print(f"\n处理类别: {class_name}")
        
        # 获取该类别下的所有 episode
        episode_dirs = sorted([d for d in class_dir.iterdir() if d.is_dir() and d.name.startswith('episode_')])
        
        for episode_dir in episode_dirs:
            try:
                data = load_episode(episode_dir)
                
                if 'metadata' not in data or 'tactile' not in data:
                    print(f"  警告: {episode_dir.name} 数据不完整，跳过")
                    continue
                
                metadata = data['metadata']
                tactile = data['tactile']
                
                # 基本统计
                duration = metadata.get('duration_s', 0)
                num_samples = metadata.get('num_samples', 0)
                sampling_rate = metadata.get('real_sampling_rate', 0)
                
                all_durations.append(duration)
                all_num_samples.append(num_samples)
                all_sampling_rates.append(sampling_rate)
                
                class_data[class_name]['episodes'].append(episode_dir.name)
                class_data[class_name]['durations'].append(duration)
                class_data[class_name]['num_samples'].append(num_samples)
                class_data[class_name]['sampling_rates'].append(sampling_rate)
                
                # 触觉数据统计
                for key, value in tactile.items():
                    if key == 'timestamps':
                        continue
                    
                    arr = np.array(value)
                    if arr.ndim == 1:
                        global_tactile[key].extend(arr.tolist())
                        class_data[class_name]['tactile_arrays'][key].extend(arr.tolist())
                    elif arr.ndim == 2:
                        # 对于多关节数据，分别统计每个关节
                        for joint_idx in range(arr.shape[1]):
                            joint_key = f"{key}_joint{joint_idx}"
                            global_tactile[joint_key].extend(arr[:, joint_idx].tolist())
                            class_data[class_name]['tactile_arrays'][joint_key].extend(arr[:, joint_idx].tolist())
                
                stats['summary']['total_episodes'] += 1
                stats['summary']['total_samples'] += num_samples
                stats['summary']['total_duration_s'] += duration
                
            except Exception as e:
                print(f"  错误处理 {episode_dir.name}: {e}")
        
        # 更新类别计数
        stats['summary']['classes'][class_name] = len(class_data[class_name]['episodes'])
        print(f"  找到 {len(class_data[class_name]['episodes'])} 个 episodes")
    
    # 计算每个类别的统计信息
    print("\n" + "=" * 70)
    print("计算统计信息...")
    print("=" * 70)
    
    for class_name, cdata in class_data.items():
        stats['per_class'][class_name] = {
            'num_episodes': len(cdata['episodes']),
            'duration': compute_array_stats(cdata['durations']) if cdata['durations'] else {},
            'num_samples': compute_array_stats(cdata['num_samples']) if cdata['num_samples'] else {},
            'sampling_rate': compute_array_stats(cdata['sampling_rates']) if cdata['sampling_rates'] else {},
            'tactile_stats': {},
        }
        
        # 触觉数据统计
        for tac_key, tac_values in cdata['tactile_arrays'].items():
            if tac_values:
                stats['per_class'][class_name]['tactile_stats'][tac_key] = compute_array_stats(tac_values)
    
    # 计算全局触觉统计
    for tac_key, tac_values in global_tactile.items():
        if tac_values:
            stats['global_tactile_stats'][tac_key] = compute_array_stats(tac_values)
    
    # 全局统计
    stats['summary']['duration_stats'] = compute_array_stats(all_durations) if all_durations else {}
    stats['summary']['num_samples_stats'] = compute_array_stats(all_num_samples) if all_num_samples else {}
    stats['summary']['sampling_rate_stats'] = compute_array_stats(all_sampling_rates) if all_sampling_rates else {}
    
    # 保存 JSON 报告
    json_path = output_dir / "dataset_statistics.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n统计报告已保存: {json_path}")
    
    # 生成可视化
    generate_visualizations(stats, class_data, output_dir)
    
    # 打印摘要报告
    print_summary_report(stats)
    
    return stats


def generate_visualizations(stats, class_data, output_dir):
    """生成可视化图表"""
    output_dir = Path(output_dir)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 类别分布饼图
    fig, ax = plt.subplots(figsize=(10, 8))
    classes = list(stats['summary']['classes'].keys())
    counts = list(stats['summary']['classes'].values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
    
    wedges, texts, autotexts = ax.pie(counts, labels=classes, autopct='%1.1f%%', 
                                       colors=colors, startangle=90)
    ax.set_title('Class Distribution (Episode Count)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'class_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. 每类别的样本数量分布
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 2a. Episode数量条形图
    ax = axes[0, 0]
    x = range(len(classes))
    ax.bar(x, counts, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_ylabel('Number of Episodes')
    ax.set_title('Episodes per Class')
    
    # 2b. 平均采样数量
    ax = axes[0, 1]
    mean_samples = [stats['per_class'][c]['num_samples']['mean'] for c in classes]
    std_samples = [stats['per_class'][c]['num_samples']['std'] for c in classes]
    ax.bar(x, mean_samples, yerr=std_samples, color=colors, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Samples per Episode (Mean ± Std)')
    
    # 2c. 平均时长
    ax = axes[1, 0]
    mean_durations = [stats['per_class'][c]['duration']['mean'] for c in classes]
    std_durations = [stats['per_class'][c]['duration']['std'] for c in classes]
    ax.bar(x, mean_durations, yerr=std_durations, color=colors, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_ylabel('Duration (seconds)')
    ax.set_title('Episode Duration (Mean ± Std)')
    
    # 2d. 采样率
    ax = axes[1, 1]
    mean_rates = [stats['per_class'][c]['sampling_rate']['mean'] for c in classes]
    std_rates = [stats['per_class'][c]['sampling_rate']['std'] for c in classes]
    ax.bar(x, mean_rates, yerr=std_rates, color=colors, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_ylabel('Sampling Rate (Hz)')
    ax.set_title('Real Sampling Rate (Mean ± Std)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_class_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. 触觉数据分布 (箱线图 - 不显示离群点)
    # 选择关键的触觉特征
    key_features = ['gripper_width_profile', 'load_profile', 'lift_current_profile', 'gripper_velocity_profile']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, feature in enumerate(key_features):
        ax = axes[idx // 2, idx % 2]
        
        box_data = []
        box_labels = []
        for class_name in classes:
            if feature in class_data[class_name]['tactile_arrays']:
                # 采样以避免数据过多
                data = np.array(class_data[class_name]['tactile_arrays'][feature])
                if len(data) > 10000:
                    data = np.random.choice(data, 10000, replace=False)
                box_data.append(data)
                box_labels.append(class_name[:12])  # 截断长标签
        
        if box_data:
            # showfliers=False 隐藏离群点，使图表更清晰
            bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True, 
                           showfliers=False, notch=True, widths=0.6)
            for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            # 设置中位数线颜色
            for median in bp['medians']:
                median.set_color('black')
                median.set_linewidth(2)
            ax.set_xticklabels(box_labels, rotation=45, ha='right')
            ax.set_title(f'{feature}')
            ax.set_ylabel('Value')
            ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'tactile_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3b. 触觉数据分布 (小提琴图 - 展示完整分布形状)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for idx, feature in enumerate(key_features):
        ax = axes[idx // 2, idx % 2]
        
        violin_data = []
        violin_labels = []
        for class_name in classes:
            if feature in class_data[class_name]['tactile_arrays']:
                data = np.array(class_data[class_name]['tactile_arrays'][feature])
                # 采样以加快绘制速度
                if len(data) > 5000:
                    data = np.random.choice(data, 5000, replace=False)
                violin_data.append(data)
                violin_labels.append(class_name[:10])
        
        if violin_data:
            parts = ax.violinplot(violin_data, positions=range(len(violin_data)), 
                                  showmeans=True, showmedians=True, showextrema=False)
            
            # 设置颜色
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i % len(colors)])
                pc.set_alpha(0.7)
                pc.set_edgecolor('black')
            
            # 设置均值和中位数线的样式
            parts['cmeans'].set_color('red')
            parts['cmeans'].set_linewidth(2)
            parts['cmedians'].set_color('black')
            parts['cmedians'].set_linewidth(2)
            
            ax.set_xticks(range(len(violin_labels)))
            ax.set_xticklabels(violin_labels, rotation=45, ha='right')
            ax.set_title(f'{feature}')
            ax.set_ylabel('Value')
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            
            # 添加图例
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], color='red', linewidth=2, label='Mean'),
                             Line2D([0], [0], color='black', linewidth=2, label='Median')]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'tactile_distributions_violin.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. 关节数据热力图 (均值)
    joint_features = ['joint_position_profile', 'joint_load_profile', 'joint_current_profile', 'joint_velocity_profile']
    num_joints = 6
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, base_feature in enumerate(joint_features):
        ax = axes[idx // 2, idx % 2]
        
        heatmap_data = np.zeros((len(classes), num_joints))
        
        for class_idx, class_name in enumerate(classes):
            for joint_idx in range(num_joints):
                key = f"{base_feature}_joint{joint_idx}"
                if key in stats['per_class'][class_name]['tactile_stats']:
                    heatmap_data[class_idx, joint_idx] = stats['per_class'][class_name]['tactile_stats'][key]['mean']
        
        im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlBu_r')
        ax.set_xticks(range(num_joints))
        ax.set_xticklabels([f'J{i}' for i in range(num_joints)])
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels([c[:15] for c in classes])
        ax.set_title(f'{base_feature} (Mean)')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'joint_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"可视化图表已保存到: {output_dir}")


def print_summary_report(stats):
    """打印摘要报告"""
    print("\n")
    print("=" * 70)
    print("                        数据集统计摘要报告")
    print("=" * 70)
    
    s = stats['summary']
    print(f"\n【总体统计】")
    print(f"  总 Episode 数:  {s['total_episodes']}")
    print(f"  总样本数:       {s['total_samples']:,}")
    print(f"  总时长:         {s['total_duration_s']:.2f} 秒 ({s['total_duration_s']/60:.2f} 分钟)")
    print(f"  类别数:         {len(s['classes'])}")
    
    print(f"\n【Episode 时长统计】")
    if s['duration_stats']:
        d = s['duration_stats']
        print(f"  平均: {d['mean']:.2f}s | 标准差: {d['std']:.2f}s | 范围: [{d['min']:.2f}s, {d['max']:.2f}s]")
    
    print(f"\n【每 Episode 样本数统计】")
    if s['num_samples_stats']:
        d = s['num_samples_stats']
        print(f"  平均: {d['mean']:.1f} | 标准差: {d['std']:.1f} | 范围: [{d['min']:.0f}, {d['max']:.0f}]")
    
    print(f"\n【采样率统计】")
    if s['sampling_rate_stats']:
        d = s['sampling_rate_stats']
        print(f"  平均: {d['mean']:.2f} Hz | 标准差: {d['std']:.2f} Hz")
    
    print(f"\n【各类别详情】")
    print("-" * 70)
    print(f"{'类别名称':<25} {'Episodes':>10} {'平均样本数':>12} {'平均时长(s)':>12}")
    print("-" * 70)
    
    for class_name, class_stats in stats['per_class'].items():
        n_eps = class_stats['num_episodes']
        mean_samples = class_stats['num_samples'].get('mean', 0)
        mean_duration = class_stats['duration'].get('mean', 0)
        print(f"{class_name:<25} {n_eps:>10} {mean_samples:>12.1f} {mean_duration:>12.2f}")
    
    print("-" * 70)
    
    print(f"\n【关键触觉特征全局统计】")
    key_features = ['gripper_width_profile', 'load_profile', 'lift_current_profile']
    for feature in key_features:
        if feature in stats['global_tactile_stats']:
            d = stats['global_tactile_stats'][feature]
            print(f"  {feature}:")
            print(f"    均值: {d['mean']:.4f} | 标准差: {d['std']:.4f} | 范围: [{d['min']:.4f}, {d['max']:.4f}]")
    
    print("\n" + "=" * 70)
    print("分析完成!")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='分析 Plaintextdataset 数据集')
    parser.add_argument('--dataset', type=str, 
                        default='/home/martina/Y3_Project/Plaintextdataset',
                        help='数据集根目录路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录路径 (默认为数据集目录下的 analysis_output)')
    
    args = parser.parse_args()
    
    analyze_dataset(args.dataset, args.output)

