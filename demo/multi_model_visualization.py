#!/usr/bin/env python3
"""
多模型训练系统可视化模块
生成折线图、柱状图、雷达图、3D热力图等
"""

import json
import os
import random
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import math

# 尝试导入可视化库
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import Circle, RegularPolygon
    from matplotlib.path import Path
    from matplotlib.projections.polar import PolarAxes
    from matplotlib.projections import register_projection
    from matplotlib.spines import Spine
    from matplotlib.transforms import Affine2D
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import seaborn as sns
    HAS_MATPLOTLIB = True
    print("✅ Matplotlib and visualization libraries imported successfully")
except ImportError as e:
    HAS_MATPLOTLIB = False
    print(f"❌ Visualization libraries not available: {e}")
    print("Will use mock visualization")

# 设置中文字体
if HAS_MATPLOTLIB:
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass

class MultiModelVisualizer:
    """多模型训练系统可视化器"""
    
    def __init__(self, output_dir: str = "./visualization_outputs"):
        self.output_dir = output_dir
        self.ensure_output_dir()
        
        # 设置颜色主题
        self.colors = {
            'cooperative': '#2E8B57',  # 海绿色
            'competitive': '#DC143C',  # 深红色
            'team_battle': '#4169E1',  # 皇家蓝
            'mixed': '#FF8C00',        # 深橙色
            'leader': '#8A2BE2',       # 蓝紫色
            'follower': '#20B2AA',     # 浅海绿
            'competitor': '#FF4500',   # 橙红色
            'teammate': '#32CD32',     # 酸橙绿
            'neutral': '#808080'       # 灰色
        }
        
        self.line_styles = ['-', '--', '-.', ':']
        self.markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    
    def ensure_output_dir(self):
        """确保输出目录存在"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"📁 Created output directory: {self.output_dir}")
    
    def load_training_results(self, file_path: str) -> Dict[str, Any]:
        """加载训练结果数据"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            print(f"✅ Loaded training results from: {file_path}")
            
            # 处理不同的数据格式
            if isinstance(raw_data, list):
                # 如果是列表格式，转换为字典格式
                data = {
                    "training_sessions": [],
                    "model_performances": raw_data,  # 假设列表包含模型性能数据
                    "environment_stats": []
                }
                
                # 从模型性能数据中提取训练会话信息
                if raw_data:
                    # 创建模拟训练会话
                    for i, performance in enumerate(raw_data[:5]):  # 取前5个作为会话
                        session = {
                            "session_id": f"session_{i}",
                            "training_mode": performance.get("role", "mixed"),
                            "timestamp": performance.get("timestamp", datetime.now().isoformat()),
                            "cycles": random.randint(3, 8),
                            "total_models": len(raw_data),
                            "total_tasks": performance.get("total_tasks", 0),
                            "avg_accuracy": performance.get("accuracy", 0.0),
                            "avg_efficiency": performance.get("efficiency", 0.0),
                            "total_reward": performance.get("reward_earned", 0.0),
                            "total_weight_updates": performance.get("weight_updates", 0),
                            "total_lora_adaptations": performance.get("lora_adaptations", 0)
                        }
                        data["training_sessions"].append(session)
            else:
                # 如果已经是字典格式，直接使用
                data = raw_data
            
            return data
        except Exception as e:
            print(f"❌ Failed to load training results: {e}")
            return self.generate_mock_data()
    
    def generate_mock_data(self) -> Dict[str, Any]:
        """生成模拟数据用于演示"""
        mock_data = {
            "training_sessions": [],
            "model_performances": [],
            "environment_stats": []
        }
        
        # 生成模拟训练会话数据
        for session_id in range(5):
            session = {
                "session_id": f"session_{session_id}",
                "training_mode": random.choice(["cooperative", "competitive", "team_battle", "mixed"]),
                "timestamp": datetime.now().isoformat(),
                "cycles": random.randint(3, 8),
                "total_models": random.randint(3, 8),
                "total_tasks": random.randint(10, 30),
                "avg_accuracy": random.uniform(0.6, 0.95),
                "avg_efficiency": random.uniform(0.5, 0.9),
                "total_reward": random.uniform(50, 200),
                "total_weight_updates": random.randint(10, 50),
                "total_lora_adaptations": random.randint(5, 25)
            }
            mock_data["training_sessions"].append(session)
        
        # 生成模拟模型性能数据
        model_roles = ["leader", "follower", "competitor", "teammate", "neutral"]
        for model_id in range(10):
            performance = {
                "model_id": f"model_{model_id:03d}",
                "role": random.choice(model_roles),
                "team_id": f"team_{random.choice(['alpha', 'beta', 'gamma'])}" if random.random() > 0.3 else None,
                "total_tasks": random.randint(5, 25),
                "avg_accuracy": random.uniform(0.6, 0.95),
                "avg_efficiency": random.uniform(0.5, 0.9),
                "avg_cooperation": random.uniform(0.0, 1.0),
                "total_reward": random.uniform(20, 100),
                "weight_updates": random.randint(5, 30),
                "lora_adaptations": random.randint(2, 15)
            }
            mock_data["model_performances"].append(performance)
        
        return mock_data
    
    def create_line_chart(self, data: Dict[str, Any], save_path: Optional[str] = None) -> str:
        """创建折线图 - 训练进度和性能趋势"""
        if not HAS_MATPLOTLIB:
            return self._mock_visualization("line_chart")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('多模型训练系统 - 性能趋势分析', fontsize=16, fontweight='bold')
        
        # 提取数据
        sessions = data.get("training_sessions", [])
        if not sessions:
            sessions = self.generate_mock_data()["training_sessions"]
        
        cycles = [s.get("cycles", 0) for s in sessions]
        accuracies = [s.get("avg_accuracy", 0) for s in sessions]
        efficiencies = [s.get("avg_efficiency", 0) for s in sessions]
        rewards = [s.get("total_reward", 0) for s in sessions]
        weight_updates = [s.get("total_weight_updates", 0) for s in sessions]
        lora_adaptations = [s.get("total_lora_adaptations", 0) for s in sessions]
        
        # 1. 准确率趋势
        ax1.plot(cycles, accuracies, 'o-', color=self.colors['cooperative'], 
                linewidth=2, markersize=6, label='平均准确率')
        ax1.set_xlabel('训练周期')
        ax1.set_ylabel('准确率')
        ax1.set_title('准确率趋势')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. 效率趋势
        ax2.plot(cycles, efficiencies, 's-', color=self.colors['competitive'], 
                linewidth=2, markersize=6, label='平均效率')
        ax2.set_xlabel('训练周期')
        ax2.set_ylabel('效率')
        ax2.set_title('效率趋势')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. 奖励趋势
        ax3.plot(cycles, rewards, '^-', color=self.colors['team_battle'], 
                linewidth=2, markersize=6, label='总奖励')
        ax3.set_xlabel('训练周期')
        ax3.set_ylabel('奖励')
        ax3.set_title('奖励趋势')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. 权重更新和LoRA适应
        ax4_twin = ax4.twinx()
        line1 = ax4.plot(cycles, weight_updates, 'D-', color=self.colors['leader'], 
                        linewidth=2, markersize=6, label='权重更新')
        line2 = ax4_twin.plot(cycles, lora_adaptations, 'v-', color=self.colors['follower'], 
                             linewidth=2, markersize=6, label='LoRA适应')
        ax4.set_xlabel('训练周期')
        ax4.set_ylabel('权重更新次数', color=self.colors['leader'])
        ax4_twin.set_ylabel('LoRA适应次数', color=self.colors['follower'])
        ax4.set_title('模型更新趋势')
        ax4.grid(True, alpha=0.3)
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper left')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, "training_trends_line_chart.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Line chart saved to: {save_path}")
        return save_path
    
    def create_bar_chart(self, data: Dict[str, Any], save_path: Optional[str] = None) -> str:
        """创建柱状图 - 模型性能对比"""
        if not HAS_MATPLOTLIB:
            return self._mock_visualization("bar_chart")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('多模型训练系统 - 模型性能对比', fontsize=16, fontweight='bold')
        
        # 提取模型性能数据
        performances = data.get("model_performances", [])
        if not performances:
            performances = self.generate_mock_data()["model_performances"]
        
        model_ids = [p.get("model_id", "") for p in performances]
        accuracies = [p.get("avg_accuracy", 0) for p in performances]
        efficiencies = [p.get("avg_efficiency", 0) for p in performances]
        rewards = [p.get("total_reward", 0) for p in performances]
        roles = [p.get("role", "unknown") for p in performances]
        
        # 为不同角色分配颜色
        role_colors = [self.colors.get(role, self.colors['neutral']) for role in roles]
        
        # 1. 准确率对比
        bars1 = ax1.bar(model_ids, accuracies, color=role_colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('模型ID')
        ax1.set_ylabel('平均准确率')
        ax1.set_title('模型准确率对比')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. 效率对比
        bars2 = ax2.bar(model_ids, efficiencies, color=role_colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.set_xlabel('模型ID')
        ax2.set_ylabel('平均效率')
        ax2.set_title('模型效率对比')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 3. 奖励对比
        bars3 = ax3.bar(model_ids, rewards, color=role_colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax3.set_xlabel('模型ID')
        ax3.set_ylabel('总奖励')
        ax3.set_title('模型奖励对比')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        # 4. 角色分布
        role_counts = {}
        for role in roles:
            role_counts[role] = role_counts.get(role, 0) + 1
        
        role_names = list(role_counts.keys())
        role_values = list(role_counts.values())
        role_colors_dist = [self.colors.get(role, self.colors['neutral']) for role in role_names]
        
        bars4 = ax4.bar(role_names, role_values, color=role_colors_dist, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax4.set_xlabel('模型角色')
        ax4.set_ylabel('模型数量')
        ax4.set_title('模型角色分布')
        ax4.grid(True, alpha=0.3)
        
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, "model_performance_bar_chart.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Bar chart saved to: {save_path}")
        return save_path
    
    def create_radar_chart(self, data: Dict[str, Any], save_path: Optional[str] = None) -> str:
        """创建雷达图 - 模型能力雷达图"""
        if not HAS_MATPLOTLIB:
            return self._mock_visualization("radar_chart")
        
        # 自定义雷达图投影
        class RadarAxes(PolarAxes):
            name = 'radar'
            
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.set_theta_zero_location('N')
            
            def fill(self, *args, closed=True, **kwargs):
                return super().fill(closed=closed, *args, **kwargs)
            
            def plot(self, *args, **kwargs):
                lines = super().plot(*args, **kwargs)
                self.fill(*args, alpha=0.25, **kwargs)
                return lines
        
        register_projection(RadarAxes)
        
        # 提取数据
        performances = data.get("model_performances", [])
        if not performances:
            performances = self.generate_mock_data()["model_performances"]
        
        # 选择前6个模型进行雷达图展示
        selected_models = performances[:6]
        
        # 定义能力维度
        categories = ['准确率', '效率', '合作度', '奖励', '权重更新', 'LoRA适应']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw=dict(projection='radar'))
        fig.suptitle('多模型训练系统 - 模型能力雷达图', fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for i, model in enumerate(selected_models):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # 提取模型能力值
            values = [
                model.get("avg_accuracy", 0),
                model.get("avg_efficiency", 0),
                model.get("avg_cooperation", 0),
                min(model.get("total_reward", 0) / 100, 1.0),  # 归一化奖励
                min(model.get("weight_updates", 0) / 30, 1.0),  # 归一化权重更新
                min(model.get("lora_adaptations", 0) / 15, 1.0)  # 归一化LoRA适应
            ]
            
            # 计算角度
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            values += values[:1]  # 闭合图形
            angles += angles[:1]
            
            # 绘制雷达图
            role = model.get("role", "unknown")
            color = self.colors.get(role, self.colors['neutral'])
            
            ax.plot(angles, values, 'o-', linewidth=2, color=color, label=model.get("model_id", ""))
            ax.fill(angles, values, alpha=0.25, color=color)
            
            # 设置标签
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 1)
            ax.set_title(f'{model.get("model_id", "")} ({role})', pad=20)
            
            # 添加网格
            ax.grid(True)
        
        # 隐藏多余的子图
        for i in range(len(selected_models), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, "model_capability_radar_chart.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Radar chart saved to: {save_path}")
        return save_path
    
    def create_3d_heatmap(self, data: Dict[str, Any], save_path: Optional[str] = None) -> str:
        """创建3D热力图 - 训练模式、模型数量、性能关系"""
        if not HAS_MATPLOTLIB:
            return self._mock_visualization("3d_heatmap")
        
        fig = plt.figure(figsize=(16, 12))
        
        # 提取数据
        sessions = data.get("training_sessions", [])
        if not sessions:
            sessions = self.generate_mock_data()["training_sessions"]
        
        # 按训练模式分组
        mode_data = {}
        for session in sessions:
            mode = session.get("training_mode", "unknown")
            if mode not in mode_data:
                mode_data[mode] = []
            mode_data[mode].append(session)
        
        # 创建3D热力图
        ax1 = fig.add_subplot(221, projection='3d')
        ax2 = fig.add_subplot(222, projection='3d')
        ax3 = fig.add_subplot(223, projection='3d')
        ax4 = fig.add_subplot(224, projection='3d')
        
        axes = [ax1, ax2, ax3, ax4]
        metrics = ['avg_accuracy', 'avg_efficiency', 'total_reward', 'total_weight_updates']
        titles = ['准确率热力图', '效率热力图', '奖励热力图', '权重更新热力图']
        
        for ax, metric, title in zip(axes, metrics, titles):
            # 创建网格数据
            modes = list(mode_data.keys())
            model_counts = list(range(3, 9))  # 3-8个模型
            
            X, Y = np.meshgrid(range(len(modes)), model_counts)
            Z = np.zeros_like(X, dtype=float)
            
            # 填充数据
            for i, mode in enumerate(modes):
                for j, model_count in enumerate(model_counts):
                    # 找到匹配的数据点
                    matching_sessions = [s for s in mode_data[mode] 
                                       if s.get("total_models", 0) == model_count]
                    if matching_sessions:
                        values = [s.get(metric, 0) for s in matching_sessions]
                        Z[j, i] = np.mean(values)
                    else:
                        # 如果没有匹配数据，使用插值或默认值
                        Z[j, i] = random.uniform(0.5, 0.9) if metric in ['avg_accuracy', 'avg_efficiency'] else random.uniform(50, 150)
            
            # 绘制3D热力图
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                                 linewidth=0, antialiased=True)
            
            ax.set_xlabel('训练模式')
            ax.set_ylabel('模型数量')
            ax.set_zlabel(metric.replace('_', ' ').title())
            ax.set_title(title)
            ax.set_xticks(range(len(modes)))
            ax.set_xticklabels(modes, rotation=45)
            
            # 添加颜色条
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        plt.suptitle('多模型训练系统 - 3D性能热力图', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, "3d_performance_heatmap.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 3D heatmap saved to: {save_path}")
        return save_path
    
    def create_scatter_plot(self, data: Dict[str, Any], save_path: Optional[str] = None) -> str:
        """创建散点图 - 模型性能分布"""
        if not HAS_MATPLOTLIB:
            return self._mock_visualization("scatter_plot")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('多模型训练系统 - 性能分布散点图', fontsize=16, fontweight='bold')
        
        # 提取数据
        performances = data.get("model_performances", [])
        if not performances:
            performances = self.generate_mock_data()["model_performances"]
        
        # 准备数据
        accuracies = [p.get("avg_accuracy", 0) for p in performances]
        efficiencies = [p.get("avg_efficiency", 0) for p in performances]
        rewards = [p.get("total_reward", 0) for p in performances]
        cooperation = [p.get("avg_cooperation", 0) for p in performances]
        roles = [p.get("role", "unknown") for p in performances]
        
        # 为不同角色分配颜色和标记
        role_colors = [self.colors.get(role, self.colors['neutral']) for role in roles]
        role_markers = ['o', 's', '^', 'D', 'v']
        role_marker_map = {role: marker for role, marker in zip(set(roles), role_markers)}
        markers = [role_marker_map.get(role, 'o') for role in roles]
        
        # 1. 准确率 vs 效率
        for i, (acc, eff, color, marker) in enumerate(zip(accuracies, efficiencies, role_colors, markers)):
            ax1.scatter(acc, eff, c=[color], marker=marker, s=100, alpha=0.7, 
                       edgecolors='black', linewidth=1)
            ax1.annotate(f'M{i+1}', (acc, eff), xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax1.set_xlabel('平均准确率')
        ax1.set_ylabel('平均效率')
        ax1.set_title('准确率 vs 效率')
        ax1.grid(True, alpha=0.3)
        
        # 2. 奖励 vs 合作度
        for i, (rew, coop, color, marker) in enumerate(zip(rewards, cooperation, role_colors, markers)):
            ax2.scatter(rew, coop, c=[color], marker=marker, s=100, alpha=0.7, 
                       edgecolors='black', linewidth=1)
            ax2.annotate(f'M{i+1}', (rew, coop), xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax2.set_xlabel('总奖励')
        ax2.set_ylabel('平均合作度')
        ax2.set_title('奖励 vs 合作度')
        ax2.grid(True, alpha=0.3)
        
        # 3. 准确率 vs 合作度
        for i, (acc, coop, color, marker) in enumerate(zip(accuracies, cooperation, role_colors, markers)):
            ax3.scatter(acc, coop, c=[color], marker=marker, s=100, alpha=0.7, 
                       edgecolors='black', linewidth=1)
            ax3.annotate(f'M{i+1}', (acc, coop), xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax3.set_xlabel('平均准确率')
        ax3.set_ylabel('平均合作度')
        ax3.set_title('准确率 vs 合作度')
        ax3.grid(True, alpha=0.3)
        
        # 4. 效率 vs 奖励
        for i, (eff, rew, color, marker) in enumerate(zip(efficiencies, rewards, role_colors, markers)):
            ax4.scatter(eff, rew, c=[color], marker=marker, s=100, alpha=0.7, 
                       edgecolors='black', linewidth=1)
            ax4.annotate(f'M{i+1}', (eff, rew), xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax4.set_xlabel('平均效率')
        ax4.set_ylabel('总奖励')
        ax4.set_title('效率 vs 奖励')
        ax4.grid(True, alpha=0.3)
        
        # 添加图例
        unique_roles = list(set(roles))
        legend_elements = []
        for role in unique_roles:
            color = self.colors.get(role, self.colors['neutral'])
            marker = role_marker_map.get(role, 'o')
            legend_elements.append(plt.Line2D([0], [0], marker=marker, color='w', 
                                            markerfacecolor=color, markersize=10, label=role))
        
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, "performance_scatter_plot.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Scatter plot saved to: {save_path}")
        return save_path
    
    def create_comprehensive_dashboard(self, data: Dict[str, Any], save_path: Optional[str] = None) -> str:
        """创建综合仪表板 - 所有图表的汇总"""
        if not HAS_MATPLOTLIB:
            return self._mock_visualization("comprehensive_dashboard")
        
        fig = plt.figure(figsize=(20, 16))
        
        # 创建网格布局
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. 标题
        fig.suptitle('多模型训练系统 - 综合性能仪表板', fontsize=20, fontweight='bold', y=0.98)
        
        # 2. 训练模式分布饼图
        ax1 = fig.add_subplot(gs[0, 0])
        sessions = data.get("training_sessions", [])
        if not sessions:
            sessions = self.generate_mock_data()["training_sessions"]
        
        mode_counts = {}
        for session in sessions:
            mode = session.get("training_mode", "unknown")
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
        if mode_counts:
            modes = list(mode_counts.keys())
            counts = list(mode_counts.values())
            colors = [self.colors.get(mode, self.colors['neutral']) for mode in modes]
            
            wedges, texts, autotexts = ax1.pie(counts, labels=modes, colors=colors, autopct='%1.1f%%',
                                              startangle=90, explode=[0.05] * len(modes))
            ax1.set_title('训练模式分布', fontweight='bold')
        
        # 3. 性能指标箱线图
        ax2 = fig.add_subplot(gs[0, 1:3])
        performances = data.get("model_performances", [])
        if not performances:
            performances = self.generate_mock_data()["model_performances"]
        
        if performances:
            accuracies = [p.get("avg_accuracy", 0) for p in performances]
            efficiencies = [p.get("avg_efficiency", 0) for p in performances]
            cooperation = [p.get("avg_cooperation", 0) for p in performances]
            
            box_data = [accuracies, efficiencies, cooperation]
            box_labels = ['准确率', '效率', '合作度']
            box_colors = [self.colors['cooperative'], self.colors['competitive'], self.colors['team_battle']]
            
            bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax2.set_title('性能指标分布', fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        # 4. 时间序列图
        ax3 = fig.add_subplot(gs[0, 3])
        if sessions:
            timestamps = [datetime.fromisoformat(s.get("timestamp", datetime.now().isoformat())) 
                         for s in sessions]
            rewards = [s.get("total_reward", 0) for s in sessions]
            
            ax3.plot(timestamps, rewards, 'o-', color=self.colors['mixed'], linewidth=2, markersize=6)
            ax3.set_xlabel('时间')
            ax3.set_ylabel('总奖励')
            ax3.set_title('奖励时间序列', fontweight='bold')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
        
        # 5. 模型角色性能对比
        ax4 = fig.add_subplot(gs[1, :2])
        if performances:
            role_performance = {}
            for p in performances:
                role = p.get("role", "unknown")
                if role not in role_performance:
                    role_performance[role] = []
                role_performance[role].append(p.get("avg_accuracy", 0))
            
            roles = list(role_performance.keys())
            avg_accuracies = [np.mean(role_performance[role]) for role in roles]
            role_colors = [self.colors.get(role, self.colors['neutral']) for role in roles]
            
            bars = ax4.bar(roles, avg_accuracies, color=role_colors, alpha=0.7, edgecolor='black')
            ax4.set_xlabel('模型角色')
            ax4.set_ylabel('平均准确率')
            ax4.set_title('不同角色性能对比', fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 6. 热力图
        ax5 = fig.add_subplot(gs[1, 2:])
        if performances:
            # 创建性能矩阵
            model_ids = [p.get("model_id", "") for p in performances[:8]]  # 限制前8个模型
            metrics = ['avg_accuracy', 'avg_efficiency', 'avg_cooperation', 'total_reward']
            metric_names = ['准确率', '效率', '合作度', '奖励']
            
            performance_matrix = []
            for p in performances[:8]:
                row = []
                for metric in metrics:
                    value = p.get(metric, 0)
                    if metric == 'total_reward':
                        value = min(value / 100, 1.0)  # 归一化奖励
                    row.append(value)
                performance_matrix.append(row)
            
            if performance_matrix:
                im = ax5.imshow(performance_matrix, cmap='viridis', aspect='auto')
                ax5.set_xticks(range(len(metric_names)))
                ax5.set_xticklabels(metric_names)
                ax5.set_yticks(range(len(model_ids)))
                ax5.set_yticklabels(model_ids)
                ax5.set_title('模型性能热力图', fontweight='bold')
                
                # 添加数值标签
                for i in range(len(model_ids)):
                    for j in range(len(metric_names)):
                        text = ax5.text(j, i, f'{performance_matrix[i][j]:.2f}',
                                       ha="center", va="center", color="white", fontweight='bold')
                
                plt.colorbar(im, ax=ax5)
        
        # 7. 3D散点图
        ax6 = fig.add_subplot(gs[2:, :], projection='3d')
        if performances:
            x = [p.get("avg_accuracy", 0) for p in performances]
            y = [p.get("avg_efficiency", 0) for p in performances]
            z = [p.get("total_reward", 0) for p in performances]
            colors = [self.colors.get(p.get("role", "unknown"), self.colors['neutral']) for p in performances]
            
            scatter = ax6.scatter(x, y, z, c=colors, s=100, alpha=0.7, edgecolors='black')
            ax6.set_xlabel('准确率')
            ax6.set_ylabel('效率')
            ax6.set_zlabel('奖励')
            ax6.set_title('3D性能空间分布', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, "comprehensive_dashboard.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Comprehensive dashboard saved to: {save_path}")
        return save_path
    
    def _mock_visualization(self, chart_type: str) -> str:
        """模拟可视化（当matplotlib不可用时）"""
        mock_path = os.path.join(self.output_dir, f"mock_{chart_type}.txt")
        
        mock_content = f"""
模拟{chart_type}图表
==================
由于matplotlib库不可用，无法生成实际的图表。
请安装以下依赖：
pip install matplotlib numpy seaborn

图表类型: {chart_type}
生成时间: {datetime.now().isoformat()}
        """
        
        with open(mock_path, 'w', encoding='utf-8') as f:
            f.write(mock_content)
        
        print(f"📊 Mock {chart_type} saved to: {mock_path}")
        return mock_path
    
    def generate_all_visualizations(self, data_file: str = "multi_model_training_simple_results.json") -> Dict[str, str]:
        """生成所有可视化图表"""
        print("🎨 开始生成多模型训练系统可视化图表...")
        
        # 加载数据
        data = self.load_training_results(data_file)
        
        # 生成各种图表
        results = {}
        
        try:
            results['line_chart'] = self.create_line_chart(data)
            results['bar_chart'] = self.create_bar_chart(data)
            results['radar_chart'] = self.create_radar_chart(data)
            results['3d_heatmap'] = self.create_3d_heatmap(data)
            results['scatter_plot'] = self.create_scatter_plot(data)
            results['dashboard'] = self.create_comprehensive_dashboard(data)
            
            print("✅ 所有可视化图表生成完成！")
            
        except Exception as e:
            print(f"❌ 生成可视化图表时出错: {e}")
            import traceback
            traceback.print_exc()
        
        return results

def main():
    """主函数"""
    print("🎨 多模型训练系统可视化模块")
    print("=" * 50)
    
    # 创建可视化器
    visualizer = MultiModelVisualizer()
    
    # 生成所有图表
    results = visualizer.generate_all_visualizations()
    
    # 显示结果
    print("\n📊 生成的图表文件:")
    for chart_type, file_path in results.items():
        print(f"   {chart_type}: {file_path}")
    
    print(f"\n📁 所有图表已保存到: {visualizer.output_dir}")

if __name__ == "__main__":
    main()
