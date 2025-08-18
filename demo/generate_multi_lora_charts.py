#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多卡多LoRA汇报图表生成器
生成详细的汇报图表，包括性能对比、资源利用、LoRA效果等
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random
from pathlib import Path

# Set font for better display
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 设置图表样式
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class MultiLoRAChartGenerator:
    """多卡多LoRA图表生成器"""
    
    def __init__(self, output_dir="visualization_outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 设置随机种子确保可重现
        np.random.seed(42)
        random.seed(42)
        
        # 生成模拟数据
        self.generate_simulation_data()
    
    def generate_simulation_data(self):
        """生成模拟数据"""
        # 8 GPU data
        self.gpu_data = {
            'gpu_id': list(range(8)),
            'memory_used_gb': np.random.uniform(14, 20, 8),
            'memory_total_gb': [24] * 8,
            'utilization_percent': np.random.uniform(75, 92, 8),
            'temperature_celsius': np.random.uniform(55, 68, 8),
            'power_watts': np.random.uniform(200, 320, 8)
        }
        
        # 8 LoRA data
        self.lora_data = {
            'lora_id': [f'lora_{i+1}' for i in range(8)],
            'rank': np.random.choice([8, 16, 32, 64], 8),
            'alpha': np.random.uniform(16, 64, 8),
            'accuracy_improvement': np.random.uniform(3.5, 8.2, 8),
            'inference_speed_tps': np.random.uniform(120, 180, 8),
            'memory_overhead_mb': np.random.uniform(80, 150, 8),
            'training_time_hours': np.random.uniform(3, 6, 8),
            'convergence_epochs': np.random.randint(5, 15, 8)
        }
        
        # Time series data (24 hours)
        self.time_data = {
            'timestamp': [datetime.now() - timedelta(hours=23-i) for i in range(24)],
            'total_requests': np.random.poisson(1200, 24),
            'successful_requests': np.random.poisson(1150, 24),
            'average_latency_ms': np.random.uniform(80, 120, 24),
            'gpu_utilization_avg': np.random.uniform(78, 88, 24),
            'memory_usage_avg_gb': np.random.uniform(16, 19, 24)
        }
        
        # Multi-model comparison data
        self.model_comparison = {
            'model_name': ['Base Model', 'LoRA-1', 'LoRA-2', 'LoRA-3', 'LoRA-4', 'LoRA-5', 'LoRA-6', 'LoRA-7', 'LoRA-8'],
            'accuracy': [82.5, 84.8, 85.2, 84.1, 86.3, 85.7, 85.9, 83.8, 86.1],
            'speed_tps': [150, 145, 142, 148, 138, 144, 141, 147, 136],
            'memory_gb': [12.0, 12.3, 12.6, 12.2, 12.8, 12.5, 12.7, 12.1, 12.9],
            'cost_per_request': [0.10, 0.11, 0.12, 0.11, 0.13, 0.12, 0.12, 0.11, 0.13]
        }
    
    def plot_gpu_utilization_heatmap(self):
        """绘制GPU利用率热力图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # GPU利用率柱状图
        gpu_df = pd.DataFrame(self.gpu_data)
        bars = ax1.bar(gpu_df['gpu_id'], gpu_df['utilization_percent'], 
                      color=plt.cm.viridis(gpu_df['utilization_percent']/100))
        ax1.set_xlabel('GPU ID')
        ax1.set_ylabel('Utilization (%)')
        ax1.set_title('8-GPU Utilization Distribution')
        ax1.set_ylim(0, 100)
        
        # 添加数值标签
        for bar, util in zip(bars, gpu_df['utilization_percent']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{util:.1f}%', ha='center', va='bottom')
        
        # GPU内存使用热力图
        memory_matrix = np.array(gpu_df['memory_used_gb']).reshape(2, 4)
        im = ax2.imshow(memory_matrix, cmap='YlOrRd', aspect='auto')
        ax2.set_title('GPU Memory Usage Heatmap (GB)')
        ax2.set_xlabel('GPU Column')
        ax2.set_ylabel('GPU Row')
        
        # 添加数值标签
        for i in range(2):
            for j in range(4):
                ax2.text(j, i, f'{memory_matrix[i, j]:.1f}GB', 
                        ha='center', va='center', color='white', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'gpu_utilization_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_lora_performance_comparison(self):
        """绘制LoRA性能对比图"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        lora_df = pd.DataFrame(self.lora_data)
        
        # 1. Accuracy improvement comparison
        bars1 = ax1.bar(lora_df['lora_id'], lora_df['accuracy_improvement'], 
                       color=plt.cm.plasma(lora_df['accuracy_improvement']/8))
        ax1.set_xlabel('LoRA ID')
        ax1.set_ylabel('Accuracy Improvement (%)')
        ax1.set_title('LoRA Accuracy Improvement Comparison')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Inference speed comparison
        bars2 = ax2.bar(lora_df['lora_id'], lora_df['inference_speed_tps'], 
                       color=plt.cm.cool(lora_df['inference_speed_tps']/180))
        ax2.set_xlabel('LoRA ID')
        ax2.set_ylabel('Inference Speed (tokens/sec)')
        ax2.set_title('LoRA Inference Speed Comparison')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Memory overhead comparison
        bars3 = ax3.bar(lora_df['lora_id'], lora_df['memory_overhead_mb'], 
                       color=plt.cm.spring(lora_df['memory_overhead_mb']/150))
        ax3.set_xlabel('LoRA ID')
        ax3.set_ylabel('Memory Overhead (MB)')
        ax3.set_title('LoRA Memory Overhead Comparison')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Rank vs Performance scatter plot
        scatter = ax4.scatter(lora_df['rank'], lora_df['accuracy_improvement'], 
                            s=lora_df['memory_overhead_mb']*2, 
                            c=lora_df['inference_speed_tps'], cmap='viridis', alpha=0.7)
        ax4.set_xlabel('LoRA Rank')
        ax4.set_ylabel('Accuracy Improvement (%)')
        ax4.set_title('Rank vs Performance Relationship')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Inference Speed (tokens/sec)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'lora_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_time_series_metrics(self):
        """绘制时间序列指标图"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        time_df = pd.DataFrame(self.time_data)
        
        # 1. Request volume time series
        ax1.plot(time_df['timestamp'], time_df['total_requests'], 'b-', linewidth=2, label='Total Requests')
        ax1.plot(time_df['timestamp'], time_df['successful_requests'], 'g-', linewidth=2, label='Successful Requests')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Request Count')
        ax1.set_title('24-Hour Request Volume Changes')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Average latency time series
        ax2.plot(time_df['timestamp'], time_df['average_latency_ms'], 'r-', linewidth=2)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Average Latency (ms)')
        ax2.set_title('24-Hour Average Latency Changes')
        ax2.grid(True, alpha=0.3)
        
        # 3. GPU utilization time series
        ax3.plot(time_df['timestamp'], time_df['gpu_utilization_avg'], 'purple', linewidth=2)
        ax3.set_xlabel('Time')
        ax3.set_ylabel('GPU Utilization (%)')
        ax3.set_title('24-Hour GPU Utilization Changes')
        ax3.grid(True, alpha=0.3)
        
        # 4. Memory usage time series
        ax4.plot(time_df['timestamp'], time_df['memory_usage_avg_gb'], 'orange', linewidth=2)
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Memory Usage (GB)')
        ax4.set_title('24-Hour Memory Usage Changes')
        ax4.grid(True, alpha=0.3)
        
        # 格式化x轴时间
        for ax in [ax1, ax2, ax3, ax4]:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'time_series_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison_radar(self):
        """绘制模型对比雷达图"""
        model_df = pd.DataFrame(self.model_comparison)
        
        # 标准化数据 (0-1)
        metrics = ['accuracy', 'speed_tps', 'memory_gb', 'cost_per_request']
        normalized_data = {}
        
        for metric in metrics:
            min_val = model_df[metric].min()
            max_val = model_df[metric].max()
            normalized_data[metric] = (model_df[metric] - min_val) / (max_val - min_val)
        
        # 创建雷达图
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        # 绘制每个模型的雷达图
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_df)))
        
        for idx, (_, row) in enumerate(model_df.iterrows()):
            values = [normalized_data[metric][idx] for metric in metrics]
            values += values[:1]  # 闭合图形
            
            ax.plot(angles, values, 'o-', linewidth=2, label=row['model_name'], color=colors[idx])
            ax.fill(angles, values, alpha=0.1, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(['Accuracy', 'Speed', 'Memory', 'Cost'])
        ax.set_ylim(0, 1)
        ax.set_title('Multi-Model Performance Comparison Radar Chart', size=16, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison_radar.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_detailed_tables(self):
        """创建详细的数据表格"""
        # 1. GPU详细信息表格
        gpu_df = pd.DataFrame(self.gpu_data)
        gpu_df['memory_usage_percent'] = (gpu_df['memory_used_gb'] / gpu_df['memory_total_gb']) * 100
        gpu_df['power_efficiency'] = gpu_df['utilization_percent'] / gpu_df['power_watts'] * 100
        
        # 2. LoRA详细信息表格
        lora_df = pd.DataFrame(self.lora_data)
        lora_df['efficiency_score'] = (lora_df['accuracy_improvement'] * lora_df['inference_speed_tps']) / lora_df['memory_overhead_mb']
        
        # 3. 时间序列汇总表格
        time_df = pd.DataFrame(self.time_data)
        time_df['success_rate'] = (time_df['successful_requests'] / time_df['total_requests']) * 100
        
        # 保存表格为CSV
        gpu_df.to_csv(self.output_dir / 'gpu_detailed_stats.csv', index=False, encoding='utf-8-sig')
        lora_df.to_csv(self.output_dir / 'lora_detailed_stats.csv', index=False, encoding='utf-8-sig')
        time_df.to_csv(self.output_dir / 'time_series_stats.csv', index=False, encoding='utf-8-sig')
        
        # Create summary table
        summary_data = {
            'Metric': [
                'GPU Average Utilization',
                'GPU Average Memory Usage',
                'GPU Average Temperature',
                'GPU Average Power',
                'LoRA Average Accuracy Improvement',
                'LoRA Average Inference Speed',
                'LoRA Average Memory Overhead',
                'System Average Request Success Rate',
                'System Average Latency'
            ],
            'Value': [
                f"{gpu_df['utilization_percent'].mean():.1f}%",
                f"{gpu_df['memory_used_gb'].mean():.1f} GB",
                f"{gpu_df['temperature_celsius'].mean():.1f}°C",
                f"{gpu_df['power_watts'].mean():.1f} W",
                f"{lora_df['accuracy_improvement'].mean():.1f}%",
                f"{lora_df['inference_speed_tps'].mean():.1f} tokens/sec",
                f"{lora_df['memory_overhead_mb'].mean():.1f} MB",
                f"{time_df['success_rate'].mean():.1f}%",
                f"{time_df['average_latency_ms'].mean():.1f} ms"
            ],
            'Max': [
                f"{gpu_df['utilization_percent'].max():.1f}%",
                f"{gpu_df['memory_used_gb'].max():.1f} GB",
                f"{gpu_df['temperature_celsius'].max():.1f}°C",
                f"{gpu_df['power_watts'].max():.1f} W",
                f"{lora_df['accuracy_improvement'].max():.1f}%",
                f"{lora_df['inference_speed_tps'].max():.1f} tokens/sec",
                f"{lora_df['memory_overhead_mb'].max():.1f} MB",
                f"{time_df['success_rate'].max():.1f}%",
                f"{time_df['average_latency_ms'].max():.1f} ms"
            ],
            'Min': [
                f"{gpu_df['utilization_percent'].min():.1f}%",
                f"{gpu_df['memory_used_gb'].min():.1f} GB",
                f"{gpu_df['temperature_celsius'].min():.1f}°C",
                f"{gpu_df['power_watts'].min():.1f} W",
                f"{lora_df['accuracy_improvement'].min():.1f}%",
                f"{lora_df['inference_speed_tps'].min():.1f} tokens/sec",
                f"{lora_df['memory_overhead_mb'].min():.1f} MB",
                f"{time_df['success_rate'].min():.1f}%",
                f"{time_df['average_latency_ms'].min():.1f} ms"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.output_dir / 'system_summary_stats.csv', index=False, encoding='utf-8-sig')
        
        return summary_df
    
    def generate_all_charts(self):
        """Generate all charts"""
        print("🎨 Starting to generate multi-GPU multi-LoRA report charts...")
        
        # Generate various charts
        self.plot_gpu_utilization_heatmap()
        print("✅ GPU utilization heatmap generated")
        
        self.plot_lora_performance_comparison()
        print("✅ LoRA performance comparison chart generated")
        
        self.plot_time_series_metrics()
        print("✅ Time series metrics chart generated")
        
        self.plot_model_comparison_radar()
        print("✅ Model comparison radar chart generated")
        
        # Create detailed tables
        summary_df = self.create_detailed_tables()
        print("✅ Detailed data tables generated")
        
        print(f"\n📊 All charts saved to: {self.output_dir}")
        print("📋 Generated files:")
        for file in self.output_dir.glob("*"):
            print(f"  - {file.name}")
        
        return summary_df


def main():
    """Main function"""
    print("🚀 Multi-GPU Multi-LoRA Report Chart Generator")
    print("=" * 50)
    
    # Create chart generator
    generator = MultiLoRAChartGenerator()
    
    # Generate all charts
    summary_df = generator.generate_all_charts()
    
    # Display summary statistics
    print("\n📈 System Summary Statistics:")
    print(summary_df.to_string(index=False))
    
    print("\n🎉 Chart generation completed!")


if __name__ == "__main__":
    main()
