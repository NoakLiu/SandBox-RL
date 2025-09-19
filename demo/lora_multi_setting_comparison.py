#!/usr/bin/env python3
"""
LoRA Multi-Setting Comparison - 6 Different Scenarios

This script creates a horizontal comparison of 6 different settings:
1. Baseline (Standard LoRA)
2. High Cooperation (Team-based)
3. High Competition (Adversarial)
4. Fast Learning Rate
5. Slow Learning Rate
6. Mixed Strategy

Each setting shows convergence patterns for 8 LoRA adapters.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import random
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class LoRAMultiSettingAnalyzer:
    """Analyzer for multiple LoRA settings comparison"""
    
    def __init__(self, num_adapters: int = 8, num_epochs: int = 100):
        self.num_adapters = num_adapters
        self.num_epochs = num_epochs
        
        # Define 6 different settings
        self.settings = {
            'Baseline': {
                'description': 'Standard LoRA Training',
                'convergence_rate': 0.03,
                'noise_level': 0.05,
                'cooperation_bonus': 0.0,
                'competition_effect': 0.0,
                'learning_variance': 0.01
            },
            'High Cooperation': {
                'description': 'Team-based Learning',
                'convergence_rate': 0.025,
                'noise_level': 0.04,
                'cooperation_bonus': 0.15,
                'competition_effect': 0.0,
                'learning_variance': 0.008
            },
            'High Competition': {
                'description': 'Adversarial Learning',
                'convergence_rate': 0.035,
                'noise_level': 0.06,
                'cooperation_bonus': 0.0,
                'competition_effect': 0.08,
                'learning_variance': 0.015
            },
            'Fast Learning': {
                'description': 'High Learning Rate',
                'convergence_rate': 0.05,
                'noise_level': 0.07,
                'cooperation_bonus': 0.0,
                'competition_effect': 0.0,
                'learning_variance': 0.02
            },
            'Slow Learning': {
                'description': 'Low Learning Rate',
                'convergence_rate': 0.015,
                'noise_level': 0.03,
                'cooperation_bonus': 0.0,
                'competition_effect': 0.0,
                'learning_variance': 0.005
            },
            'Mixed Strategy': {
                'description': 'Cooperation + Competition',
                'convergence_rate': 0.03,
                'noise_level': 0.05,
                'cooperation_bonus': 0.08,
                'competition_effect': 0.04,
                'learning_variance': 0.012
            }
        }
        
        self.convergence_data = {}
    
    def generate_convergence_data(self) -> Dict[str, Any]:
        """Generate convergence data for all settings"""
        
        for setting_name, setting_config in self.settings.items():
            setting_data = {}
            
            for adapter_id in range(1, self.num_adapters + 1):
                # All adapters start from the same reward
                base_reward = 0.5
                
                # Add some variance to learning rates
                convergence_rate = setting_config['convergence_rate'] + \
                    random.uniform(-setting_config['learning_variance'], 
                                 setting_config['learning_variance'])
                
                # Generate base reward curve
                epochs = np.arange(1, self.num_epochs + 1)
                target_reward = 0.8 + 0.1 * (adapter_id / self.num_adapters)
                reward_curve = base_reward + (target_reward - base_reward) * \
                    (1 - np.exp(-convergence_rate * epochs))
                
                # Add cooperation effects
                if setting_config['cooperation_bonus'] > 0:
                    if setting_name == 'High Cooperation':
                        # Team-based: adapters 1-4 and 5-8 form teams
                        team_id = 1 if adapter_id <= 4 else 2
                        cooperation_effect = setting_config['cooperation_bonus'] * \
                            (1 - np.exp(-0.02 * epochs)) * (1.0 if team_id == 1 else 0.8)
                    else:
                        # General cooperation
                        cooperation_effect = setting_config['cooperation_bonus'] * \
                            (1 - np.exp(-0.02 * epochs))
                    reward_curve += cooperation_effect
                
                # Add competition effects
                if setting_config['competition_effect'] > 0:
                    if setting_name == 'High Competition':
                        # Strong adversarial effects
                        competition_effect = setting_config['competition_effect'] * \
                            np.sin(epochs * 0.15) * (adapter_id / self.num_adapters)
                    else:
                        # Moderate competition
                        competition_effect = setting_config['competition_effect'] * \
                            np.sin(epochs * 0.1) * (adapter_id / self.num_adapters)
                    reward_curve += competition_effect
                
                # Add noise with smoothing
                noise = np.random.normal(0, setting_config['noise_level'], self.num_epochs)
                window_size = 5
                smoothed_noise = np.convolve(noise, np.ones(window_size)/window_size, mode='same')
                reward_curve += smoothed_noise
                
                # Apply final smoothing
                reward_curve = np.convolve(reward_curve, np.ones(3)/3, mode='same')
                reward_curve = np.clip(reward_curve, 0.1, 1.0)
                
                # Ensure upward trend
                for i in range(1, len(reward_curve)):
                    if reward_curve[i] < reward_curve[i-1] - 0.05:
                        reward_curve[i] = reward_curve[i-1] + 0.01
                
                setting_data[f'adapter_{adapter_id}'] = {
                    'rewards': reward_curve,
                    'epochs': epochs
                }
            
            self.convergence_data[setting_name] = setting_data
        
        return self.convergence_data
    
    def plot_horizontal_comparison(self, figsize: Tuple[int, int] = (24, 8)):
        """Plot horizontal comparison of all 6 settings"""
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        colors = plt.cm.Set3(np.linspace(0, 1, self.num_adapters))
        
        for setting_idx, (setting_name, setting_config) in enumerate(self.settings.items()):
            ax = axes[setting_idx]
            
            # Plot all adapters for this setting
            for adapter_id in range(1, self.num_adapters + 1):
                adapter_key = f'adapter_{adapter_id}'
                epochs = self.convergence_data[setting_name][adapter_key]['epochs']
                rewards = self.convergence_data[setting_name][adapter_key]['rewards']
                
                ax.plot(epochs, rewards, 
                       color=colors[adapter_id-1], 
                       linewidth=2, 
                       alpha=0.8,
                       label=f'LoRA {adapter_id}')
            
            # Customize subplot
            ax.set_title(f'{setting_name}\n{setting_config["description"]}', 
                        fontsize=14, fontweight='bold')
            ax.set_ylabel('Reward', fontsize=12)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.4, 1.0)
            
            # Add legend only for first subplot
            if setting_idx == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            
            # Add setting characteristics
            chars_text = f'Conv: {setting_config["convergence_rate"]:.3f}\n'
            chars_text += f'Noise: {setting_config["noise_level"]:.3f}\n'
            if setting_config['cooperation_bonus'] > 0:
                chars_text += f'Coop: {setting_config["cooperation_bonus"]:.3f}\n'
            if setting_config['competition_effect'] > 0:
                chars_text += f'Comp: {setting_config["competition_effect"]:.3f}'
            
            ax.text(0.02, 0.98, chars_text,
                   transform=ax.transAxes, 
                   verticalalignment='top',
                   fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('lora_multi_setting_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_performance_summary(self, figsize: Tuple[int, int] = (20, 12)):
        """Plot performance summary across all settings"""
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for setting_idx, (setting_name, _) in enumerate(self.settings.items()):
            ax = axes[setting_idx]
            
            # Calculate performance metrics for each adapter
            final_rewards = []
            convergence_speeds = []
            
            for adapter_id in range(1, self.num_adapters + 1):
                adapter_key = f'adapter_{adapter_id}'
                rewards = self.convergence_data[setting_name][adapter_key]['rewards']
                
                final_rewards.append(rewards[-1])
                
                # Calculate convergence speed (epoch to reach 90% of final reward)
                target = 0.9 * rewards[-1]
                convergence_epoch = np.where(rewards >= target)[0]
                convergence_epoch = convergence_epoch[0] if len(convergence_epoch) > 0 else self.num_epochs
                convergence_speeds.append(1.0 / (convergence_epoch + 1))
            
            # Create bar chart
            x_pos = np.arange(self.num_adapters)
            bars1 = ax.bar(x_pos - 0.2, final_rewards, 0.4, 
                          label='Final Reward', alpha=0.8, color='skyblue')
            bars2 = ax.bar(x_pos + 0.2, convergence_speeds, 0.4, 
                          label='Convergence Speed', alpha=0.8, color='lightcoral')
            
            ax.set_title(f'{setting_name} Performance', fontsize=14, fontweight='bold')
            ax.set_xlabel('LoRA Adapter', fontsize=12)
            ax.set_ylabel('Performance Metric', fontsize=12)
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f'LoRA {i}' for i in range(1, self.num_adapters + 1)])
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('lora_performance_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_statistical_comparison(self, figsize: Tuple[int, int] = (16, 10)):
        """Plot statistical comparison across settings"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Prepare data
        setting_names = list(self.settings.keys())
        final_rewards_data = []
        convergence_speeds_data = []
        reward_variances_data = []
        improvement_rates_data = []
        
        for setting_name in setting_names:
            setting_final_rewards = []
            setting_convergence_speeds = []
            setting_reward_variances = []
            setting_improvement_rates = []
            
            for adapter_id in range(1, self.num_adapters + 1):
                adapter_key = f'adapter_{adapter_id}'
                rewards = self.convergence_data[setting_name][adapter_key]['rewards']
                
                # Final reward
                setting_final_rewards.append(rewards[-1])
                
                # Convergence speed
                target = 0.9 * rewards[-1]
                convergence_epoch = np.where(rewards >= target)[0]
                convergence_epoch = convergence_epoch[0] if len(convergence_epoch) > 0 else self.num_epochs
                setting_convergence_speeds.append(1.0 / (convergence_epoch + 1))
                
                # Reward variance
                setting_reward_variances.append(np.std(rewards))
                
                # Improvement rate
                setting_improvement_rates.append((rewards[-1] - rewards[0]) / self.num_epochs)
            
            final_rewards_data.append(setting_final_rewards)
            convergence_speeds_data.append(setting_convergence_speeds)
            reward_variances_data.append(setting_reward_variances)
            improvement_rates_data.append(setting_improvement_rates)
        
        # 1. Final Reward Comparison
        ax1 = axes[0, 0]
        bp1 = ax1.boxplot(final_rewards_data, labels=setting_names, patch_artist=True)
        ax1.set_title('Final Reward Distribution', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Final Reward', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # Color the boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(setting_names)))
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
        
        # 2. Convergence Speed Comparison
        ax2 = axes[0, 1]
        bp2 = ax2.boxplot(convergence_speeds_data, labels=setting_names, patch_artist=True)
        ax2.set_title('Convergence Speed Distribution', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Convergence Speed', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        
        for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)
        
        # 3. Reward Variance Comparison
        ax3 = axes[1, 0]
        bp3 = ax3.boxplot(reward_variances_data, labels=setting_names, patch_artist=True)
        ax3.set_title('Reward Stability', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Reward Standard Deviation', fontsize=12)
        ax3.tick_params(axis='x', rotation=45)
        
        for patch, color in zip(bp3['boxes'], colors):
            patch.set_facecolor(color)
        
        # 4. Improvement Rate Comparison
        ax4 = axes[1, 1]
        bp4 = ax4.boxplot(improvement_rates_data, labels=setting_names, patch_artist=True)
        ax4.set_title('Improvement Rate', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Improvement Rate (reward/epoch)', fontsize=12)
        ax4.tick_params(axis='x', rotation=45)
        
        for patch, color in zip(bp4['boxes'], colors):
            patch.set_facecolor(color)
        
        plt.tight_layout()
        plt.savefig('lora_statistical_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_comparison_report(self) -> str:
        """Generate comprehensive comparison report"""
        report = []
        report.append("=" * 80)
        report.append("LoRA MULTI-SETTING COMPARISON REPORT")
        report.append("6 Different Training Strategies")
        report.append("=" * 80)
        report.append("")
        
        for setting_name, setting_config in self.settings.items():
            report.append(f"SETTING: {setting_name}")
            report.append("-" * 50)
            report.append(f"Description: {setting_config['description']}")
            report.append(f"Configuration:")
            report.append(f"  - Convergence Rate: {setting_config['convergence_rate']:.3f}")
            report.append(f"  - Noise Level: {setting_config['noise_level']:.3f}")
            report.append(f"  - Cooperation Bonus: {setting_config['cooperation_bonus']:.3f}")
            report.append(f"  - Competition Effect: {setting_config['competition_effect']:.3f}")
            report.append("")
            
            # Calculate statistics
            setting_final_rewards = []
            setting_convergence_speeds = []
            
            for adapter_id in range(1, self.num_adapters + 1):
                adapter_key = f'adapter_{adapter_id}'
                rewards = self.convergence_data[setting_name][adapter_key]['rewards']
                
                setting_final_rewards.append(rewards[-1])
                
                target = 0.9 * rewards[-1]
                convergence_epoch = np.where(rewards >= target)[0]
                convergence_epoch = convergence_epoch[0] if len(convergence_epoch) > 0 else self.num_epochs
                setting_convergence_speeds.append(1.0 / (convergence_epoch + 1))
            
            report.append(f"Performance Summary:")
            report.append(f"  - Average Final Reward: {np.mean(setting_final_rewards):.4f} ± {np.std(setting_final_rewards):.4f}")
            report.append(f"  - Average Convergence Speed: {np.mean(setting_convergence_speeds):.4f} ± {np.std(setting_convergence_speeds):.4f}")
            report.append(f"  - Best Performer: LoRA {np.argmax(setting_final_rewards) + 1} ({max(setting_final_rewards):.4f})")
            report.append(f"  - Worst Performer: LoRA {np.argmin(setting_final_rewards) + 1} ({min(setting_final_rewards):.4f})")
            report.append("")
        
        # Cross-setting comparison
        report.append("CROSS-SETTING COMPARISON")
        report.append("-" * 50)
        
        # Find best and worst settings
        all_final_rewards = []
        for setting_name in self.settings.keys():
            setting_rewards = []
            for adapter_id in range(1, self.num_adapters + 1):
                adapter_key = f'adapter_{adapter_id}'
                rewards = self.convergence_data[setting_name][adapter_key]['rewards']
                setting_rewards.append(rewards[-1])
            all_final_rewards.append(np.mean(setting_rewards))
        
        best_setting_idx = np.argmax(all_final_rewards)
        worst_setting_idx = np.argmin(all_final_rewards)
        
        report.append(f"Best Overall Setting: {list(self.settings.keys())[best_setting_idx]} "
                     f"(Avg Reward: {all_final_rewards[best_setting_idx]:.4f})")
        report.append(f"Worst Overall Setting: {list(self.settings.keys())[worst_setting_idx]} "
                     f"(Avg Reward: {all_final_rewards[worst_setting_idx]:.4f})")
        report.append(f"Performance Range: {max(all_final_rewards) - min(all_final_rewards):.4f}")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """Main function to run the multi-setting comparison"""
    print("🚀 LoRA Multi-Setting Comparison")
    print("6 Different Training Strategies")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = LoRAMultiSettingAnalyzer(num_adapters=8, num_epochs=100)
    
    # Generate data
    print("Generating convergence data for all settings...")
    analyzer.generate_convergence_data()
    
    # Create visualizations
    print("Creating visualizations...")
    
    # 1. Horizontal comparison
    print("1. Plotting horizontal comparison...")
    analyzer.plot_horizontal_comparison()
    
    # 2. Performance summary
    print("2. Plotting performance summary...")
    analyzer.plot_performance_summary()
    
    # 3. Statistical comparison
    print("3. Plotting statistical comparison...")
    analyzer.plot_statistical_comparison()
    
    # Generate report
    print("4. Generating comparison report...")
    report = analyzer.generate_comparison_report()
    
    # Save report
    with open('lora_multi_setting_report.txt', 'w') as f:
        f.write(report)
    
    print("\n" + report)
    print("\n✅ Multi-setting comparison complete!")
    print("📊 Generated files:")
    print("  - lora_multi_setting_comparison.png")
    print("  - lora_performance_summary.png")
    print("  - lora_statistical_comparison.png")
    print("  - lora_multi_setting_report.txt")


if __name__ == "__main__":
    main()
