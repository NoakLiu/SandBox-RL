#!/usr/bin/env python3
"""
LoRA 10 Charts Comparison - Specific Parameters Experiment

This script generates 10 different visualization charts with detailed parameters:
1. Reward convergence curves (6 settings)
2. Learning rate impact analysis  
3. Cooperation vs Competition trade-off
4. Noise level sensitivity
5. Convergence speed distribution
6. Performance stability analysis
7. Adapter specialization patterns
8. Resource utilization efficiency
9. Training dynamics comparison
10. Multi-metric radar charts

Specific Parameters:
- Baseline: Conv=0.03, Noise=0.05
- High Cooperation: Conv=0.025, Coop=0.15, Noise=0.04
- High Competition: Conv=0.035, Comp=0.08, Noise=0.06
- Fast Learning: Conv=0.05, Noise=0.07
- Slow Learning: Conv=0.015, Noise=0.03
- Mixed Strategy: Conv=0.03, Coop=0.08, Comp=0.04, Noise=0.05
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

class LoRA10ChartsAnalyzer:
    """Analyzer for 10 different LoRA comparison charts"""
    
    def __init__(self, num_adapters: int = 8, num_epochs: int = 100):
        self.num_adapters = num_adapters
        self.num_epochs = num_epochs
        
        # Specific parameter configurations
        self.settings = {
            'Baseline': {
                'convergence_rate': 0.03,
                'noise_level': 0.05,
                'cooperation_bonus': 0.0,
                'competition_effect': 0.0,
                'resource_efficiency': 0.8,
                'specialization_factor': 0.1
            },
            'High Cooperation': {
                'convergence_rate': 0.025,
                'noise_level': 0.04,
                'cooperation_bonus': 0.15,
                'competition_effect': 0.0,
                'resource_efficiency': 0.9,
                'specialization_factor': 0.2
            },
            'High Competition': {
                'convergence_rate': 0.035,
                'noise_level': 0.06,
                'cooperation_bonus': 0.0,
                'competition_effect': 0.08,
                'resource_efficiency': 0.7,
                'specialization_factor': 0.3
            },
            'Fast Learning': {
                'convergence_rate': 0.05,
                'noise_level': 0.07,
                'cooperation_bonus': 0.0,
                'competition_effect': 0.0,
                'resource_efficiency': 0.6,
                'specialization_factor': 0.15
            },
            'Slow Learning': {
                'convergence_rate': 0.015,
                'noise_level': 0.03,
                'cooperation_bonus': 0.0,
                'competition_effect': 0.0,
                'resource_efficiency': 0.95,
                'specialization_factor': 0.05
            },
            'Mixed Strategy': {
                'convergence_rate': 0.03,
                'noise_level': 0.05,
                'cooperation_bonus': 0.08,
                'competition_effect': 0.04,
                'resource_efficiency': 0.85,
                'specialization_factor': 0.18
            }
        }
        
        self.convergence_data = {}
    
    def generate_data(self):
        """Generate convergence data for all settings"""
        for setting_name, config in self.settings.items():
            setting_data = {}
            
            for adapter_id in range(1, self.num_adapters + 1):
                base_reward = 0.5
                epochs = np.arange(1, self.num_epochs + 1)
                target_reward = 0.8 + 0.1 * (adapter_id / self.num_adapters)
                
                # Generate reward curve
                reward_curve = base_reward + (target_reward - base_reward) * \
                    (1 - np.exp(-config['convergence_rate'] * epochs))
                
                # Add cooperation effects
                if config['cooperation_bonus'] > 0:
                    if setting_name == 'High Cooperation':
                        team_id = 1 if adapter_id <= 4 else 2
                        cooperation_effect = config['cooperation_bonus'] * \
                            (1 - np.exp(-0.02 * epochs)) * (1.0 if team_id == 1 else 0.8)
                    else:
                        cooperation_effect = config['cooperation_bonus'] * \
                            (1 - np.exp(-0.02 * epochs))
                    reward_curve += cooperation_effect
                
                # Add competition effects
                if config['competition_effect'] > 0:
                    competition_effect = config['competition_effect'] * \
                        np.sin(epochs * 0.1) * (adapter_id / self.num_adapters)
                    reward_curve += competition_effect
                
                # Add noise and smoothing
                noise = np.random.normal(0, config['noise_level'], self.num_epochs)
                smoothed_noise = np.convolve(noise, np.ones(5)/5, mode='same')
                reward_curve += smoothed_noise
                reward_curve = np.convolve(reward_curve, np.ones(3)/3, mode='same')
                reward_curve = np.clip(reward_curve, 0.1, 1.0)
                
                # Ensure upward trend
                for i in range(1, len(reward_curve)):
                    if reward_curve[i] < reward_curve[i-1] - 0.05:
                        reward_curve[i] = reward_curve[i-1] + 0.01
                
                # Additional metrics
                efficiency_curve = config['resource_efficiency'] * \
                    (1 - np.exp(-0.03 * epochs)) + np.random.normal(0, 0.02, self.num_epochs)
                efficiency_curve = np.clip(efficiency_curve, 0.0, 1.0)
                
                specialization_curve = config['specialization_factor'] * \
                    (1 - np.exp(-0.01 * epochs)) + np.random.normal(0, 0.01, self.num_epochs)
                specialization_curve = np.clip(specialization_curve, 0.0, 1.0)
                
                setting_data[f'adapter_{adapter_id}'] = {
                    'rewards': reward_curve,
                    'efficiency': efficiency_curve,
                    'specialization': specialization_curve,
                    'epochs': epochs
                }
            
            self.convergence_data[setting_name] = setting_data
    
    def chart1_reward_convergence(self):
        """Chart 1: Reward convergence curves"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        colors = plt.cm.Set3(np.linspace(0, 1, self.num_adapters))
        
        for setting_idx, (setting_name, config) in enumerate(self.settings.items()):
            ax = axes[setting_idx]
            
            for adapter_id in range(1, self.num_adapters + 1):
                adapter_key = f'adapter_{adapter_id}'
                epochs = self.convergence_data[setting_name][adapter_key]['epochs']
                rewards = self.convergence_data[setting_name][adapter_key]['rewards']
                
                ax.plot(epochs, rewards, color=colors[adapter_id-1], linewidth=2, alpha=0.8)
            
            ax.set_title(f'{setting_name}\nConv={config["convergence_rate"]:.3f}, Noise={config["noise_level"]:.3f}', 
                        fontsize=14, fontweight='bold')
            ax.set_ylabel('Reward', fontsize=12)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.4, 1.0)
        
        plt.tight_layout()
        plt.savefig('chart1_reward_convergence.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def chart2_learning_rate_impact(self):
        """Chart 2: Learning rate impact analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        lr_settings = ['Slow Learning', 'Baseline', 'Fast Learning']
        lr_values = [0.015, 0.03, 0.05]
        final_rewards = []
        convergence_speeds = []
        
        for setting_name in lr_settings:
            setting_rewards = []
            setting_speeds = []
            
            for adapter_id in range(1, self.num_adapters + 1):
                adapter_key = f'adapter_{adapter_id}'
                rewards = self.convergence_data[setting_name][adapter_key]['rewards']
                setting_rewards.append(rewards[-1])
                
                target = 0.9 * rewards[-1]
                convergence_epoch = np.where(rewards >= target)[0]
                convergence_epoch = convergence_epoch[0] if len(convergence_epoch) > 0 else self.num_epochs
                setting_speeds.append(1.0 / (convergence_epoch + 1))
            
            final_rewards.append(np.mean(setting_rewards))
            convergence_speeds.append(np.mean(setting_speeds))
        
        ax1.plot(lr_values, final_rewards, 'o-', linewidth=3, markersize=10, color='blue')
        ax1.set_xlabel('Learning Rate', fontsize=12)
        ax1.set_ylabel('Final Reward', fontsize=12)
        ax1.set_title('Learning Rate vs Final Performance', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(lr_values, convergence_speeds, 's-', linewidth=3, markersize=10, color='red')
        ax2.set_xlabel('Learning Rate', fontsize=12)
        ax2.set_ylabel('Convergence Speed', fontsize=12)
        ax2.set_title('Learning Rate vs Convergence Speed', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('chart2_learning_rate_impact.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def chart3_cooperation_vs_competition(self):
        """Chart 3: Cooperation vs Competition trade-off"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        coop_settings = ['Baseline', 'High Cooperation', 'Mixed Strategy']
        comp_settings = ['Baseline', 'High Competition', 'Mixed Strategy']
        coop_values = [0.0, 0.15, 0.08]
        comp_values = [0.0, 0.0, 0.04]
        
        coop_performance = []
        comp_performance = []
        
        for setting_name in coop_settings:
            setting_rewards = [self.convergence_data[setting_name][f'adapter_{i}']['rewards'][-1] 
                             for i in range(1, self.num_adapters + 1)]
            coop_performance.append(np.mean(setting_rewards))
        
        for setting_name in comp_settings:
            setting_rewards = [self.convergence_data[setting_name][f'adapter_{i}']['rewards'][-1] 
                             for i in range(1, self.num_adapters + 1)]
            comp_performance.append(np.mean(setting_rewards))
        
        ax1.plot(coop_values, coop_performance, 'o-', linewidth=3, markersize=10, color='green')
        ax1.set_xlabel('Cooperation Bonus', fontsize=12)
        ax1.set_ylabel('Final Reward', fontsize=12)
        ax1.set_title('Cooperation Impact', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(comp_values, comp_performance, 's-', linewidth=3, markersize=10, color='orange')
        ax2.set_xlabel('Competition Effect', fontsize=12)
        ax2.set_ylabel('Final Reward', fontsize=12)
        ax2.set_title('Competition Impact', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('chart3_cooperation_vs_competition.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def chart4_noise_sensitivity(self):
        """Chart 4: Noise level sensitivity"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        noise_levels = []
        final_rewards = []
        reward_variances = []
        
        for setting_name, config in self.settings.items():
            noise_levels.append(config['noise_level'])
            
            setting_rewards = [self.convergence_data[setting_name][f'adapter_{i}']['rewards'][-1] 
                             for i in range(1, self.num_adapters + 1)]
            final_rewards.append(np.mean(setting_rewards))
            reward_variances.append(np.std(setting_rewards))
        
        ax1.scatter(noise_levels, final_rewards, s=100, alpha=0.7, c='purple')
        ax1.set_xlabel('Noise Level', fontsize=12)
        ax1.set_ylabel('Final Reward', fontsize=12)
        ax1.set_title('Noise vs Performance', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        ax2.scatter(noise_levels, reward_variances, s=100, alpha=0.7, c='brown')
        ax2.set_xlabel('Noise Level', fontsize=12)
        ax2.set_ylabel('Reward Variance', fontsize=12)
        ax2.set_title('Noise vs Stability', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('chart4_noise_sensitivity.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def chart5_convergence_speed_distribution(self):
        """Chart 5: Convergence speed distribution"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        for setting_idx, (setting_name, _) in enumerate(self.settings.items()):
            ax = axes[setting_idx]
            
            convergence_speeds = []
            for adapter_id in range(1, self.num_adapters + 1):
                adapter_key = f'adapter_{adapter_id}'
                rewards = self.convergence_data[setting_name][adapter_key]['rewards']
                
                target = 0.9 * rewards[-1]
                convergence_epoch = np.where(rewards >= target)[0]
                convergence_epoch = convergence_epoch[0] if len(convergence_epoch) > 0 else self.num_epochs
                convergence_speeds.append(1.0 / (convergence_epoch + 1))
            
            ax.hist(convergence_speeds, bins=5, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_xlabel('Convergence Speed', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'{setting_name}\nConvergence Distribution', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('chart5_convergence_speed_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def chart6_performance_stability(self):
        """Chart 6: Performance stability analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        setting_names = list(self.settings.keys())
        reward_variances = []
        performance_gaps = []
        
        for setting_name in setting_names:
            setting_rewards = [self.convergence_data[setting_name][f'adapter_{i}']['rewards'][-1] 
                             for i in range(1, self.num_adapters + 1)]
            reward_variances.append(np.std(setting_rewards))
            performance_gaps.append(max(setting_rewards) - min(setting_rewards))
        
        bars1 = ax1.bar(setting_names, reward_variances, alpha=0.7, color='lightcoral')
        ax1.set_xlabel('Setting', fontsize=12)
        ax1.set_ylabel('Reward Variance', fontsize=12)
        ax1.set_title('Performance Stability', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        bars2 = ax2.bar(setting_names, performance_gaps, alpha=0.7, color='lightgreen')
        ax2.set_xlabel('Setting', fontsize=12)
        ax2.set_ylabel('Performance Gap', fontsize=12)
        ax2.set_title('Performance Gap', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('chart6_performance_stability.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def chart7_adapter_specialization(self):
        """Chart 7: Adapter specialization patterns"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        for setting_idx, (setting_name, _) in enumerate(self.settings.items()):
            ax = axes[setting_idx]
            
            final_rewards = []
            specialization_scores = []
            
            for adapter_id in range(1, self.num_adapters + 1):
                adapter_key = f'adapter_{adapter_id}'
                rewards = self.convergence_data[setting_name][adapter_key]['rewards']
                specialization = self.convergence_data[setting_name][adapter_key]['specialization']
                
                final_rewards.append(rewards[-1])
                specialization_scores.append(specialization[-1])
            
            ax.scatter(specialization_scores, final_rewards, s=100, alpha=0.7, c='purple')
            ax.set_xlabel('Specialization Score', fontsize=10)
            ax.set_ylabel('Final Reward', fontsize=10)
            ax.set_title(f'{setting_name}\nSpecialization vs Performance', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('chart7_adapter_specialization.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def chart8_resource_efficiency(self):
        """Chart 8: Resource utilization efficiency"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        setting_names = list(self.settings.keys())
        resource_efficiencies = []
        performance_efficiencies = []
        
        for setting_name, config in self.settings.items():
            resource_efficiencies.append(config['resource_efficiency'])
            
            setting_rewards = [self.convergence_data[setting_name][f'adapter_{i}']['rewards'][-1] 
                             for i in range(1, self.num_adapters + 1)]
            setting_efficiencies = [self.convergence_data[setting_name][f'adapter_{i}']['efficiency'][-1] 
                                  for i in range(1, self.num_adapters + 1)]
            
            performance_efficiencies.append(np.mean(setting_rewards) / np.mean(setting_efficiencies))
        
        bars1 = ax1.bar(setting_names, resource_efficiencies, alpha=0.7, color='gold')
        ax1.set_xlabel('Setting', fontsize=12)
        ax1.set_ylabel('Resource Efficiency', fontsize=12)
        ax1.set_title('Resource Utilization', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        bars2 = ax2.bar(setting_names, performance_efficiencies, alpha=0.7, color='lightblue')
        ax2.set_xlabel('Setting', fontsize=12)
        ax2.set_ylabel('Performance Efficiency', fontsize=12)
        ax2.set_title('Performance per Resource', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('chart8_resource_efficiency.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def chart9_training_dynamics(self):
        """Chart 9: Training dynamics comparison"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        for setting_idx, (setting_name, _) in enumerate(self.settings.items()):
            ax = axes[setting_idx]
            
            epochs = self.convergence_data[setting_name]['adapter_1']['epochs']
            avg_rewards = []
            avg_efficiency = []
            
            for epoch_idx in range(len(epochs)):
                epoch_rewards = []
                epoch_efficiency = []
                
                for adapter_id in range(1, self.num_adapters + 1):
                    adapter_key = f'adapter_{adapter_id}'
                    rewards = self.convergence_data[setting_name][adapter_key]['rewards']
                    efficiency = self.convergence_data[setting_name][adapter_key]['efficiency']
                    
                    epoch_rewards.append(rewards[epoch_idx])
                    epoch_efficiency.append(efficiency[epoch_idx])
                
                avg_rewards.append(np.mean(epoch_rewards))
                avg_efficiency.append(np.mean(epoch_efficiency))
            
            ax.plot(epochs, avg_rewards, 'b-', linewidth=2, label='Reward', alpha=0.8)
            ax_twin = ax.twinx()
            ax_twin.plot(epochs, avg_efficiency, 'r--', linewidth=2, label='Efficiency', alpha=0.8)
            
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('Average Reward', fontsize=10, color='blue')
            ax_twin.set_ylabel('Average Efficiency', fontsize=10, color='red')
            ax.set_title(f'{setting_name}\nTraining Dynamics', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('chart9_training_dynamics.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def chart10_radar_charts(self):
        """Chart 10: Multi-metric radar charts"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        metrics = ['Final Reward', 'Convergence Speed', 'Stability', 'Efficiency', 'Specialization']
        
        for setting_idx, (setting_name, _) in enumerate(self.settings.items()):
            ax = axes[setting_idx]
            
            # Calculate metrics
            setting_rewards = [self.convergence_data[setting_name][f'adapter_{i}']['rewards'][-1] 
                             for i in range(1, self.num_adapters + 1)]
            setting_efficiencies = [self.convergence_data[setting_name][f'adapter_{i}']['efficiency'][-1] 
                                  for i in range(1, self.num_adapters + 1)]
            setting_specializations = [self.convergence_data[setting_name][f'adapter_{i}']['specialization'][-1] 
                                     for i in range(1, self.num_adapters + 1)]
            
            # Calculate convergence speeds
            convergence_speeds = []
            for adapter_id in range(1, self.num_adapters + 1):
                adapter_key = f'adapter_{adapter_id}'
                rewards = self.convergence_data[setting_name][adapter_key]['rewards']
                target = 0.9 * rewards[-1]
                convergence_epoch = np.where(rewards >= target)[0]
                convergence_epoch = convergence_epoch[0] if len(convergence_epoch) > 0 else self.num_epochs
                convergence_speeds.append(1.0 / (convergence_epoch + 1))
            
            # Normalize metrics
            final_reward_norm = np.mean(setting_rewards)
            convergence_speed_norm = np.mean(convergence_speeds) * 10
            stability_norm = 1.0 - np.std(setting_rewards)
            efficiency_norm = np.mean(setting_efficiencies)
            specialization_norm = np.mean(setting_specializations)
            
            values = [final_reward_norm, convergence_speed_norm, stability_norm, 
                     efficiency_norm, specialization_norm]
            
            # Create radar chart
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            values += values[:1]
            angles += angles[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, color='blue', alpha=0.7)
            ax.fill(angles, values, alpha=0.25, color='blue')
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics, fontsize=8)
            ax.set_ylim(0, 1)
            ax.set_title(f'{setting_name}\nMulti-Metric Performance', fontsize=12, fontweight='bold')
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('chart10_radar_charts.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main function to generate all 10 charts"""
    print("🚀 LoRA 10 Charts Comparison")
    print("Specific Parameters Experiment")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = LoRA10ChartsAnalyzer(num_adapters=8, num_epochs=100)
    
    # Generate data
    print("Generating convergence data...")
    analyzer.generate_data()
    
    # Generate all 10 charts
    print("Generating 10 charts...")
    
    print("1. Reward convergence curves...")
    analyzer.chart1_reward_convergence()
    
    print("2. Learning rate impact analysis...")
    analyzer.chart2_learning_rate_impact()
    
    print("3. Cooperation vs Competition trade-off...")
    analyzer.chart3_cooperation_vs_competition()
    
    print("4. Noise sensitivity analysis...")
    analyzer.chart4_noise_sensitivity()
    
    print("5. Convergence speed distribution...")
    analyzer.chart5_convergence_speed_distribution()
    
    print("6. Performance stability analysis...")
    analyzer.chart6_performance_stability()
    
    print("7. Adapter specialization patterns...")
    analyzer.chart7_adapter_specialization()
    
    print("8. Resource efficiency analysis...")
    analyzer.chart8_resource_efficiency()
    
    print("9. Training dynamics comparison...")
    analyzer.chart9_training_dynamics()
    
    print("10. Multi-metric radar charts...")
    analyzer.chart10_radar_charts()
    
    print("\n✅ All 10 charts generated!")
    print("📊 Generated files:")
    for i in range(1, 11):
        print(f"  - chart{i}_*.png")


if __name__ == "__main__":
    main()
