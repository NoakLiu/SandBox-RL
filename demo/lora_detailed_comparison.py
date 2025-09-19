#!/usr/bin/env python3
"""
LoRA Detailed Comparison - 10 Different Visualizations with Specific Parameters

This script creates comprehensive comparison with detailed parameters:
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

class LoRADetailedAnalyzer:
    """Detailed analyzer for LoRA comparison with specific parameters"""
    
    def __init__(self, num_adapters: int = 8, num_epochs: int = 100):
        self.num_adapters = num_adapters
        self.num_epochs = num_epochs
        
        # Detailed parameter configurations
        self.settings = {
            'Baseline': {
                'description': 'Standard LoRA Training',
                'convergence_rate': 0.03,
                'noise_level': 0.05,
                'cooperation_bonus': 0.0,
                'competition_effect': 0.0,
                'learning_variance': 0.01,
                'resource_efficiency': 0.8,
                'specialization_factor': 0.1
            },
            'High Cooperation': {
                'description': 'Team-based Learning (Coop=0.15)',
                'convergence_rate': 0.025,
                'noise_level': 0.04,
                'cooperation_bonus': 0.15,
                'competition_effect': 0.0,
                'learning_variance': 0.008,
                'resource_efficiency': 0.9,
                'specialization_factor': 0.2
            },
            'High Competition': {
                'description': 'Adversarial Learning (Comp=0.08)',
                'convergence_rate': 0.035,
                'noise_level': 0.06,
                'cooperation_bonus': 0.0,
                'competition_effect': 0.08,
                'learning_variance': 0.015,
                'resource_efficiency': 0.7,
                'specialization_factor': 0.3
            },
            'Fast Learning': {
                'description': 'High Learning Rate (LR=0.05)',
                'convergence_rate': 0.05,
                'noise_level': 0.07,
                'cooperation_bonus': 0.0,
                'competition_effect': 0.0,
                'learning_variance': 0.02,
                'resource_efficiency': 0.6,
                'specialization_factor': 0.15
            },
            'Slow Learning': {
                'description': 'Low Learning Rate (LR=0.015)',
                'convergence_rate': 0.015,
                'noise_level': 0.03,
                'cooperation_bonus': 0.0,
                'competition_effect': 0.0,
                'learning_variance': 0.005,
                'resource_efficiency': 0.95,
                'specialization_factor': 0.05
            },
            'Mixed Strategy': {
                'description': 'Cooperation + Competition (Coop=0.08, Comp=0.04)',
                'convergence_rate': 0.03,
                'noise_level': 0.05,
                'cooperation_bonus': 0.08,
                'competition_effect': 0.04,
                'learning_variance': 0.012,
                'resource_efficiency': 0.85,
                'specialization_factor': 0.18
            }
        }
        
        self.convergence_data = {}
    
    def generate_convergence_data(self) -> Dict[str, Any]:
        """Generate detailed convergence data with specific parameters"""
        
        for setting_name, setting_config in self.settings.items():
            setting_data = {}
            
            for adapter_id in range(1, self.num_adapters + 1):
                # All adapters start from the same reward
                base_reward = 0.5
                
                # Add variance to learning rates based on parameter
                convergence_rate = setting_config['convergence_rate'] + \
                    random.uniform(-setting_config['learning_variance'], 
                                 setting_config['learning_variance'])
                
                # Generate base reward curve
                epochs = np.arange(1, self.num_epochs + 1)
                target_reward = 0.8 + 0.1 * (adapter_id / self.num_adapters)
                reward_curve = base_reward + (target_reward - base_reward) * \
                    (1 - np.exp(-convergence_rate * epochs))
                
                # Add cooperation effects with specific parameters
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
                
                # Add competition effects with specific parameters
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
                
                # Add noise with specific noise level parameter
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
                
                # Generate additional metrics based on parameters
                efficiency_curve = setting_config['resource_efficiency'] * \
                    (1 - np.exp(-0.03 * epochs)) + np.random.normal(0, 0.02, self.num_epochs)
                efficiency_curve = np.clip(efficiency_curve, 0.0, 1.0)
                
                specialization_curve = setting_config['specialization_factor'] * \
                    (1 - np.exp(-0.01 * epochs)) + np.random.normal(0, 0.01, self.num_epochs)
                specialization_curve = np.clip(specialization_curve, 0.0, 1.0)
                
                setting_data[f'adapter_{adapter_id}'] = {
                    'rewards': reward_curve,
                    'efficiency': efficiency_curve,
                    'specialization': specialization_curve,
                    'epochs': epochs
                }
            
            self.convergence_data[setting_name] = setting_data
        
        return self.convergence_data
    
    def plot_1_reward_convergence(self, figsize: Tuple[int, int] = (20, 12)):
        """Chart 1: Reward convergence curves with detailed parameters"""
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        colors = plt.cm.Set3(np.linspace(0, 1, self.num_adapters))
        
        for setting_idx, (setting_name, setting_config) in enumerate(self.settings.items()):
            ax = axes[setting_idx]
            
            for adapter_id in range(1, self.num_adapters + 1):
                adapter_key = f'adapter_{adapter_id}'
                epochs = self.convergence_data[setting_name][adapter_key]['epochs']
                rewards = self.convergence_data[setting_name][adapter_key]['rewards']
                
                ax.plot(epochs, rewards, 
                       color=colors[adapter_id-1], 
                       linewidth=2, 
                       alpha=0.8,
                       label=f'LoRA {adapter_id}')
            
            ax.set_title(f'{setting_name}\n{setting_config["description"]}', 
                        fontsize=14, fontweight='bold')
            ax.set_ylabel('Reward', fontsize=12)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.4, 1.0)
            
            # Add parameter details
            param_text = f'Conv: {setting_config["convergence_rate"]:.3f}\n'
            param_text += f'Noise: {setting_config["noise_level"]:.3f}\n'
            if setting_config['cooperation_bonus'] > 0:
                param_text += f'Coop: {setting_config["cooperation_bonus"]:.3f}\n'
            if setting_config['competition_effect'] > 0:
                param_text += f'Comp: {setting_config["competition_effect"]:.3f}'
            
            ax.text(0.02, 0.98, param_text,
                   transform=ax.transAxes, 
                   verticalalignment='top',
                   fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('chart1_reward_convergence.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main function to run the detailed comparison"""
    print("🚀 LoRA Detailed Comparison")
    print("10 Different Visualizations with Specific Parameters")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = LoRADetailedAnalyzer(num_adapters=8, num_epochs=100)
    
    # Generate data
    print("Generating detailed convergence data...")
    analyzer.generate_convergence_data()
    
    # Create first visualization
    print("1. Creating reward convergence curves...")
    analyzer.plot_1_reward_convergence()
    
    print("\n✅ First chart complete!")
    print("📊 Generated: chart1_reward_convergence.png")


if __name__ == "__main__":
    main()
