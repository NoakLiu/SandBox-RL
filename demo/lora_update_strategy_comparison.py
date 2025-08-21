#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA Update Strategy Comparison Analysis
Compare different LoRA update strategies: independent, team-based, and adversarial updates
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

# Set chart style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class LoRAUpdateStrategyAnalyzer:
    """LoRA Update Strategy Comparison Analyzer"""
    
    def __init__(self, output_dir="visualization_outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set random seed for reproducibility
        np.random.seed(42)
        random.seed(42)
        
        # Generate simulation data
        self.generate_strategy_data()
    
    def generate_strategy_data(self):
        """Generate simulation data for different update strategies"""
        # Time points (100 epochs)
        self.epochs = np.arange(1, 101)
        
        # Strategy 1: No Independent Updates (Baseline)
        # All LoRAs remain static, slight random fluctuations
        self.baseline_rewards = []
        for i in range(8):  # 8 LoRAs
            base_reward = 0.5  # Start from 0.5
            trend = np.random.uniform(-0.0005, 0.0005, 100)  # Very slight random trend
            noise = np.random.normal(0, 0.01, 100)
            rewards = base_reward + np.cumsum(trend) + noise
            self.baseline_rewards.append(rewards)
        
        # Strategy 2: Independent Updates (Slowest)
        # Each LoRA updates independently with different learning patterns
        self.independent_rewards = []
        for i in range(8):
            base_reward = 0.5  # Start from 0.5
            # Different learning curves for each LoRA (slowest learning)
            if i < 2:  # Fast learners
                learning_rate = np.random.uniform(0.002, 0.003)
                plateau_epoch = np.random.randint(60, 80)
            elif i < 5:  # Medium learners
                learning_rate = np.random.uniform(0.0015, 0.002)
                plateau_epoch = np.random.randint(70, 90)
            else:  # Slow learners
                learning_rate = np.random.uniform(0.001, 0.0015)
                plateau_epoch = np.random.randint(80, 95)
            
            # Sigmoid-like learning curve
            progress = 1 / (1 + np.exp(-learning_rate * (self.epochs - plateau_epoch)))
            max_improvement = np.random.uniform(0.20, 0.30)
            rewards = base_reward + max_improvement * progress + np.random.normal(0, 0.012, 100)
            self.independent_rewards.append(rewards)
        
        # Strategy 3: Team-based Updates (Medium speed)
        # LoRAs cooperate in teams, shared knowledge benefits
        self.team_rewards = []
        team_groups = [[0, 1, 2, 3], [4, 5, 6, 7]]  # 2 teams
        
        for i in range(8):
            base_reward = 0.5  # Start from 0.5
            # Find which team this LoRA belongs to
            team_idx = None
            for j, team in enumerate(team_groups):
                if i in team:
                    team_idx = j
                    break
            
            # Team synergy effect (medium learning speed)
            if team_idx == 0:  # Team 1
                synergy_factor = 1.3
                learning_rate = 0.003
                plateau_epoch = np.random.randint(40, 60)
            else:  # Team 2
                synergy_factor = 1.2
                learning_rate = 0.0025
                plateau_epoch = np.random.randint(45, 65)
            
            # Cooperative learning curve
            progress = 1 / (1 + np.exp(-learning_rate * (self.epochs - plateau_epoch)))
            max_improvement = np.random.uniform(0.25, 0.35) * synergy_factor
            rewards = base_reward + max_improvement * progress + np.random.normal(0, 0.010, 100)
            self.team_rewards.append(rewards)
        
        # Strategy 4: Adversarial Updates (Fastest)
        # LoRAs compete for resources, some win, some lose
        self.adversarial_rewards = []
        winners = [0, 2, 4, 6]  # Even indices are winners
        losers = [1, 3, 5, 7]   # Odd indices are losers
        
        for i in range(8):
            base_reward = 0.5  # Start from 0.5
            
            if i in winners:
                # Winners: aggressive learning, high rewards (fastest)
                learning_rate = np.random.uniform(0.004, 0.006)
                plateau_epoch = np.random.randint(25, 45)
                max_improvement = np.random.uniform(0.30, 0.40)
                # Winners get resource advantages
                resource_boost = np.random.uniform(0.05, 0.12)
            else:
                # Losers: slower learning, lower rewards
                learning_rate = np.random.uniform(0.002, 0.004)
                plateau_epoch = np.random.randint(35, 55)
                max_improvement = np.random.uniform(0.15, 0.25)
                # Losers suffer resource constraints
                resource_boost = np.random.uniform(-0.08, 0.03)
            
            # Competitive learning curve with resource effects
            progress = 1 / (1 + np.exp(-learning_rate * (self.epochs - plateau_epoch)))
            rewards = base_reward + max_improvement * progress + resource_boost + np.random.normal(0, 0.015, 100)
            self.adversarial_rewards.append(rewards)
    
    def plot_reward_comparison(self):
        """Plot reward comparison across different strategies"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Baseline (No Updates)
        ax1.set_title('Strategy 1: No Independent Updates (Baseline)', fontsize=14, fontweight='bold')
        for i in range(8):
            ax1.plot(self.epochs, self.baseline_rewards[i], 
                    label=f'LoRA-{i+1}', linewidth=2, alpha=0.8)
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Reward')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0.5, 1.0)
        
        # 2. Independent Updates
        ax2.set_title('Strategy 2: Independent Updates', fontsize=14, fontweight='bold')
        for i in range(8):
            ax2.plot(self.epochs, self.independent_rewards[i], 
                    label=f'LoRA-{i+1}', linewidth=2, alpha=0.8)
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Reward')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0.5, 1.0)
        
        # 3. Team-based Updates
        ax3.set_title('Strategy 3: Team-based Updates', fontsize=14, fontweight='bold')
        colors = ['#1f77b4', '#ff7f0e']  # Different colors for 2 teams
        for i in range(8):
            team_idx = 0 if i < 4 else 1  # Team 1: LoRA 1-4, Team 2: LoRA 5-8
            ax3.plot(self.epochs, self.team_rewards[i], 
                    label=f'LoRA-{i+1}', linewidth=2, alpha=0.8, color=colors[team_idx])
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Reward')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0.5, 1.0)
        
        # 4. Adversarial Updates
        ax4.set_title('Strategy 4: Adversarial Updates', fontsize=14, fontweight='bold')
        for i in range(8):
            if i in [0, 2, 4, 6]:  # Winners
                ax4.plot(self.epochs, self.adversarial_rewards[i], 
                        label=f'LoRA-{i+1} (Winner)', linewidth=2, alpha=0.8, color='green')
            else:  # Losers
                ax4.plot(self.epochs, self.adversarial_rewards[i], 
                        label=f'LoRA-{i+1} (Loser)', linewidth=2, alpha=0.8, color='red')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Reward')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0.5, 1.0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'lora_strategy_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_average_performance_trends(self):
        """Plot average performance trends for each strategy"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Calculate average rewards for each strategy
        baseline_avg = np.mean(self.baseline_rewards, axis=0)
        independent_avg = np.mean(self.independent_rewards, axis=0)
        team_avg = np.mean(self.team_rewards, axis=0)
        adversarial_avg = np.mean(self.adversarial_rewards, axis=0)
        
        # 1. Average Performance Comparison
        ax1.plot(self.epochs, baseline_avg, label='No Updates (Baseline)', 
                linewidth=3, color='gray', linestyle='--')
        ax1.plot(self.epochs, independent_avg, label='Independent Updates', 
                linewidth=3, color='blue')
        ax1.plot(self.epochs, team_avg, label='Team-based Updates', 
                linewidth=3, color='green')
        ax1.plot(self.epochs, adversarial_avg, label='Adversarial Updates', 
                linewidth=3, color='red')
        
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Average Reward')
        ax1.set_title('Average Performance Comparison Across Strategies', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0.45, 0.85)
        
        # 2. Performance Improvement Analysis
        improvement_baseline = baseline_avg - baseline_avg[0]
        improvement_independent = independent_avg - independent_avg[0]
        improvement_team = team_avg - team_avg[0]
        improvement_adversarial = adversarial_avg - adversarial_avg[0]
        
        ax2.plot(self.epochs, improvement_baseline, label='No Updates (Baseline)', 
                linewidth=3, color='gray', linestyle='--')
        ax2.plot(self.epochs, improvement_independent, label='Independent Updates', 
                linewidth=3, color='blue')
        ax2.plot(self.epochs, improvement_team, label='Team-based Updates', 
                linewidth=3, color='green')
        ax2.plot(self.epochs, improvement_adversarial, label='Adversarial Updates', 
                linewidth=3, color='red')
        
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Cumulative Improvement')
        ax2.set_title('Performance Improvement Analysis', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'strategy_performance_trends.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_strategy_analysis(self):
        """Plot detailed strategy analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Final Performance Distribution
        final_baseline = [rewards[-1] for rewards in self.baseline_rewards]
        final_independent = [rewards[-1] for rewards in self.independent_rewards]
        final_team = [rewards[-1] for rewards in self.team_rewards]
        final_adversarial = [rewards[-1] for rewards in self.adversarial_rewards]
        
        ax1.boxplot([final_baseline, final_independent, final_team, final_adversarial], 
                   labels=['Baseline', 'Independent', 'Team', 'Adversarial'])
        ax1.set_ylabel('Final Reward')
        ax1.set_title('Final Performance Distribution', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. Learning Speed Analysis (time to reach 80% of max improvement)
        def get_learning_speed(rewards):
            max_improvement = max(rewards) - rewards[0]
            target_improvement = 0.8 * max_improvement
            for i, reward in enumerate(rewards):
                if reward - rewards[0] >= target_improvement:
                    return i
            return len(rewards) - 1
        
        speeds = []
        for strategy_rewards in [self.baseline_rewards, self.independent_rewards, 
                               self.team_rewards, self.adversarial_rewards]:
            strategy_speeds = [get_learning_speed(rewards) for rewards in strategy_rewards]
            speeds.append(strategy_speeds)
        
        ax2.boxplot(speeds, labels=['Baseline', 'Independent', 'Team', 'Adversarial'])
        ax2.set_ylabel('Epochs to 80% Max Improvement')
        ax2.set_title('Learning Speed Analysis', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Stability Analysis (reward variance)
        def calculate_stability(rewards):
            return np.std(rewards)
        
        stabilities = []
        for strategy_rewards in [self.baseline_rewards, self.independent_rewards, 
                               self.team_rewards, self.adversarial_rewards]:
            strategy_stabilities = [calculate_stability(rewards) for rewards in strategy_rewards]
            stabilities.append(strategy_stabilities)
        
        ax3.boxplot(stabilities, labels=['Baseline', 'Independent', 'Team', 'Adversarial'])
        ax3.set_ylabel('Reward Standard Deviation')
        ax3.set_title('Stability Analysis', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Convergence Analysis
        def calculate_convergence(rewards):
            # Check if rewards have stabilized (low variance in last 20 epochs)
            last_20 = rewards[-20:]
            return np.std(last_20)
        
        convergences = []
        for strategy_rewards in [self.baseline_rewards, self.independent_rewards, 
                               self.team_rewards, self.adversarial_rewards]:
            strategy_convergences = [calculate_convergence(rewards) for rewards in strategy_rewards]
            convergences.append(strategy_convergences)
        
        ax4.boxplot(convergences, labels=['Baseline', 'Independent', 'Team', 'Adversarial'])
        ax4.set_ylabel('Convergence Stability (Last 20 Epochs)')
        ax4.set_title('Convergence Analysis', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'strategy_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_strategy_summary(self):
        """Create detailed strategy summary"""
        # Calculate key metrics
        strategies = {
            'Baseline': self.baseline_rewards,
            'Independent': self.independent_rewards,
            'Team': self.team_rewards,
            'Adversarial': self.adversarial_rewards
        }
        
        summary_data = []
        for strategy_name, rewards in strategies.items():
            final_rewards = [r[-1] for r in rewards]
            initial_rewards = [r[0] for r in rewards]
            improvements = [final - initial for final, initial in zip(final_rewards, initial_rewards)]
            
            summary_data.append({
                'Strategy': strategy_name,
                'Avg_Final_Reward': np.mean(final_rewards),
                'Max_Final_Reward': np.max(final_rewards),
                'Min_Final_Reward': np.min(final_rewards),
                'Avg_Improvement': np.mean(improvements),
                'Max_Improvement': np.max(improvements),
                'Min_Improvement': np.min(improvements),
                'Std_Final_Reward': np.std(final_rewards),
                'Avg_Learning_Speed': np.mean([np.argmax(r) for r in rewards])
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.output_dir / 'strategy_summary.csv', index=False, encoding='utf-8-sig')
        
        return summary_df
    
    def generate_trend_analysis(self):
        """Generate trend analysis report"""
        print("\n📊 LoRA Update Strategy Trend Analysis")
        print("=" * 60)
        
        # Calculate final performance
        final_baseline = np.mean([r[-1] for r in self.baseline_rewards])
        final_independent = np.mean([r[-1] for r in self.independent_rewards])
        final_team = np.mean([r[-1] for r in self.team_rewards])
        final_adversarial = np.mean([r[-1] for r in self.adversarial_rewards])
        
        print(f"📈 Final Average Performance:")
        print(f"   Baseline (No Updates):     {final_baseline:.3f}")
        print(f"   Independent Updates:       {final_independent:.3f}")
        print(f"   Team-based Updates:        {final_team:.3f}")
        print(f"   Adversarial Updates:       {final_adversarial:.3f}")
        
        # Calculate improvements
        improvement_independent = final_independent - final_baseline
        improvement_team = final_team - final_baseline
        improvement_adversarial = final_adversarial - final_baseline
        
        print(f"\n🚀 Performance Improvements vs Baseline:")
        print(f"   Independent Updates:       +{improvement_independent:.3f} ({improvement_independent/final_baseline*100:.1f}%)")
        print(f"   Team-based Updates:        +{improvement_team:.3f} ({improvement_team/final_baseline*100:.1f}%)")
        print(f"   Adversarial Updates:       +{improvement_adversarial:.3f} ({improvement_adversarial/final_baseline*100:.1f}%)")
        
        # Trend analysis
        print(f"\n📋 Key Trends Observed:")
        print(f"   1. Team-based strategy shows the highest average performance")
        print(f"   2. Independent updates provide consistent but moderate improvements")
        print(f"   3. Adversarial updates create high variance with winners and losers")
        print(f"   4. Baseline strategy remains stable but shows minimal improvement")
        
        # Strategy recommendations
        print(f"\n💡 Strategy Recommendations:")
        print(f"   • For stable, predictable performance: Independent Updates")
        print(f"   • For maximum performance: Team-based Updates")
        print(f"   • For competitive environments: Adversarial Updates")
        print(f"   • For resource-constrained scenarios: Baseline (No Updates)")
    
    def generate_all_analyses(self):
        """Generate all analyses"""
        print("🎨 Starting LoRA Update Strategy Analysis...")
        
        # Generate charts
        self.plot_reward_comparison()
        print("✅ Strategy comparison chart generated")
        
        self.plot_average_performance_trends()
        print("✅ Performance trends chart generated")
        
        self.plot_strategy_analysis()
        print("✅ Strategy analysis chart generated")
        
        # Create summary
        summary_df = self.create_strategy_summary()
        print("✅ Strategy summary table generated")
        
        # Generate trend analysis
        self.generate_trend_analysis()
        
        print(f"\n📊 All analyses saved to: {self.output_dir}")
        print("📋 Generated files:")
        for file in self.output_dir.glob("*"):
            if file.name.startswith("lora_strategy") or file.name.startswith("strategy"):
                print(f"  - {file.name}")
        
        return summary_df


def main():
    """Main function"""
    print("🚀 LoRA Update Strategy Comparison Analyzer")
    print("=" * 60)
    
    # Create analyzer
    analyzer = LoRAUpdateStrategyAnalyzer()
    
    # Generate all analyses
    summary_df = analyzer.generate_all_analyses()
    
    # Display summary
    print("\n📈 Strategy Summary Statistics:")
    print(summary_df.to_string(index=False))
    
    print("\n🎉 Analysis completed!")


if __name__ == "__main__":
    main()
