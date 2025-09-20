#!/usr/bin/env python3
"""
Custom Environments Example
===========================

Example showing how to create and use custom training environments
for specialized multi-model RL scenarios.
"""

import asyncio
import logging
from core_srl import (
    MultiModelTrainer,
    MultiModelConfig,
    TrainingMode,
    create_multi_model_coop_compete_env,
    create_multi_model_team_battle,
    create_multi_model_staged_env,
    create_maze_training_env,
    create_social_training_env
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def maze_environment_training():
    """Training in maze navigation environment"""
    
    print("🗺️ Maze Environment Training")
    print("=" * 30)
    
    # Create maze environment
    maze_env = create_maze_training_env(complexity="medium")
    
    print("🗺️ Maze environment created:")
    print("   Complexity: Medium")
    print("   Goal: Navigate to target while avoiding obstacles")
    print("   Cooperation: Share path information")
    print("   Competition: Race to reach goal first")
    
    # Configure training for maze
    config = MultiModelConfig(
        num_models=4,
        training_mode=TrainingMode.MIXED,
        cooperation_strength=0.5,  # Balanced for maze navigation
        competition_intensity=0.5,
        max_episodes=150
    )
    
    trainer = MultiModelTrainer(config)
    trainer.environment = maze_env
    
    try:
        print("\n🏃‍♂️ Starting maze navigation training...")
        
        # Custom monitoring for maze environment
        navigation_stats = []
        
        for episode in range(150):
            episode_result = await trainer.train_multi_model_episode(episode)
            
            # Analyze maze-specific metrics
            env_info = episode_result.get('environment_info', {})
            
            # Calculate navigation efficiency
            total_steps = sum(env_info.get('model_steps', {}).values())
            successful_navigations = sum(1 for r in episode_result['rewards'].values() if r > 1.0)
            
            navigation_stats.append({
                'episode': episode,
                'success_rate': successful_navigations / config.num_models,
                'avg_steps': total_steps / config.num_models,
                'cooperation_ratio': episode_result['cooperation_ratio']
            })
            
            # Log progress
            if episode % 30 == 0:
                recent_stats = navigation_stats[-10:] if len(navigation_stats) >= 10 else navigation_stats
                avg_success = sum(s['success_rate'] for s in recent_stats) / len(recent_stats)
                avg_steps = sum(s['avg_steps'] for s in recent_stats) / len(recent_stats)
                
                print(f"   Episode {episode}: Success rate={avg_success:.3f}, Avg steps={avg_steps:.1f}")
        
        print("\n✅ Maze training completed!")
        
        # Analyze navigation learning
        final_stats = navigation_stats[-20:]  # Last 20 episodes
        final_success_rate = sum(s['success_rate'] for s in final_stats) / len(final_stats)
        final_avg_steps = sum(s['avg_steps'] for s in final_stats) / len(final_stats)
        
        print(f"\n🗺️ Maze Navigation Results:")
        print(f"   Final success rate: {final_success_rate:.3f}")
        print(f"   Final average steps: {final_avg_steps:.1f}")
        
        if final_success_rate > 0.8:
            print("   ✅ Excellent navigation learning")
        elif final_success_rate > 0.6:
            print("   ✅ Good navigation learning")
        else:
            print("   ⚠️ Navigation learning needs improvement")
        
        return navigation_stats
        
    finally:
        await trainer.shutdown()


async def social_interaction_training():
    """Training in social interaction environment"""
    
    print("\n👥 Social Interaction Training")
    print("=" * 33)
    
    # Create social environment
    social_env = create_social_training_env(scenario="negotiation", num_models=6)
    
    print("👥 Social environment created:")
    print("   Scenario: Negotiation")
    print("   Participants: 6 models")
    print("   Goal: Reach mutually beneficial agreements")
    print("   Dynamics: Trust building and strategic communication")
    
    # Configure for social interaction
    config = MultiModelConfig(
        num_models=6,
        training_mode=TrainingMode.COOPERATIVE,  # Social scenarios benefit from cooperation
        cooperation_strength=0.7,
        max_episodes=120
    )
    
    trainer = MultiModelTrainer(config)
    trainer.environment = social_env
    
    try:
        print("\n💬 Starting social interaction training...")
        
        # Track social dynamics
        social_stats = []
        
        for episode in range(120):
            episode_result = await trainer.train_multi_model_episode(episode)
            
            # Analyze social metrics
            env_info = episode_result.get('environment_info', {})
            
            # Calculate social success metrics
            agreement_rate = env_info.get('agreement_rate', 0.5)
            trust_scores = env_info.get('trust_scores', {})
            avg_trust = sum(trust_scores.values()) / len(trust_scores) if trust_scores else 0.5
            
            social_stats.append({
                'episode': episode,
                'agreement_rate': agreement_rate,
                'avg_trust': avg_trust,
                'cooperation_ratio': episode_result['cooperation_ratio'],
                'total_reward': episode_result['total_reward']
            })
            
            # Log social progress
            if episode % 24 == 0:
                recent_stats = social_stats[-12:] if len(social_stats) >= 12 else social_stats
                avg_agreement = sum(s['agreement_rate'] for s in recent_stats) / len(recent_stats)
                avg_trust = sum(s['avg_trust'] for s in recent_stats) / len(recent_stats)
                
                print(f"   Episode {episode}: Agreements={avg_agreement:.3f}, Trust={avg_trust:.3f}")
        
        print("\n✅ Social interaction training completed!")
        
        # Analyze social learning
        final_stats = social_stats[-15:]  # Last 15 episodes
        final_agreement_rate = sum(s['agreement_rate'] for s in final_stats) / len(final_stats)
        final_trust = sum(s['avg_trust'] for s in final_stats) / len(final_stats)
        
        print(f"\n👥 Social Interaction Results:")
        print(f"   Final agreement rate: {final_agreement_rate:.3f}")
        print(f"   Final trust level: {final_trust:.3f}")
        
        # Social learning assessment
        if final_agreement_rate > 0.8 and final_trust > 0.7:
            print("   ✅ Excellent social learning - high cooperation and trust")
        elif final_agreement_rate > 0.6 and final_trust > 0.5:
            print("   ✅ Good social learning - effective communication")
        else:
            print("   ⚠️ Social learning challenges - consider adjusting cooperation parameters")
        
        return social_stats
        
    finally:
        await trainer.shutdown()


async def team_battle_training():
    """Training in team battle environment"""
    
    print("\n⚔️ Team Battle Training")
    print("=" * 25)
    
    # Create team battle environment (4v4)
    team_env = create_multi_model_team_battle()
    
    print("⚔️ Team battle environment created:")
    print("   Format: 4v4 team battle")
    print("   Intra-team: Cooperation required")
    print("   Inter-team: Competition for victory")
    print("   Strategy: Coordinate with teammates, outperform opponents")
    
    # Configure for team dynamics
    config = MultiModelConfig(
        num_models=8,  # 4 per team
        training_mode=TrainingMode.MIXED,  # Cooperation within teams, competition between
        cooperation_strength=0.8,  # High intra-team cooperation
        competition_intensity=0.6,  # Moderate inter-team competition
        max_episodes=200
    )
    
    trainer = MultiModelTrainer(config)
    trainer.environment = team_env
    
    try:
        print("\n⚔️ Starting team battle training...")
        
        # Track team performance
        team_stats = []
        
        for episode in range(200):
            episode_result = await trainer.train_multi_model_episode(episode)
            
            # Analyze team dynamics
            model_rewards = episode_result['rewards']
            
            # Split into teams (assuming models 0-3 are team A, 4-7 are team B)
            team_a_rewards = [model_rewards[f'model_{i}_qwen3'] for i in range(4) 
                             if f'model_{i}_qwen3' in model_rewards]
            team_b_rewards = [model_rewards[f'model_{i}_qwen3'] for i in range(4, 8) 
                             if f'model_{i}_qwen3' in model_rewards]
            
            team_a_total = sum(team_a_rewards) if team_a_rewards else 0
            team_b_total = sum(team_b_rewards) if team_b_rewards else 0
            
            # Team coordination (low variance = good coordination)
            team_a_var = sum((r - team_a_total/len(team_a_rewards))**2 for r in team_a_rewards) / len(team_a_rewards) if team_a_rewards else 0
            team_b_var = sum((r - team_b_total/len(team_b_rewards))**2 for r in team_b_rewards) / len(team_b_rewards) if team_b_rewards else 0
            
            team_stats.append({
                'episode': episode,
                'team_a_total': team_a_total,
                'team_b_total': team_b_total,
                'team_a_coordination': 1.0 / (1.0 + team_a_var),
                'team_b_coordination': 1.0 / (1.0 + team_b_var),
                'winner': 'A' if team_a_total > team_b_total else 'B'
            })
            
            # Log team progress
            if episode % 40 == 0:
                recent_stats = team_stats[-20:] if len(team_stats) >= 20 else team_stats
                team_a_wins = sum(1 for s in recent_stats if s['winner'] == 'A')
                team_b_wins = len(recent_stats) - team_a_wins
                
                avg_a_coord = sum(s['team_a_coordination'] for s in recent_stats) / len(recent_stats)
                avg_b_coord = sum(s['team_b_coordination'] for s in recent_stats) / len(recent_stats)
                
                print(f"   Episode {episode}: A wins={team_a_wins}, B wins={team_b_wins}")
                print(f"      Team A coordination: {avg_a_coord:.3f}")
                print(f"      Team B coordination: {avg_b_coord:.3f}")
        
        print("\n✅ Team battle training completed!")
        
        # Final team analysis
        final_stats = team_stats[-30:]  # Last 30 episodes
        
        team_a_wins = sum(1 for s in final_stats if s['winner'] == 'A')
        team_b_wins = len(final_stats) - team_a_wins
        
        final_a_coord = sum(s['team_a_coordination'] for s in final_stats) / len(final_stats)
        final_b_coord = sum(s['team_b_coordination'] for s in final_stats) / len(final_stats)
        
        print(f"\n⚔️ Team Battle Results:")
        print(f"   Team A wins: {team_a_wins}/{len(final_stats)} ({team_a_wins/len(final_stats):.1%})")
        print(f"   Team B wins: {team_b_wins}/{len(final_stats)} ({team_b_wins/len(final_stats):.1%})")
        print(f"   Team A coordination: {final_a_coord:.3f}")
        print(f"   Team B coordination: {final_b_coord:.3f}")
        
        # Balance assessment
        win_balance = abs(team_a_wins - team_b_wins) / len(final_stats)
        
        if win_balance < 0.2:
            print("   ⚖️ Well-balanced teams - competitive training successful")
        elif win_balance < 0.4:
            print("   ✅ Reasonably balanced - good team dynamics")
        else:
            winning_team = 'A' if team_a_wins > team_b_wins else 'B'
            print(f"   ⚠️ Team {winning_team} dominates - consider rebalancing")
        
        return team_stats
        
    finally:
        await trainer.shutdown()


async def staged_environment_training():
    """Training in staged environment with evolving difficulty"""
    
    print("\n📈 Staged Environment Training")
    print("=" * 33)
    
    # Create staged environment
    staged_env = create_multi_model_staged_env(num_models=6)
    
    print("📈 Staged environment created:")
    print("   Models: 6")
    print("   Stages: Warmup → Intermediate → Advanced")
    print("   Evolution: Gradually increasing complexity and competition")
    
    # Configure for staged learning
    config = MultiModelConfig(
        num_models=6,
        training_mode=TrainingMode.MIXED,
        max_episodes=180  # 60 episodes per stage
    )
    
    trainer = MultiModelTrainer(config)
    trainer.environment = staged_env
    
    try:
        print("\n📊 Starting staged environment training...")
        
        # Track progression through stages
        stage_stats = []
        
        for episode in range(180):
            episode_result = await trainer.train_multi_model_episode(episode)
            
            # Determine current stage
            current_stage = "Warmup" if episode < 60 else "Intermediate" if episode < 120 else "Advanced"
            
            # Analyze stage-specific metrics
            env_info = episode_result.get('environment_info', {})
            difficulty_level = env_info.get('difficulty_level', 0.5)
            
            stage_stats.append({
                'episode': episode,
                'stage': current_stage,
                'difficulty': difficulty_level,
                'total_reward': episode_result['total_reward'],
                'cooperation_ratio': episode_result['cooperation_ratio']
            })
            
            # Log stage progress
            if episode % 30 == 0:
                stage_episodes = [s for s in stage_stats if s['stage'] == current_stage]
                if stage_episodes:
                    avg_reward = sum(s['total_reward'] for s in stage_episodes) / len(stage_episodes)
                    avg_coop = sum(s['cooperation_ratio'] for s in stage_episodes) / len(stage_episodes)
                    
                    print(f"   Episode {episode} ({current_stage}): "
                          f"Reward={avg_reward:.2f}, Coop={avg_coop:.3f}, Difficulty={difficulty_level:.3f}")
        
        print("\n✅ Staged environment training completed!")
        
        # Analyze progression across stages
        warmup_stats = [s for s in stage_stats if s['stage'] == 'Warmup']
        intermediate_stats = [s for s in stage_stats if s['stage'] == 'Intermediate']
        advanced_stats = [s for s in stage_stats if s['stage'] == 'Advanced']
        
        print(f"\n📈 Stage Progression Analysis:")
        
        for stage_name, stage_data in [('Warmup', warmup_stats), ('Intermediate', intermediate_stats), ('Advanced', advanced_stats)]:
            if stage_data:
                avg_reward = sum(s['total_reward'] for s in stage_data) / len(stage_data)
                avg_difficulty = sum(s['difficulty'] for s in stage_data) / len(stage_data)
                
                print(f"   {stage_name}: Avg reward={avg_reward:.2f}, Avg difficulty={avg_difficulty:.3f}")
        
        # Learning progression assessment
        if warmup_stats and advanced_stats:
            warmup_avg = sum(s['total_reward'] for s in warmup_stats) / len(warmup_stats)
            advanced_avg = sum(s['total_reward'] for s in advanced_stats) / len(advanced_stats)
            
            # Account for increasing difficulty
            progression_score = advanced_avg / max(warmup_avg, 0.1)
            
            print(f"\n🎯 Learning Progression:")
            print(f"   Progression score: {progression_score:.3f}")
            
            if progression_score > 1.2:
                print("   ✅ Excellent adaptation - models improved despite increasing difficulty")
            elif progression_score > 0.9:
                print("   ✅ Good adaptation - models maintained performance")
            else:
                print("   ⚠️ Adaptation challenges - consider adjusting difficulty progression")
        
        return stage_stats
        
    finally:
        await trainer.shutdown()


if __name__ == "__main__":
    print("=" * 60)
    print("🎮 Core SRL - Custom Environment Examples")
    print("=" * 60)
    
    # Run maze environment training
    maze_stats = asyncio.run(maze_environment_training())
    
    # Run social interaction training
    social_stats = asyncio.run(social_interaction_training())
    
    # Run team battle training
    team_stats = asyncio.run(team_battle_training())
    
    # Run staged environment training
    stage_stats = asyncio.run(staged_environment_training())
    
    print("\n" + "=" * 60)
    print("✨ Custom environment examples completed!")
    print("💡 Key insights:")
    print("   - Different environments require different cooperation/competition balances")
    print("   - Maze environments benefit from moderate mixed strategies")
    print("   - Social scenarios thrive with high cooperation")
    print("   - Team battles need strong intra-team cooperation")
    print("   - Staged environments test adaptation capabilities")
    print("=" * 60)
