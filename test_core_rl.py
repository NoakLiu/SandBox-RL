#!/usr/bin/env python3
"""
Test Core RL Functionality

简单测试core模块中的RL功能
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from sandbox_rl.core.rl_algorithms import (
        CooperationType,
        CompetenceType,
        CooperationFactor,
        CompetenceFactor,
        RLAlgorithm,
        RLConfig,
        TrajectoryStep,
        OnPolicyRLAgent,
        MultiAgentOnPolicyRL
    )
    print("✅ Successfully imported core RL modules")
    
    # Test cooperation factors
    print("\n🔗 Testing Cooperation Factors:")
    coop_config = CooperationFactor(
        cooperation_type=CooperationType.TEAM_BASED,
        cooperation_strength=0.3,
        team_size=4,
        shared_reward_ratio=0.6
    )
    print(f"  - Cooperation Type: {coop_config.cooperation_type.value}")
    print(f"  - Cooperation Strength: {coop_config.cooperation_strength}")
    print(f"  - Team Size: {coop_config.team_size}")
    
    # Test competence factors
    print("\n🎯 Testing Competence Factors:")
    comp_config = CompetenceFactor(
        competence_type=CompetenceType.ADAPTIVE,
        base_capability=0.5,
        learning_rate=0.02,
        adaptation_speed=0.15
    )
    print(f"  - Competence Type: {comp_config.competence_type.value}")
    print(f"  - Base Capability: {comp_config.base_capability}")
    print(f"  - Learning Rate: {comp_config.learning_rate}")
    
    # Test RL config
    print("\n⚙️ Testing RL Config:")
    rl_config = RLConfig(
        algorithm=RLAlgorithm.ON_POLICY_PPO,
        cooperation_factor=coop_config,
        competence_factor=comp_config
    )
    print(f"  - Algorithm: {rl_config.algorithm.value}")
    print(f"  - Learning Rate: {rl_config.learning_rate}")
    
    # Test trajectory step
    print("\n📊 Testing Trajectory Step:")
    step = TrajectoryStep(
        state={"position": [1, 2, 3]},
        action="action_1",
        reward=0.5,
        value=0.6,
        log_prob=-0.7,
        done=False
    )
    print(f"  - Action: {step.action}")
    print(f"  - Reward: {step.reward}")
    print(f"  - Value: {step.value}")
    
    # Test multi-agent system
    print("\n🤖 Testing Multi-Agent System:")
    multi_agent = MultiAgentOnPolicyRL(
        num_agents=4,
        state_dim=32,
        action_dim=5
    )
    print(f"  - Number of Agents: {len(multi_agent.agents)}")
    print(f"  - Number of Teams: {len(multi_agent.teams)}")
    
    # Test agent step
    print("\n🚀 Testing Agent Step:")
    state = {"position": [0, 0, 0], "energy": 1.0}
    action, log_prob, value = multi_agent.step("agent_0", state)
    print(f"  - Action: {action}")
    print(f"  - Log Prob: {log_prob:.3f}")
    print(f"  - Value: {value:.3f}")
    
    # Test agent stats
    print("\n📈 Testing Agent Stats:")
    stats = multi_agent.get_agent_stats()
    for agent_id, agent_stats in stats.items():
        print(f"  - {agent_id}: capability={agent_stats['capability']:.3f}")
    
    print("\n✅ All tests passed successfully!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the Sandbox-RL root directory")
except Exception as e:
    print(f"❌ Test error: {e}")
    import traceback
    traceback.print_exc()
