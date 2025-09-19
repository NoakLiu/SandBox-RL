#!/usr/bin/env python3
"""
Test Concordia Sandbox

简单测试Concordia Contest沙盒功能
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from sandbox_rl.core.concordia_sandbox import (
        ConcordiaScenario,
        ConcordiaRole,
        ConcordiaConfig,
        ConcordiaSandbox,
        create_concordia_sandbox,
        create_trading_scenario
    )
    from sandbox_rl.core.rl_algorithms import (
        CooperationType, CompetenceType,
        CooperationFactor, CompetenceFactor
    )
    print("✅ Successfully imported Concordia sandbox modules")
    
    # Test Concordia scenarios
    print("\n🎯 Testing Concordia Scenarios:")
    for scenario in ConcordiaScenario:
        print(f"  - {scenario.value}")
    
    # Test Concordia roles
    print("\n👥 Testing Concordia Roles:")
    for role in ConcordiaRole:
        print(f"  - {role.value}")
    
    # Test Concordia config
    print("\n⚙️ Testing Concordia Config:")
    config = ConcordiaConfig(
        scenario=ConcordiaScenario.TRADING,
        role=ConcordiaRole.TRADER_A,
        max_turns=30,
        cooperation_factor=CooperationFactor(
            cooperation_type=CooperationType.SHARED_REWARDS,
            cooperation_strength=0.5,
            shared_reward_ratio=0.7
        )
    )
    print(f"  - Scenario: {config.scenario.value}")
    print(f"  - Role: {config.role.value}")
    print(f"  - Max Turns: {config.max_turns}")
    print(f"  - Cooperation Type: {config.cooperation_factor.cooperation_type.value}")
    
    # Test sandbox creation
    print("\n🤖 Testing Sandbox Creation:")
    sandbox = create_trading_scenario("trader_a")
    print(f"  - Scenario: {sandbox.scenario}")
    print(f"  - Role: {sandbox.role}")
    print(f"  - Environment: {type(sandbox.environment).__name__}")
    
    # Test case generation
    print("\n📝 Testing Case Generation:")
    case = sandbox.case_generator()
    print(f"  - Observation: {case['obs'][:50]}...")
    print(f"  - Turn: {case['turn']}")
    print(f"  - Scenario: {case['scenario']}")
    print(f"  - Role: {case['role']}")
    
    # Test prompt generation
    print("\n💬 Testing Prompt Generation:")
    prompt = sandbox.prompt_func(case)
    print(f"  - Prompt Length: {len(prompt)} characters")
    print(f"  - Prompt Preview: {prompt[:100]}...")
    
    # Test action verification
    print("\n✅ Testing Action Verification:")
    action = "I offer to trade my apple for your orange."
    reward = sandbox.verify_score(action, case)
    print(f"  - Action: {action}")
    print(f"  - Reward: {reward.reward:.3f}")
    print(f"  - Done: {reward.done}")
    print(f"  - Aux Data: {list(reward.aux.keys())}")
    
    # Test state management
    print("\n📊 Testing State Management:")
    state = sandbox.get_state()
    print(f"  - Turn: {state['turn']}")
    print(f"  - Memory Entries: {len(state['memory'])}")
    print(f"  - Metrics: {list(state['metrics'].keys())}")
    
    print("\n✅ All tests passed successfully!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the Sandbox-RL root directory")
except Exception as e:
    print(f"❌ Test error: {e}")
    import traceback
    traceback.print_exc()
