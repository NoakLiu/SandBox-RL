#!/usr/bin/env python3
"""
Test script for SandGraph Async Architecture

This script tests the async architecture components to ensure they work correctly.
"""

import asyncio
import time
import logging
from typing import List

# Import async architecture components
from sandgraph.core.async_architecture import (
    RewardBasedSlotManager, OASISSandbox, AgentGraph, 
    AgentState, BeliefType
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_slot_manager():
    """Test reward-based slot manager"""
    logger.info("Testing RewardBasedSlotManager...")
    
    slot_manager = RewardBasedSlotManager(max_slots=3)
    
    # Test slot allocation
    assert slot_manager.allocate_slot("agent_1", 0.5)
    assert slot_manager.allocate_slot("agent_2", 0.3)
    assert slot_manager.allocate_slot("agent_3", 0.7)
    assert not slot_manager.allocate_slot("agent_4", 0.1)  # Should fail, max slots reached
    
    # Test reward updates
    assert slot_manager.update_slot_reward("agent_1", 0.8)
    assert slot_manager.update_slot_reward("agent_2", 0.6)
    
    # Test top slots
    top_slots = slot_manager.get_top_slots(k=2)
    assert len(top_slots) == 2
    assert "agent_1" in top_slots or "agent_3" in top_slots
    
    # Test slot release
    assert slot_manager.release_slot("agent_2")
    assert slot_manager.allocate_slot("agent_4", 0.2)  # Should succeed now
    
    logger.info("✅ RewardBasedSlotManager tests passed")


async def test_sandbox():
    """Test OASIS sandbox functionality"""
    logger.info("Testing OASIS Sandbox...")
    
    # Create agents
    agents = [
        AgentState(1, BeliefType.POSITIVE, 0.8, [2], "group_a"),
        AgentState(2, BeliefType.NEGATIVE, 0.6, [1], "group_b"),
        AgentState(3, BeliefType.NEUTRAL, 0.4, [], "group_a")
    ]
    
    # Create sandbox
    sandbox = OASISSandbox(BeliefType.POSITIVE, [agents[0]])
    
    # Test initial state
    assert len(sandbox.get_agents()) == 1
    assert sandbox.total_influence == 0.8
    
    # Test adding agent
    sandbox.add_agent(agents[2])  # Add neutral agent (should still work)
    assert len(sandbox.get_agents()) == 2
    assert sandbox.total_influence == 1.2
    
    # Test removing agent
    sandbox.remove_agent(1)
    assert len(sandbox.get_agents()) == 1
    assert sandbox.total_influence == 0.4
    
    logger.info("✅ OASIS Sandbox tests passed")


async def test_agent_graph():
    """Test agent graph functionality"""
    logger.info("Testing Agent Graph...")
    
    agent_graph = AgentGraph()
    
    # Create agents
    agents = [
        AgentState(1, BeliefType.POSITIVE, 0.8, [2], "group_a"),
        AgentState(2, BeliefType.NEGATIVE, 0.6, [1], "group_b"),
        AgentState(3, BeliefType.NEUTRAL, 0.4, [], "group_a")
    ]
    
    # Test adding agents
    for agent in agents:
        agent_graph.add_agent(agent)
    
    assert len(agent_graph.get_agents()) == 3
    
    # Test getting agent
    agent = agent_graph.get_agent(1)
    assert agent is not None
    assert agent.agent_id == 1
    assert agent.belief_type == BeliefType.POSITIVE
    
    # Test getting non-existent agent
    agent = agent_graph.get_agent(999)
    assert agent is None
    
    # Test removing agent
    agent_graph.remove_agent(2)
    assert len(agent_graph.get_agents()) == 2
    assert agent_graph.get_agent(2) is None
    
    logger.info("✅ Agent Graph tests passed")


async def test_agent_state():
    """Test agent state functionality"""
    logger.info("Testing Agent State...")
    
    # Test agent creation
    agent = AgentState(
        agent_id=1,
        belief_type=BeliefType.POSITIVE,
        influence_score=0.8,
        neighbors=[2, 3],
        group="test_group",
        priority=0.9
    )
    
    assert agent.agent_id == 1
    assert agent.belief_type == BeliefType.POSITIVE
    assert agent.influence_score == 0.8
    assert agent.neighbors == [2, 3]
    assert agent.group == "test_group"
    assert agent.priority == 0.9
    assert agent.reward_history == []  # Should be empty list by default
    
    # Test reward history
    agent.reward_history.extend([0.1, 0.2, 0.3])
    assert len(agent.reward_history) == 3
    assert agent.reward_history == [0.1, 0.2, 0.3]
    
    logger.info("✅ Agent State tests passed")


async def test_belief_types():
    """Test belief type enum"""
    logger.info("Testing Belief Types...")
    
    assert BeliefType.POSITIVE.value == "positive"
    assert BeliefType.NEGATIVE.value == "negative"
    assert BeliefType.NEUTRAL.value == "neutral"
    
    # Test enum membership
    assert BeliefType.POSITIVE in BeliefType
    assert BeliefType.NEGATIVE in BeliefType
    assert BeliefType.NEUTRAL in BeliefType
    
    logger.info("✅ Belief Types tests passed")


async def test_concurrent_operations():
    """Test concurrent operations for thread safety"""
    logger.info("Testing Concurrent Operations...")
    
    slot_manager = RewardBasedSlotManager(max_slots=10)
    agent_graph = AgentGraph()
    
    # Create multiple agents
    agents = []
    for i in range(5):
        agent = AgentState(
            agent_id=i,
            belief_type=BeliefType.POSITIVE,
            influence_score=0.5 + i * 0.1,
            neighbors=list(range(max(0, i-1), min(5, i+2))),
            group=f"group_{i % 2}"
        )
        agents.append(agent)
    
    # Test concurrent slot allocation
    async def allocate_slots():
        for i, agent in enumerate(agents):
            slot_manager.allocate_slot(f"agent_{i}", agent.influence_score)
            await asyncio.sleep(0.01)  # Small delay to simulate async work
    
    # Test concurrent agent addition
    async def add_agents():
        for agent in agents:
            agent_graph.add_agent(agent)
            await asyncio.sleep(0.01)  # Small delay to simulate async work
    
    # Run concurrent operations
    await asyncio.gather(allocate_slots(), add_agents())
    
    # Verify results
    assert len(slot_manager.active_slots) == 5
    assert len(agent_graph.get_agents()) == 5
    
    logger.info("✅ Concurrent Operations tests passed")


async def main():
    """Run all tests"""
    logger.info("🧪 Starting Async Architecture Tests")
    logger.info("=" * 50)
    
    start_time = time.time()
    
    try:
        await test_belief_types()
        await test_agent_state()
        await test_slot_manager()
        await test_sandbox()
        await test_agent_graph()
        await test_concurrent_operations()
        
        end_time = time.time()
        logger.info("=" * 50)
        logger.info(f"✅ All tests passed in {end_time - start_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
