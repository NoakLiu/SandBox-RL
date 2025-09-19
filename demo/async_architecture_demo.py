#!/usr/bin/env python3
"""
Sandbox-RL Async Architecture Demo

This demo showcases the asynchronous architecture components described in Sandbox-RL_Archi.md,
including async LLM calls, async environment management, intelligent scheduling, and distributed architecture.
"""

import asyncio
import time
import logging
import argparse
from typing import Dict, List
import random

# Import async architecture components
from sandbox_rl.core.async_architecture import (
    VLLMClient, RewardBasedSlotManager, OASISSandbox, AsyncAgentWorkflow,
    LLMPolicy, AgentGraph, OASISCorrectSimulation, AgentState, BeliefType
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockConfig:
    """Mock configuration for simulation"""
    def __init__(self, num_steps: int = 10):
        self.num_steps = num_steps


class MockVLLMClient:
    """Mock VLLM client for demo purposes"""
    
    def __init__(self, endpoint: str, model_name: str, max_retries: int = 3):
        self.endpoint = endpoint
        self.model_name = model_name
        self.max_retries = max_retries
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        pass
    
    async def generate(self, prompt: str) -> str:
        """Mock text generation"""
        # Simulate async processing time
        await asyncio.sleep(0.1)
        return f"Mock response for: {prompt[:50]}..."


async def demo_vllm_client():
    """Demo VLLM client functionality"""
    logger.info("=== VLLM Client Demo ===")
    
    client = MockVLLMClient("http://localhost:8001/v1", "mock-model")
    
    async with client as llm:
        prompts = [
            "Generate a response about social networks",
            "Analyze user behavior patterns",
            "Predict information spread"
        ]
        
        tasks = [llm.generate(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks)
        
        for i, result in enumerate(results):
            logger.info(f"Prompt {i+1}: {result}")
    
    logger.info("VLLM Client Demo completed\n")


async def demo_slot_manager():
    """Demo reward-based slot management"""
    logger.info("=== Reward-Based Slot Manager Demo ===")
    
    slot_manager = RewardBasedSlotManager(max_slots=5)
    
    # Allocate slots
    slot_ids = ["agent_1", "agent_2", "agent_3", "agent_4", "agent_5"]
    for slot_id in slot_ids:
        success = slot_manager.allocate_slot(slot_id, initial_reward=random.random())
        logger.info(f"Allocated slot {slot_id}: {success}")
    
    # Update rewards
    slot_manager.update_slot_reward("agent_1", 0.8)
    slot_manager.update_slot_reward("agent_2", 0.6)
    slot_manager.update_slot_reward("agent_3", 0.9)
    
    # Get top slots
    top_slots = slot_manager.get_top_slots(k=3)
    logger.info(f"Top 3 slots by reward: {top_slots}")
    
    # Release a slot
    slot_manager.release_slot("agent_4")
    logger.info("Released agent_4 slot")
    
    logger.info("Slot Manager Demo completed\n")


async def demo_sandbox():
    """Demo OASIS sandbox functionality"""
    logger.info("=== OASIS Sandbox Demo ===")
    
    # Create agents
    agents = [
        AgentState(1, BeliefType.POSITIVE, 0.8, [2, 3], "group_a"),
        AgentState(2, BeliefType.NEGATIVE, 0.6, [1, 4], "group_b"),
        AgentState(3, BeliefType.NEUTRAL, 0.4, [1, 4], "group_a"),
        AgentState(4, BeliefType.POSITIVE, 0.7, [2, 3], "group_b")
    ]
    
    # Create sandboxes by belief type
    positive_sandbox = OASISSandbox(BeliefType.POSITIVE, [agents[0], agents[3]])
    negative_sandbox = OASISSandbox(BeliefType.NEGATIVE, [agents[1]])
    neutral_sandbox = OASISSandbox(BeliefType.NEUTRAL, [agents[2]])
    
    logger.info(f"Positive sandbox: {len(positive_sandbox.get_agents())} agents, influence: {positive_sandbox.total_influence}")
    logger.info(f"Negative sandbox: {len(negative_sandbox.get_agents())} agents, influence: {negative_sandbox.total_influence}")
    logger.info(f"Neutral sandbox: {len(neutral_sandbox.get_agents())} agents, influence: {neutral_sandbox.total_influence}")
    
    # Add new agent
    new_agent = AgentState(5, BeliefType.POSITIVE, 0.5, [1, 2], "group_c")
    positive_sandbox.add_agent(new_agent)
    logger.info(f"Added agent 5 to positive sandbox. New influence: {positive_sandbox.total_influence}")
    
    logger.info("Sandbox Demo completed\n")


async def demo_agent_workflow():
    """Demo async agent workflow"""
    logger.info("=== Async Agent Workflow Demo ===")
    
    # Create agent graph
    agent_graph = AgentGraph()
    
    # Add agents
    agents = [
        AgentState(1, BeliefType.POSITIVE, 0.8, [2, 3], "group_a"),
        AgentState(2, BeliefType.NEGATIVE, 0.6, [1, 4], "group_b"),
        AgentState(3, BeliefType.NEUTRAL, 0.4, [1, 4], "group_a"),
        AgentState(4, BeliefType.POSITIVE, 0.7, [2, 3], "group_b")
    ]
    
    for agent in agents:
        agent_graph.add_agent(agent)
    
    # Create slot manager
    slot_manager = RewardBasedSlotManager(max_slots=10)
    
    # Create LLM policy
    llm_policy = LLMPolicy(model_name="mock-model", url="http://localhost:8001/v1")
    
    # Create workflow
    workflow = AsyncAgentWorkflow(agent_graph, llm_policy, slot_manager)
    
    # Start workflow
    logger.info("Starting async workflow...")
    workers = await workflow.start_async_workflow()
    
    # Let it run for a few seconds
    await asyncio.sleep(2)
    
    # Stop workflow
    logger.info("Stopping async workflow...")
    await workflow.stop_async_workflow()
    
    logger.info("Agent Workflow Demo completed\n")


async def demo_oasis_simulation():
    """Demo OASIS simulation with async architecture"""
    logger.info("=== OASIS Simulation Demo ===")
    
    # Create mock config
    config = MockConfig(num_steps=5)
    
    # Create simulation
    simulation = OASISCorrectSimulation(
        config=config,
        vllm_url="http://localhost:8001/v1",
        model_name="mock-model"
    )
    
    # Add some agents to the simulation
    agents = [
        AgentState(1, BeliefType.POSITIVE, 0.8, [2, 3], "group_a"),
        AgentState(2, BeliefType.NEGATIVE, 0.6, [1, 4], "group_b"),
        AgentState(3, BeliefType.NEUTRAL, 0.4, [1, 4], "group_a"),
        AgentState(4, BeliefType.POSITIVE, 0.7, [2, 3], "group_b")
    ]
    
    for agent in agents:
        simulation.agent_graph.add_agent(agent)
    
    # Run simulation
    logger.info("Running OASIS simulation...")
    start_time = time.time()
    
    await simulation.run_simulation()
    
    end_time = time.time()
    logger.info(f"Simulation completed in {end_time - start_time:.2f} seconds")
    logger.info(f"History length: {len(simulation.history)}")
    
    # Show some results
    for i, step_result in enumerate(simulation.history):
        logger.info(f"Step {i}: {len(step_result['actions'])} actions, {step_result['belief_changes']} belief changes")
    
    logger.info("OASIS Simulation Demo completed\n")


async def demo_parallel_processing():
    """Demo parallel processing capabilities"""
    logger.info("=== Parallel Processing Demo ===")
    
    # Create agent graph
    agent_graph = AgentGraph()
    
    # Add more agents for parallel processing
    for i in range(10):
        agent = AgentState(
            agent_id=i,
            belief_type=random.choice(list(BeliefType)),
            influence_score=random.random(),
            neighbors=list(range(max(0, i-2), min(10, i+3))),
            group=f"group_{i % 3}"
        )
        agent_graph.add_agent(agent)
    
    # Create slot manager
    slot_manager = RewardBasedSlotManager(max_slots=15)
    
    # Create LLM policy
    llm_policy = LLMPolicy(model_name="mock-model", url="http://localhost:8001/v1")
    
    # Create workflow
    workflow = AsyncAgentWorkflow(agent_graph, llm_policy, slot_manager)
    
    # Test parallel inference
    agent_ids = list(range(10))
    logger.info(f"Running parallel inference for {len(agent_ids)} agents...")
    
    start_time = time.time()
    inference_results = await workflow.parallel_inference_batch(agent_ids)
    end_time = time.time()
    
    logger.info(f"Parallel inference completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Results: {len(inference_results)} successful, {len([r for r in inference_results if isinstance(r, Exception)])} failed")
    
    # Test parallel weight updates
    logger.info("Running parallel weight updates...")
    
    start_time = time.time()
    update_results = await workflow.parallel_weight_update_batch(agent_ids[:5])
    end_time = time.time()
    
    logger.info(f"Parallel weight updates completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Results: {len(update_results)} successful, {len([r for r in update_results if isinstance(r, Exception)])} failed")
    
    logger.info("Parallel Processing Demo completed\n")


async def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="Sandbox-RL Async Architecture Demo")
    parser.add_argument("--demo", choices=["all", "vllm", "slot", "sandbox", "workflow", "simulation", "parallel"], 
                       default="all", help="Which demo to run")
    parser.add_argument("--steps", type=int, default=5, help="Number of simulation steps")
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting Sandbox-RL Async Architecture Demo")
    logger.info("=" * 50)
    
    try:
        if args.demo == "all" or args.demo == "vllm":
            await demo_vllm_client()
        
        if args.demo == "all" or args.demo == "slot":
            await demo_slot_manager()
        
        if args.demo == "all" or args.demo == "sandbox":
            await demo_sandbox()
        
        if args.demo == "all" or args.demo == "workflow":
            await demo_agent_workflow()
        
        if args.demo == "all" or args.demo == "simulation":
            await demo_oasis_simulation()
        
        if args.demo == "all" or args.demo == "parallel":
            await demo_parallel_processing()
        
        logger.info("✅ All demos completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
