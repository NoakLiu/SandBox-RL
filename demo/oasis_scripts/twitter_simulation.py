import asyncio
import os

from camel.models import ModelFactory
from camel.types import ModelPlatformType

import oasis
from oasis import (ActionType, LLMAction, ManualAction,
                  generate_reddit_agent_graph)

async def main():
    # Define the models for agents. Agents will select models based on
    # pre-defined scheduling strategies
    vllm_model_1 = ModelFactory.create(
        model_platform=ModelPlatformType.VLLM,
        model_type="qwen-2",
        url="http://localhost:8001/v1",
    )
    vllm_model_2 = ModelFactory.create(
        model_platform=ModelPlatformType.VLLM,
        model_type="qwen-2",
        url="http://localhost:8001/v1",
    )
    models = [vllm_model_1, vllm_model_2]

    # Define the available actions for the agents
    available_actions = [
        ActionType.CREATE_POST,
        ActionType.LIKE_POST,
        ActionType.REPOST,
        ActionType.FOLLOW,
        ActionType.DO_NOTHING,
        ActionType.QUOTE_POST,
    ]

    agent_graph = await generate_reddit_agent_graph(
        profile_path="user_data_36.json",
        model=models,
        available_actions=available_actions,
    )

    # 生成 agent graph 后，给每个 agent 分配 group
    import random

    trump_ratio = 0.5  # 50% Trump, 50% Biden
    agent_ids = [id for id, _ in agent_graph.get_agents()]
    trump_agents = set(random.sample(agent_ids, int(len(agent_ids) * trump_ratio)))
    for id, agent in agent_graph.get_agents():
        agent.group = "TRUMP" if id in trump_agents else "BIDEN"

    # Define the path to the database
    db_path = "twitter_simulation.db"

    # Delete the old database
    if os.path.exists(db_path):
        os.remove(db_path)

    # Make the environment
    env = oasis.make(
        agent_graph=agent_graph,
        platform=oasis.DefaultPlatformType.TWITTER,
        database_path=db_path,
    )

    # Run the environment
    await env.reset()

    actions_1 = {}

    actions_1[env.agent_graph.get_agent(0)] = ManualAction(
        action_type=ActionType.CREATE_POST,
        action_args={"content": "Earth is flat."})
    await env.step(actions_1)

    actions_2 = {
        agent: LLMAction()
        # Activate 5 agents with id 1, 3, 5, 7, 9
        for _, agent in env.agent_graph.get_agents([1, 3, 5, 7, 9])
    }

    await env.step(actions_2)

    actions_3 = {}

    actions_3[env.agent_graph.get_agent(1)] = ManualAction(
        action_type=ActionType.CREATE_POST,
        action_args={"content": "Earth is not flat."})
    await env.step(actions_3)

    actions_4 = {
        agent: LLMAction()
        # get all agents
        for _, agent in env.agent_graph.get_agents()
    }
    await env.step(actions_4)

    # 假设 agent_graph 已生成
    import random

    trump_ratio = 0.5
    agent_ids = [id for id, _ in agent_graph.get_agents()]
    trump_agents = set(random.sample(agent_ids, int(len(agent_ids) * trump_ratio)))
    for id, agent in agent_graph.get_agents():
        agent.group = "TRUMP" if id in trump_agents else "BIDEN"

    for step in range(30):
        actions = {}
        for id, agent in agent_graph.get_agents():
            neighbors = agent.get_neighbors()
            neighbor_groups = [n.group for n in neighbors]
            prompt = (
                f"You are a {agent.group} supporter. "
                f"Your neighbors' groups: {neighbor_groups}. "
                "Will you post/forward TRUMP or BIDEN message this round?"
            )
            resp = llm.generate(prompt)
            print(f"[LLM][Agent {id}] Output: {resp}")
            if "TRUMP" in str(resp).upper():
                actions[id] = "TRUMP"
            else:
                actions[id] = "BIDEN"
        # 传播规则
        for id, agent in agent_graph.get_agents():
            action = actions[id]
            neighbors = agent.get_neighbors()
            neighbor_groups = [n.group for n in neighbors]
            trump_ratio = neighbor_groups.count("TRUMP") / len(neighbor_groups) if neighbors else 0
            biden_ratio = 1 - trump_ratio
            if action != agent.group:
                if (action == "TRUMP" and trump_ratio > 0.6) or (action == "BIDEN" and biden_ratio > 0.6):
                    agent.group = action
        trump_count = sum(1 for _, agent in agent_graph.get_agents() if agent.group == "TRUMP")
        biden_count = sum(1 for _, agent in agent_graph.get_agents() if agent.group == "BIDEN")
        print(f"Step {step+1}: TRUMP={trump_count} BIDEN={biden_count}")
        # 可视化等

    # Close the environment
    await env.close()


if __name__ == "__main__":
    asyncio.run(main())