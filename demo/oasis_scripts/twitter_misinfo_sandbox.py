import random
import matplotlib.pyplot as plt
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from oasis import generate_twitter_agent_graph

async def main():
    # 1. Agent graph
    agent_graph = await generate_twitter_agent_graph(
        profile_path="user_data.json",
        model=[],
        available_actions=[],
    )
    # 2. Group assignment
    trump_ratio = 0.5
    agent_ids = [id for id, _ in agent_graph.get_agents()]
    trump_agents = set(random.sample(agent_ids, int(len(agent_ids) * trump_ratio)))
    for id, agent in agent_graph.get_agents():
        agent.group = "TRUMP" if id in trump_agents else "BIDEN"

    # 3. LLM
    llm = ModelFactory.create(
        model_platform=ModelPlatformType.VLLM,
        model_type="qwen-2",
        url="http://localhost:8001/v1",
    )

    # 4. Simulation
    num_steps = 30
    history = []

    for step in range(num_steps):
        actions = {}
        for id, agent in agent_graph.get_agents():
            neighbors = agent.get_neighbors()
            neighbor_groups = [n.group for n in neighbors]
            prompt = (
                f"You are a {agent.group} supporter. "
                f"Your goal is to convince others that '{agent.group} won the election'. "
                f"Your neighbors' groups: {neighbor_groups}. "
                "What will you post or forward to maximize your group's influence?"
            )
            # LLM 决策
            resp = llm.generate(prompt)
            print(f"[LLM][Agent {id}][{agent.group}] Output: {resp}")
            # 简单决策规则：如果 LLM 输出包含对方 group，则视为被说服
            if agent.group == "TRUMP" and "BIDEN" in resp.upper():
                actions[id] = "SWITCH"
            elif agent.group == "BIDEN" and "TRUMP" in resp.upper():
                actions[id] = "SWITCH"
            else:
                actions[id] = "STAY"

        # 传播与信仰转变
        for id, agent in agent_graph.get_agents():
            if actions[id] == "SWITCH":
                agent.group = "BIDEN" if agent.group == "TRUMP" else "TRUMP"

        # 统计每轮各组人数
        trump_count = sum(1 for _, agent in agent_graph.get_agents() if agent.group == "TRUMP")
        biden_count = sum(1 for _, agent in agent_graph.get_agents() if agent.group == "BIDEN")
        history.append((trump_count, biden_count))
        print(f"Step {step}: TRUMP={trump_count}, BIDEN={biden_count}")

    # 5. Visualization
    trump_history = [h[0] for h in history]
    biden_history = [h[1] for h in history]
    plt.plot(trump_history, label="TRUMP")
    plt.plot(biden_history, label="BIDEN")
    plt.xlabel("Step")
    plt.ylabel("Agent count")
    plt.legend()
    plt.title("Trump vs Biden Misinformation Spread")
    plt.show()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 