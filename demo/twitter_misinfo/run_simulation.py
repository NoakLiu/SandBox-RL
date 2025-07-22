import json
import matplotlib.pyplot as plt
from workflow import TwitterMisinfoWorkflow

def load_agent_graph(path):
    """
    加载或生成Twitter agent graph，格式:
    {agent_id: {"neighbors": [id1, id2, ...]}}
    """
    with open(path, 'r') as f:
        return json.load(f)

def main():
    agent_graph = load_agent_graph("user_data.json")
    # 可选: llm_mode='adaptive' 体验RL权重更新
    workflow = TwitterMisinfoWorkflow(agent_graph, llm_mode='adaptive')
    history, rewards, slot_rewards = workflow.run(max_steps=30)
    trump_history = [h["trump"] for h in history]
    biden_history = [h["biden"] for h in history]
    plt.plot(trump_history, label="Trump wins")
    plt.plot(biden_history, label="Biden wins")
    plt.xlabel("Step")
    plt.ylabel("Agent count")
    plt.legend()
    plt.title("Trump vs Biden Misinformation Spread")
    plt.show()
    plt.plot(rewards, label="Reward")
    plt.plot(slot_rewards, label="Slot Reward")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.legend()
    plt.title("Reward & Slot Reward Curve")
    plt.show()

if __name__ == "__main__":
    main() 