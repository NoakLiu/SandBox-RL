import random
from sandgraph.core.sandbox import Sandbox

class TwitterMisinformationSandbox(Sandbox):
    """
    环境子集/沙盒：抽象Twitter子图，维护节点信仰、邻居、传播规则，支持step和奖励接口。
    """
    def __init__(self, agent_graph, trump_ratio=0.5, seed=42):
        super().__init__("twitter_misinfo", "Twitter Misinformation Spread Sandbox")
        self.agent_graph = agent_graph  # {agent_id: {"neighbors": [...], ...}}
        self.num_agents = len(agent_graph)
        self.random = random.Random(seed)
        self.trump_ratio = trump_ratio
        self.beliefs = self._init_beliefs()
        self.step_count = 0
        self.history = []

    def _init_beliefs(self):
        trump_count = int(self.num_agents * self.trump_ratio)
        trump_agents = set(self.random.sample(list(self.agent_graph.keys()), trump_count))
        beliefs = {}
        for i in self.agent_graph:
            beliefs[i] = "TRUMP" if i in trump_agents else "BIDEN"
        return beliefs

    def get_prompts(self):
        """
        返回每个agent的prompt，包含邻居观点和自身信仰，供LLM决策。
        """
        prompts = {}
        for agent_id, info in self.agent_graph.items():
            neighbor_beliefs = [self.beliefs[n] for n in info["neighbors"]]
            prompts[agent_id] = (
                f"Your neighbors believe: {neighbor_beliefs}. "
                f"Your current belief: {self.beliefs[agent_id]}. "
                f"Should you post/forward TRUMP or BIDEN?"
            )
        return prompts

    def step(self, actions):
        """
        输入所有agent的动作，更新信仰，返回新状态、奖励、done。
        """
        new_beliefs = self.beliefs.copy()
        for agent_id, action in actions.items():
            neighbors = self.agent_graph[agent_id]["neighbors"]
            if not neighbors:
                continue
            neighbor_beliefs = [self.beliefs[n] for n in neighbors]
            trump_ratio = neighbor_beliefs.count("TRUMP") / len(neighbor_beliefs)
            biden_ratio = 1 - trump_ratio
            if action == "TRUMP" and trump_ratio > 0.6:
                new_beliefs[agent_id] = "TRUMP"
            elif action == "BIDEN" and biden_ratio > 0.6:
                new_beliefs[agent_id] = "BIDEN"
        self.beliefs = new_beliefs
        self.step_count += 1
        trump_count = list(self.beliefs.values()).count("TRUMP")
        biden_count = self.num_agents - trump_count
        self.history.append((self.step_count, trump_count, biden_count))
        done = (trump_count == 0 or biden_count == 0 or self.step_count >= 30)
        return self.get_state(), {"trump": trump_count, "biden": biden_count}, done

    def get_state(self):
        return {
            "beliefs": self.beliefs.copy(),
            "step": self.step_count
        } 