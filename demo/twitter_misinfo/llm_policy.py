from sandgraph.core.llm_interface import create_shared_llm_manager
from sandgraph.core.rl_algorithms import RLTrainer, RLConfig

class LLMPolicy:
    """
    智能决策系统：支持frozen（只用LLM）和adaptive（RL微调）两种模式，集成SandGraph LLM和RLTrainer。
    """
    def __init__(self, mode='frozen', reward_fn=None, model_name="qwen-2", backend="vllm", url="http://localhost:8001/v1"):
        self.mode = mode
        self.llm_manager = create_shared_llm_manager(
            model_name=model_name,
            backend=backend,
            url=url,
            temperature=0.7
        )
        self.reward_fn = reward_fn
        if mode == 'adaptive':
            rl_config = RLConfig(algorithm="PPO")
            self.rl_trainer = RLTrainer(rl_config, self.llm_manager, reward_fn=reward_fn)
        else:
            self.rl_trainer = None

    def decide(self, prompts, state=None):
        """
        prompts: {agent_id: prompt}
        返回: {agent_id: action}
        """
        actions = {}
        # frozen: 直接用LLM决策
        if self.mode == 'frozen':
            for agent_id, prompt in prompts.items():
                resp = self.llm_manager.run(prompt)
                print(f"[LLM][Agent {agent_id}] Output: {resp}")
                if "TRUMP" in resp.upper():
                    actions[agent_id] = "TRUMP"
                else:
                    actions[agent_id] = "BIDEN"
        # adaptive: RL微调，权重更新
        elif self.mode == 'adaptive':
            # RLTrainer会自动采样、更新policy
            actions = self.rl_trainer.act(prompts, state)
        return actions

    def update_weights(self):
        if self.rl_trainer:
            self.rl_trainer.update_policy() 