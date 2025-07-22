from sandgraph.core.llm_interface import create_shared_llm_manager, create_lora_llm_manager
from sandgraph.core.rl_algorithms import RLTrainer, RLConfig

class LLMPolicy:
    """
    智能决策系统：支持frozen（只用LLM）、adaptive（RL微调）、lora（LoRA权重可插拔微调）三种模式，全部调用SandGraph core。
    """
    def __init__(self, mode='frozen', reward_fn=None, model_name="qwen-2", backend="vllm", url="http://localhost:8001/v1", lora_path=None):
        self.mode = mode
        self.reward_fn = reward_fn
        self.lora_path = lora_path
        if mode == 'frozen':
            self.llm_manager = create_shared_llm_manager(
                model_name=model_name,
                backend=backend,
                url=url,
                temperature=0.7
            )
            self.rl_trainer = None
        elif mode == 'adaptive':
            self.llm_manager = create_shared_llm_manager(
                model_name=model_name,
                backend=backend,
                url=url,
                temperature=0.7
            )
            rl_config = RLConfig(algorithm="PPO")
            self.rl_trainer = RLTrainer(rl_config, self.llm_manager, reward_fn=reward_fn)
        elif mode == 'lora':
            self.llm_manager = create_lora_llm_manager(
                model_name=model_name,
                lora_path=lora_path,
                backend=backend,
                url=url,
                temperature=0.7
            )
            rl_config = RLConfig(algorithm="PPO")
            self.rl_trainer = RLTrainer(rl_config, self.llm_manager, reward_fn=reward_fn)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def decide(self, prompts, state=None):
        """
        prompts: {agent_id: prompt}
        返回: {agent_id: action}
        """
        actions = {}
        if self.mode == 'frozen':
            for agent_id, prompt in prompts.items():
                resp = self.llm_manager.run(prompt)
                print(f"[LLM][Agent {agent_id}] Output: {resp}")
                if "TRUMP" in resp.upper():
                    actions[agent_id] = "TRUMP"
                else:
                    actions[agent_id] = "BIDEN"
        elif self.mode in ('adaptive', 'lora'):
            actions = self.rl_trainer.act(prompts, state)
        return actions

    def update_weights(self):
        if self.rl_trainer:
            self.rl_trainer.update_policy() 