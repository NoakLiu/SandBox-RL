from sandbox import TwitterMisinformationSandbox
from llm_policy import LLMPolicy
from reward import trump_dominance_reward, slot_reward

class TwitterMisinfoWorkflow:
    """
    工作流图：组织沙盒、LLM、奖励、RL等节点，支持多阶段仿真、对抗、权重更新、slot reward。
    """
    def __init__(self, agent_graph, reward_fn=trump_dominance_reward, llm_mode='frozen'):
        self.sandbox = TwitterMisinformationSandbox(agent_graph)
        self.llm_policy = LLMPolicy(mode=llm_mode, reward_fn=reward_fn)
        self.reward_fn = reward_fn
        self.state = self.sandbox.get_state()
        self.rewards = []
        self.slot_rewards = []

    def run(self, max_steps=30):
        history = []
        for step in range(max_steps):
            prompts = self.sandbox.get_prompts()
            actions = self.llm_policy.decide(prompts, self.state)
            next_state, score, done = self.sandbox.step(actions)
            reward = self.reward_fn(self.state, actions, next_state)
            slot_r = slot_reward(self.state, actions, next_state)
            self.rewards.append(reward)
            self.slot_rewards.append(slot_r)
            self.state = next_state
            history.append(score)
            print(f"Step {step+1}: TRUMP={score['trump']} BIDEN={score['biden']} Reward={reward:.3f} SlotReward={slot_r:.3f}")
            # RL权重更新
            if self.llm_policy.mode == 'adaptive':
                self.llm_policy.update_weights()
            if done:
                break
        return history, self.rewards, self.slot_rewards 