import matplotlib.pyplot as plt
from sandgraph.sandbox_implementations import MisinformationSpreadSandbox
from sandgraph.core.workflow import WorkflowGraph, WorkflowNode, NodeType

# === 1. 初始化 vLLM (OASIS) ===
from camel.models import ModelFactory
from camel.types import ModelPlatformType

llm = ModelFactory.create(
    model_platform=ModelPlatformType.VLLM,
    model_type="qwen-2",
    url="http://localhost:8001/v1",
)

def vllm_llm_decision(prompts):
    prompt_list = [prompt for _, prompt in sorted(prompts.items())]
    responses = [llm.generate(prompt) for prompt in prompt_list]
    actions = {}
    for (agent, _), resp in zip(sorted(prompts.items()), responses):
        print(f"[LLM][Agent {agent}] Output: {resp}")
        if "TRUMP" in str(resp).upper():
            actions[agent] = "TRUMP"
        else:
            actions[agent] = "BIDEN"
    return actions

def main():
    sandbox = MisinformationSpreadSandbox(num_agents=50, edge_prob=0.12, seed=42)
    graph = WorkflowGraph("misinfo_spread")
    env_node = WorkflowNode("env", NodeType.SANDBOX, sandbox=sandbox)
    llm_node = WorkflowNode("llm_decision", NodeType.LLM, llm_func=vllm_llm_decision)
    graph.add_node(env_node)
    graph.add_node(llm_node)
    graph.add_edge("env", "llm_decision")
    graph.add_edge("llm_decision", "env")

    state = sandbox.case_generator()
    round_history, flat_history = [], []
    for _ in range(30):
        prompts = sandbox.prompt_func(state)
        actions = vllm_llm_decision(prompts)
        state, score, done = sandbox.execute(actions)
        round_history.append(score["round"])
        flat_history.append(score["flat"])
        print(f"Step {_+1}: ROUND={score['round']} FLAT={score['flat']}")
        if done:
            break

    plt.plot(round_history, label="Earth is round")
    plt.plot(flat_history, label="Earth is flat")
    plt.xlabel("Step")
    plt.ylabel("Agent count")
    plt.legend()
    plt.title("Misinformation Spread Simulation (vLLM)")
    plt.show()

if __name__ == "__main__":
    main() 