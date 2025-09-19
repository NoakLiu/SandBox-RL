机制名
技术描述
系统效益
asyn Rollout
每个 agent 独立运行环境，无需集中同步，可在不同状态下并行生成数据
提升环境利用率 提高多样性，减少采样等待时间
asyn Weight Update
训练阶段与采样阶段解耦，trainer 按需批量更新，高效利用空闲 GPU slot
提高训练吞吐 降低系统耦合度，支持动态 agent 更新
经验队列 Streaming
使用异步队列传输经验，支持弹性缓存、优先采样与跨节点通信
 避免同步阻塞 兼容复杂调度与跨设备训练
KV Cache 协调控制
动态追踪 KV cache 占用，根据上下文长度与生成规模限制 cache 浪涌
显存占用稳定 避免多 agent 同时生成导致 OOM
并发推理调度器
控制最大并发 agent 数量，结合 reward-aware 策略优先分配计算资源给高质量 agent
保障系统稳定性 高效利用异构模型（如 frozen/trainable）资源池
注：以下包含了一些简单实现和设想、具体代码请看github repo
1. 异步架构设计
1.1 异步LLM调用
class VLLMClient:
    async def aenter(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def aexit(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def generate(self, prompt: str) -> str:
        """异步生成文本"""
        async with self.session.post(endpoint, json=payload) as response:
            result = await response.json()
            return result["choices"][0]["message"]["content"]
亮点：
- 非阻塞调用：使用`aiohttp`实现真正的异步HTTP请求
- 资源管理：通过`async with`确保连接正确关闭
- 并发处理：支持多个agent同时进行LLM决策
- 错误恢复：多重端点尝试和fallback机制

1.2 异步环境管理
class OASISCorrectSimulation:
    async def run_simulation(self):
        """异步仿真运行"""
        async with VLLMClient(self.vllm_url, self.model_name) as llm:
            for step in range(self.config.num_steps):
                step_result = await self._simulate_step(step, llm)
                self.history.append(step_result)
- 生命周期管理：完整的异步资源初始化、运行、清理流程
- 状态同步：确保所有agent状态在仿真步骤间正确同步
- 内存优化：及时释放不需要的资源
2. 智能调度系统

2.1 多模型调度策略
class LLMPolicy:
    def init(self, mode='frozen', reward_fn=None, model_name="qwen-2", 
                 backend="huggingface", url="http://localhost:8001/v1", 
                 lora_path=None, enable_monitoring=True):
        self.scheduling_strategy = 'random_model'  # 随机模型调度
        self.mode = mode  # frozen/adaptive/lora模式
        self.enable_monitoring = enable_monitoring
- 多模型支持和动态调度：根据任务需求选择合适的模型进行更新

2.2 reward-base驱动的slot管理
class RewardBasedSlotManager:
    def init(self, max_slots: int = 10):
        self.max_slots = max_slots
        self.active_slots = {}
        self.slot_rewards = {}
    
    def update_slot_reward(self, slot_id: str, new_reward: float) -> bool:
        """基于奖励更新槽位"""
        if slot_id in self.active_slots:
            self.slot_rewards[slot_id] = new_reward
            return True
        return False
    
    def update_slots(self, reward_update: float) -> bool:
        """批量更新所有运行中slot的reward值"""
        # advanced implementation可以是优先队列
        updated = False
        for slot_id in self.active_slots:
            if slot_id in self.slot_rewards:
                self.slot_rewards[slot_id] += reward_update
                updated = True
        return updated

亮点：
- 资源优化：基于奖励动态分配计算资源
- adaptive scheduling和batch processing：支持批量更新提高效率

3. 分布式架构
3.1 多Agent并行处理
async def _simulate_step(self, step: int, llm=None):
    """并行处理所有agent的决策"""
    belief_changes = 0
    actions = {}
    
    # 为每个agent生成决策（可并行化）
    for id, agent in self.agent_graph.get_agents():
        neighbors = agent.get_neighbors()
        neighbor_groups = [n.group for n in neighbors] if neighbors else []
        
        prompt = self._build_prompt(agent, neighbor_groups)
        resp = await llm.generate(prompt)  # 异步LLM调用
        
        actions[id] = self._parse_response(resp)
- 并行决策：所有agent可以同时进行LLM决策
- 异步通信：agent间通过异步消息做inference, training
3.2 异步推理和权重更新
class AsyncAgentWorkflow:
    """Sandbox-RL Workflow - 多个独立agent的异步推理和权重更新"""
    
    def init(self, agent_graph, llm_policy, slot_manager):
        self.agent_graph = agent_graph
        self.llm_policy = llm_policy
        self.slot_manager = slot_manager
        self.inference_queue = asyncio.Queue()
        self.weight_update_queue = asyncio.Queue()
        self.active_tasks = {}
    
    async def start_async_workflow(self):
        """启动异步工作流"""
        # 启动推理工作器
        inference_workers = [
            asyncio.create_task(self._inference_worker(i))
            for i in range(5)  # 5个推理工作器
        ]
        
        # 启动权重更新工作器
        weight_update_workers = [
            asyncio.create_task(self._weight_update_worker(i))
            for i in range(3)  # 3个权重更新工作器
        ]
        
        # 启动任务分发器
        dispatcher = asyncio.create_task(self._task_dispatcher())
        
        return inference_workers + weight_update_workers + [dispatcher]
    
    async def _task_dispatcher(self):
        """异步任务分发器"""
        while True:
            # 收集所有agent的推理任务
            inference_tasks = []
            for agent_id, agent in self.agent_graph.get_agents():
                if agent.needs_inference():
                    task = {
                        'agent_id': agent_id,
                        'task_type': 'inference',
                        'prompt': agent.build_inference_prompt(),
                        'priority': agent.get_priority()
                    }
                    inference_tasks.append(task)
            
            # 异步分发推理任务
            for task in inference_tasks:
                await self.inference_queue.put(task)
            
            # 收集权重更新任务
            weight_update_tasks = []
            for agent_id, agent in self.agent_graph.get_agents():
                if agent.needs_weight_update():
                    task = {
                        'agent_id': agent_id,
                        'task_type': 'weight_update',
                        'gradients': agent.get_gradients(),
                        'learning_rate': agent.get_learning_rate()
                    }
                    weight_update_tasks.append(task)
            
            # 异步分发权重更新任务
            for task in weight_update_tasks:
                await self.weight_update_queue.put(task)
            
            await asyncio.sleep(0.1)  # 避免过度占用CPU
    
    async def _inference_worker(self, worker_id: int):
        """异步推理工作器"""
        while True:
            try:
                task = await self.inference_queue.get()
                agent_id = task['agent_id']
                prompt = task['prompt']
                
                # 异步LLM推理
                async with self.llm_policy.get_llm_client() as llm:
                    response = await llm.generate(prompt)
                
                # 更新agent状态
                agent = self.agent_graph.get_agent(agent_id)
                agent.update_inference_result(response)
                
                # 记录推理完成
                self.active_tasks[f"inference_{agent_id}"] = {
                    'status': 'completed',
                    'worker_id': worker_id,
                    'timestamp': time.time()
                }
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Inference worker {worker_id} error: {e}")
    
    async def _weight_update_worker(self, worker_id: int):
        """异步权重更新工作器"""
        while True:
            try:
                task = await self.weight_update_queue.get()
                agent_id = task['agent_id']
                gradients = task['gradients']
                learning_rate = task['learning_rate']
                
                # 异步权重更新
                agent = self.agent_graph.get_agent(agent_id)
                await agent.async_update_weights(gradients, learning_rate)
                
                # 更新槽位奖励
                new_reward = agent.calculate_reward()
                self.slot_manager.update_slot_reward(f"agent_{agent_id}", new_reward)
                
                # 记录权重更新完成
                self.active_tasks[f"weight_update_{agent_id}"] = {
                    'status': 'completed',
                    'worker_id': worker_id,
                    'timestamp': time.time()
                }
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Weight update worker {worker_id} error: {e}")
    
    async def parallel_inference_batch(self, agent_ids: List[int]):
        """批量并行推理"""
        tasks = []
        for agent_id in agent_ids:
            agent = self.agent_graph.get_agent(agent_id)
            prompt = agent.build_inference_prompt()
            task = asyncio.create_task(self._single_inference(agent_id, prompt))
            tasks.append(task)
        
        # 等待所有推理任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    async def _single_inference(self, agent_id: int, prompt: str):
        """单个agent的异步推理"""
        async with self.llm_policy.get_llm_client() as llm:
            response = await llm.generate(prompt)
            agent = self.agent_graph.get_agent(agent_id)
            agent.update_inference_result(response)
            return response
    
    async def parallel_weight_update_batch(self, agent_ids: List[int]):
        """批量并行权重更新"""
        tasks = []
        for agent_id in agent_ids:
            agent = self.agent_graph.get_agent(agent_id)
            gradients = agent.get_gradients()
            learning_rate = agent.get_learning_rate()
            task = asyncio.create_task(
                agent.async_update_weights(gradients, learning_rate)
            )
            tasks.append(task)
        
        # 等待所有权重更新任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
- 独立异步推理：每个agent独立进行LLM推理，互不阻塞
- 权重更新机制：多个agent可用异步锁更新模型权重
- Queue or Pool of workflow：使用工作池和异步队列处理推理和权重更新任务
- 批量处理：支持批量推理和权重更新提高效率
- 资源优化：根据任务优先级动态分配计算资源
3.3 沙盒与LLM weight update和inference隔离
class OASISSandbox:
    """Sandbox-RL Sandbox - 不同信念的agents分组管理"""
    
    def init(self, belief_type: BeliefType, agents: List[OASISAgentState]):
        self.belief_type = belief_type
        self.agents = agents
        self.total_influence = sum(agent.influence_score for agent in agents)
    
    def add_agent(self, agent: OASISAgentState):
        """动态添加agent到sandbox"""
        self.agents.append(agent)
        self.total_influence += agent.influence_score
    
    def remove_agent(self, agent_id: int):
        """从sandbox移除agent"""
        for i, agent in enumerate(self.agents):
            if agent.agent_id == agent_id:
                self.total_influence -= agent.influence_score
                del self.agents[i]
                break
- 资源隔离：不同agents在相互独立沙盒中运行
- 动态管理：支持agent在沙盒间的动态迁移

4. 奖励引导的路由与模型专用化

尽管 AReaL 系统引入了异步 rollout 与异步训练机制以提升系统效率，但其默认假设所有 agent 使用统一的策略更新路径。然而，在真实多智能体场景中，不同 agent 往往呈现异质的行为模式、奖励水平及学习进度。为此，我们提出一种基于奖励门控的模型选择机制，结合 KV cache 感知调度和选择性权重更新策略，从系统和算法两个角度提升整体性能。
4.1 奖励门控的模型选择机制（Reward-Gated Model Selection）
我们设计了一个动态模型选择模块，根据 agent 最近的奖励信号将其路由到不同的策略实例上。每个 agent a 维护一个滑动平均奖励值 $R(a)$，我们选取前 $$k$$ 个奖励值最高的 agent 分配至共享的可训练策略，其余则使用冻结策略：
\mathcal{A}_{\text{train}} = \operatorname{argmax}_{a \in \mathcal{A}}^{[k]} R(a)

每个 rollout worker 本地维护一个路由表，并根据 agent id 实时查询其当前使用的策略。该路由表由中央控制器定期更新，从而将训练资源集中用于高奖励 agent，提升样本效率和训练稳定性。
4.2 多智能体异步 Rollout
我们扩展 rollout worker 池，使其支持基于 agent 的并行环境及异步经验上传机制。每个 agent 使用各自分配的策略生成 trajectory，并附带行为策略版本号上传：
trajectory = {
    "agent_id": a,
    "prompt": q,
    "actions": [a1, ..., aH],
    "policy_version": v_behav,
    "prox_version": v_prox
}
整个流程无需中心同步，兼容 AReaL 的中断式采样机制，且与解耦版 PPO 目标函数保持一致。

4.3 选择性权重更新
为避免低效的训练更新，我们仅对 \(\mathcal{A}_{\text{train}}\) 中的 agent 进行策略优化。其余 agent 的数据仅用于评估或可视化：
if agent_id in A_train:
    include_in_ppo_update = True
该策略可显著提升训练稳定性，避免 reward signal 噪声干扰主策略收敛。
4.4 KV Cache 感知的并发调度机制
由于可训练策略添加了 KV 缓存、梯度状态、优化器变量等资源，我们引入一个调度器，限制 simultaneously-active trainable agent 数量。每次 rollout 前估计该 agent 的记忆占用：
\text{Mem}(a) = \text{len(prompt)} + \text{len(generated)} + \text{KV}_\theta
如果所有并发 agent 的资源需求超过预约预算 $B$，该请求将被延后处理，从而有效防止 OOM 并提升运行稳定性。

5. 未来发展方向
1. 增强异步能力
- 分布式部署：支持多机集群部署
- 流式处理：实现LLM响应的流式处理
- 缓存机制：智能缓存减少重复LLM调用
2. 优化调度策略
- adaptive scheduling：基于历史数据优化调度决策,基于agent优先级和资源状态智能调度任务, 或者预测资源需求提前分配
- load-balancing：根据负载自动调整资源规模
3. 增强异步推理和权重更新
- 分布式推理：支持跨多机的分布式推理
- 增量权重更新：实现增量式权重更新减少计算开销
- 自适应工作器池：根据负载动态调整工作器数量