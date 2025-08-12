# VLLM集成说明

## 概述

本多模型训练系统已集成VLLM接口，支持使用Camel和Oasis框架的VLLM模型进行训练和推理。

## 集成方式

### 1. Camel和Oasis VLLM接口

系统优先使用Camel和Oasis提供的VLLM接口：

```python
from camel.models import ModelFactory
from camel.types import ModelPlatformType

# 创建VLLM模型
vllm_model = ModelFactory.create(
    model_platform=ModelPlatformType.VLLM,
    model_type="qwen-2",
    url="http://localhost:8001/v1",
)
```

### 2. 备用HTTP客户端

如果Camel和Oasis不可用，系统会回退到HTTP客户端：

```python
import aiohttp

async with aiohttp.ClientSession() as session:
    payload = {
        "model": "qwen-2",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    
    async with session.post(f"{vllm_url}/chat/completions", json=payload) as response:
        result = await response.json()
        content = result["choices"][0]["message"]["content"]
```

### 3. 模拟模式

如果所有VLLM接口都不可用，系统会使用模拟响应。

## 使用方法

### 1. 启动VLLM服务器

```bash
# 启动VLLM服务器
python -m vllm.entrypoints.openai.api_server \
    --model qwen/Qwen2-7B-Instruct \
    --host 0.0.0.0 \
    --port 8001
```

### 2. 运行多模型训练

```bash
# 运行训练系统
python demo/multi_model_single_env_simple.py
```

### 3. 测试VLLM集成

```bash
# 测试Camel和Oasis VLLM接口
python demo/camel_oasis_vllm_example.py
```

## 配置说明

### VLLM客户端配置

```python
class VLLMClient:
    def __init__(self, url: str = "http://localhost:8001/v1", model_name: str = "qwen-2"):
        self.url = url
        self.model_name = model_name
        self.camel_model = None
        self.connection_available = False
        self._initialize_camel_model()
```

### 多模型环境配置

```python
env = MultiModelEnvironment(
    vllm_url="http://localhost:8001/v1",
    training_mode=TrainingMode.COOPERATIVE,
    max_models=10
)
```

## 功能特性

### 1. 自动检测

系统会自动检测VLLM服务器的可用性：

- Camel和Oasis VLLM接口
- HTTP客户端
- 模拟模式

### 2. 智能回退

如果主要接口失败，系统会自动回退到备用方案：

1. Camel和Oasis VLLM接口
2. HTTP客户端
3. 模拟响应

### 3. 任务提示词

系统会为每个任务构建专门的提示词：

```python
def _build_task_prompt(self, task: TrainingTask, strategy: Dict[str, Any], 
                      other_models: List['LoRAModel']) -> str:
    prompt = f"""
你是一个AI模型，正在参与多模型训练任务。

任务信息:
- 任务ID: {task.task_id}
- 任务类型: {task.task_type}
- 难度: {task.difficulty:.2f}
- 奖励池: {task.reward_pool:.2f}
- 合作级别: {task.cooperation_level:.2f}

你的角色: {self.config.role.value}
你的团队: {self.config.team_id or '无'}
你的专长: {self.config.specialization}

当前策略:
- 模式: {strategy.get('mode', 'unknown')}
- 团队合作级别: {strategy.get('teamwork_level', 0):.2f}
- 通信: {strategy.get('communication', False)}
- 资源共享: {strategy.get('resource_sharing', False)}

其他模型数量: {len(other_models)}

请根据以上信息，为这个任务提供执行策略建议。考虑你的角色、任务类型和合作级别。
"""
    return prompt
```

### 4. 性能优化

VLLM响应会影响模型性能：

```python
# VLLM响应质量加成
vllm_bonus = 0.1 if "协同" in vllm_response or "合作" in vllm_response else 0.05
if "竞争" in vllm_response or "优化" in vllm_response:
    vllm_bonus += 0.05

# 应用到性能计算
accuracy = min(1.0, base_accuracy + cooperation_bonus + vllm_bonus)
efficiency = min(1.0, base_efficiency + cooperation_bonus + vllm_bonus)
```

## 训练模式

### 1. 协同训练

模型之间相互合作，共享VLLM生成的策略建议。

### 2. 竞争训练

模型独立使用VLLM生成竞争策略。

### 3. 组队博弈

团队内部共享VLLM策略，团队间竞争。

### 4. 混合训练

结合多种训练方式，动态调整VLLM使用策略。

## 监控和日志

### VLLM调用日志

```
🤖 Camel VLLM生成: 基于当前任务分析，我建议采用协同策略...
🤖 HTTP VLLM生成: 通过竞争机制可以获得更好的性能表现...
🤖 模拟VLLM生成: 团队合作是解决复杂问题的关键...
```

### 性能指标

- VLLM响应质量
- 调用成功率
- 响应时间
- 策略效果

## 故障排除

### 1. VLLM服务器连接失败

```bash
# 检查VLLM服务器状态
curl http://localhost:8001/v1/models

# 重启VLLM服务器
python -m vllm.entrypoints.openai.api_server \
    --model qwen/Qwen2-7B-Instruct \
    --host 0.0.0.0 \
    --port 8001
```

### 2. Camel和Oasis导入失败

```bash
# 安装依赖
pip install camel-ai oasis

# 检查安装
python -c "import camel; import oasis; print('OK')"
```

### 3. 性能问题

- 检查VLLM服务器资源使用情况
- 调整模型参数（max_tokens, temperature）
- 优化提示词长度

## 扩展功能

### 1. 自定义VLLM模型

```python
# 添加自定义模型
custom_model = ModelFactory.create(
    model_platform=ModelPlatformType.VLLM,
    model_type="custom-model",
    url="http://localhost:8002/v1",
)
```

### 2. 多VLLM服务器

```python
# 支持多个VLLM服务器
vllm_servers = [
    "http://localhost:8001/v1",
    "http://localhost:8002/v1",
    "http://localhost:8003/v1"
]
```

### 3. 负载均衡

```python
# 实现负载均衡
import random
vllm_url = random.choice(vllm_servers)
```

## 总结

VLLM集成为多模型训练系统提供了强大的推理能力：

- ✅ 支持Camel和Oasis VLLM接口
- ✅ 自动回退机制
- ✅ 智能提示词构建
- ✅ 性能优化
- ✅ 完整的监控和日志
- ✅ 灵活的配置选项

通过VLLM集成，系统能够：

1. 生成更智能的训练策略
2. 提高模型协作效率
3. 优化资源分配
4. 增强训练效果
5. 支持复杂的多模型交互
