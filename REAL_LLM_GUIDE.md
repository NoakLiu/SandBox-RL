# SandGraph 真实LLM使用指南

本指南介绍如何在SandGraph框架中使用真实的大语言模型，包括GPT-2、LLaMA、Qwen和OpenAI API。

## 支持的LLM后端

### 1. Mock LLM
- **用途**: 测试和演示
- **优点**: 无需额外依赖，快速启动
- **缺点**: 仅模拟响应，无真实推理能力

### 2. HuggingFace模型
- **支持模型**: GPT-2、LLaMA、Qwen、ChatGLM等
- **优点**: 本地运行，隐私保护，可自定义
- **缺点**: 需要GPU资源，模型下载较大

### 3. OpenAI API
- **支持模型**: GPT-3.5-turbo、GPT-4等
- **优点**: 性能强大，无需本地资源
- **缺点**: 需要API密钥，有使用成本

## 安装依赖

### 基础依赖
```bash
# SandGraph核心依赖
pip install -r requirements.txt
```

### HuggingFace模型依赖
```bash
# PyTorch (根据你的系统选择合适版本)
pip install torch torchvision torchaudio

# HuggingFace库
pip install transformers accelerate

# 可选：GPU加速
pip install bitsandbytes  # 量化支持
```

### OpenAI API依赖
```bash
pip install openai
```

## 使用示例

### 1. 创建GPT-2模型管理器

```python
from sandgraph.core.llm_interface import create_gpt2_manager

# 创建GPT-2管理器
llm_manager = create_gpt2_manager(
    model_size="gpt2",  # 可选: "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"
    device="auto"       # 自动选择设备，或指定 "cpu", "cuda"
)

# 加载模型
llm_manager.load_model()

# 注册节点
llm_manager.register_node("gpt2_node", {
    "role": "文本生成助手",
    "temperature": 0.7
})

# 生成文本
response = llm_manager.generate_for_node(
    "gpt2_node", 
    "The future of AI is",
    max_length=100
)
print(response.text)

# 卸载模型释放内存
llm_manager.unload_model()
```

### 2. 创建LLaMA模型管理器

```python
from sandgraph.core.llm_interface import create_llama_manager

# 创建LLaMA管理器
llm_manager = create_llama_manager(
    model_path="meta-llama/Llama-2-7b-chat-hf",  # HuggingFace模型名或本地路径
    device="auto"
)

# 使用方法与GPT-2相同
llm_manager.load_model()
# ... 其他操作
```

### 3. 创建Qwen模型管理器

```python
from sandgraph.core.llm_interface import create_qwen_manager

# 创建Qwen管理器
llm_manager = create_qwen_manager(
    model_name="Qwen/Qwen-7B-Chat",  # 或其他Qwen模型
    device="auto"
)

# 使用方法相同
llm_manager.load_model()
# ... 其他操作
```

### 4. 创建OpenAI API管理器

```python
import os
from sandgraph.core.llm_interface import create_openai_manager

# 设置API密钥
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# 创建OpenAI管理器
llm_manager = create_openai_manager(
    model_name="gpt-3.5-turbo",  # 或 "gpt-4"
    api_key=os.getenv("OPENAI_API_KEY")
)

# 注册节点
llm_manager.register_node("openai_node", {
    "role": "智能助手",
    "temperature": 0.7
})

# 生成响应
response = llm_manager.generate_for_node(
    "openai_node",
    "解释什么是强化学习",
    max_tokens=200
)
print(response.text)
```

### 5. 在工作流中使用真实LLM

```python
from sandgraph.core.sg_workflow import SG_Workflow, WorkflowMode
from sandgraph.core.llm_interface import create_gpt2_manager

# 创建LLM管理器
llm_manager = create_gpt2_manager(device="cpu")
llm_manager.load_model()

# 创建纯沙盒工作流（所有节点都是沙盒，但使用LLM推理）
graph = SG_Workflow("real_llm_workflow", WorkflowMode.SANDBOX_ONLY, llm_manager)

# 添加节点...
# 执行工作流...

# 清理
llm_manager.unload_model()
```

### 6. 在强化学习中使用真实LLM

```python
from sandgraph.core.rl_framework import create_rl_framework, RLAlgorithm

# 创建RL框架，使用GPT-2
rl_framework = create_rl_framework(
    model_name="gpt2",
    algorithm=RLAlgorithm.PPO,
    learning_rate=1e-4
)

# 加载模型
rl_framework.llm_manager.load_model()

# 创建RL节点
rl_llm_func = rl_framework.create_rl_enabled_llm_node(
    "rl_node",
    {"role": "问题解决器", "temperature": 0.7}
)

# 进行RL训练...

# 清理
rl_framework.llm_manager.unload_model()
```

## 配置选项

### LLMConfig参数说明

```python
from sandgraph.core.llm_interface import LLMConfig, LLMBackend

config = LLMConfig(
    backend=LLMBackend.HUGGINGFACE,  # 后端类型
    model_name="gpt2",               # 模型名称
    device="auto",                   # 设备: "cpu", "cuda", "auto"
    max_length=512,                  # 最大生成长度
    temperature=0.7,                 # 温度参数
    top_p=0.9,                      # Top-p采样
    top_k=50,                       # Top-k采样
    do_sample=True,                 # 是否采样
    torch_dtype="auto",             # 数据类型: "float16", "float32", "auto"
    
    # API相关
    api_key="your-api-key",         # API密钥
    api_base="https://api.openai.com/v1",  # API基础URL
    
    # 模型路径
    model_path="/path/to/model",    # 本地模型路径
    cache_dir="/path/to/cache",     # 缓存目录
    
    # 性能配置
    batch_size=1,                   # 批次大小
    use_cache=True                  # 是否使用缓存
)
```

## 性能优化建议

### 1. 内存优化
```python
# 使用float16减少内存占用
config = LLMConfig(
    torch_dtype="float16",
    device="cuda"
)

# 及时卸载模型
llm_manager.unload_model()
```

### 2. GPU加速
```python
# 自动选择最佳设备
config = LLMConfig(device="auto")

# 使用GPU加速
config = LLMConfig(device="cuda")
```

### 3. 批处理
```python
# 对于多个请求，考虑批处理
# (需要自定义实现)
```

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```python
   # 使用CPU或更小的模型
   llm_manager = create_gpt2_manager(device="cpu")
   
   # 或使用float16
   config = LLMConfig(torch_dtype="float16")
   ```

2. **模型下载失败**
   ```bash
   # 设置HuggingFace镜像
   export HF_ENDPOINT=https://hf-mirror.com
   ```

3. **OpenAI API错误**
   ```python
   # 检查API密钥
   import os
   print(os.getenv("OPENAI_API_KEY"))
   
   # 检查网络连接
   # 检查API配额
   ```

### 调试技巧

```python
import logging

# 启用详细日志
logging.basicConfig(level=logging.INFO)

# 检查模型状态
stats = llm_manager.get_global_stats()
print(f"模型已加载: {stats['llm_internal_stats']['model_loaded']}")
print(f"设备: {stats['llm_internal_stats'].get('device', 'unknown')}")
```

## 运行演示

```bash
# 运行真实LLM演示
python real_llm_demo.py

# 运行增强演示（包含真实LLM支持）
python sg_workflow_demo.py
```

## 注意事项

1. **资源需求**: 真实LLM需要大量内存和计算资源
2. **首次运行**: 会下载模型文件，需要时间和网络
3. **API成本**: OpenAI API按使用量收费
4. **隐私考虑**: 本地模型更安全，API模型需要发送数据到外部服务
5. **许可证**: 确保遵守模型的使用许可证

## 扩展支持

要添加新的LLM后端，可以：

1. 继承`BaseLLM`类
2. 实现必要的方法
3. 在`create_llm`函数中添加支持
4. 创建便利函数

```python
class CustomLLM(BaseLLM):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        # 初始化自定义LLM
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        # 实现生成逻辑
        pass
    
    # 实现其他必要方法...
``` 