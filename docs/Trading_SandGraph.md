# Trading SandGraph 技术文档

## 1. 系统概述

Trading SandGraph 是一个基于图结构的智能交易系统，它通过将交易决策过程分解为多个专业节点，形成一个有向工作流，实现自动化的交易决策和执行。系统支持两种交易引擎：Trading Gym 和 Backtrader。

### 1.1 核心特性

- 基于图结构的交易决策流程
- 多节点协作的智能交易系统
- 支持多种交易引擎（Trading Gym/Backtrader）
- 内置风险控制和资源管理
- 可扩展的节点系统

### 1.2 系统架构

```
+----------------+     +----------------+     +----------------+     +----------------+
| 市场分析节点    | --> | 策略生成节点    | --> | 交易执行节点    | --> | 风险评估节点    |
| Market Analyzer|     | Strategy Gen.  |     | Trading Exec.  |     | Risk Assessor  |
+----------------+     +----------------+     +----------------+     +----------------+
```

## 2. 节点系统

### 2.1 节点类型

系统包含四种核心节点：

1. **市场分析节点 (Market Analyzer)**
   - 类型：`NodeType.LLM`
   - 功能：分析市场数据，识别趋势和机会
   - 资源消耗：energy=5, tokens=3
   - 角色：市场分析师
   - 推理类型：analytical

2. **策略生成节点 (Strategy Generator)**
   - 类型：`NodeType.LLM`
   - 功能：基于市场分析生成交易策略
   - 资源消耗：energy=5, tokens=3
   - 角色：策略生成器
   - 推理类型：strategic

3. **交易执行节点 (Trading Executor)**
   - 类型：`NodeType.SANDBOX`
   - 功能：执行交易操作
   - 资源消耗：energy=10, tokens=5
   - 最大访问次数：5
   - 支持引擎：Trading Gym/Backtrader

4. **风险评估节点 (Risk Assessor)**
   - 类型：`NodeType.LLM`
   - 功能：评估交易风险
   - 资源消耗：energy=5, tokens=3
   - 角色：风险评估师
   - 推理类型：analytical

### 2.2 节点配置

每个节点都通过 `EnhancedWorkflowNode` 类进行配置：

```python
node = EnhancedWorkflowNode(
    node_id,           # 节点唯一标识
    node_type,         # 节点类型 (LLM/SANDBOX)
    llm_func,          # LLM处理函数（LLM节点）
    sandbox,           # 沙盒实例（SANDBOX节点）
    condition,         # 节点执行条件
    limits            # 节点资源限制
)
```

## 3. 交易引擎

### 3.1 Trading Gym

Trading Gym 是一个轻量级的交易环境，适合快速原型开发和测试。

#### 配置参数
```python
sandbox = TradingGymSandbox(
    initial_balance=100000.0,  # 初始资金
    trading_fee=0.001,         # 交易手续费
    max_position=0.2,          # 最大持仓比例
    symbols=["AAPL", "GOOGL", "MSFT", "AMZN"]  # 交易标的
)
```

### 3.2 Backtrader

Backtrader 是一个功能完整的回测框架，支持复杂策略和历史数据回测。

#### 配置参数
```python
sandbox = BacktraderSandbox(
    initial_cash=100000.0,     # 初始资金
    commission=0.001,          # 交易手续费
    data_source="yahoo",       # 数据源
    symbols=["AAPL", "GOOGL", "MSFT", "AMZN"],  # 交易标的
    start_date="2023-01-01",   # 回测开始日期
    end_date="2023-12-31"      # 回测结束日期
)
```

## 4. 使用方法

### 4.1 安装依赖

```bash
pip install sandgraph
pip install trading-gym  # 使用 Trading Gym 引擎
pip install backtrader   # 使用 Backtrader 引擎
pip install yfinance     # 获取市场数据
```

### 4.2 运行交易演示

1. 使用 Trading Gym（默认）：
```bash
python trading_demo.py
```

2. 使用 Backtrader：
```bash
python trading_demo.py --strategy backtrader
```

### 4.3 自定义工作流

```python
from sandgraph.core.llm_interface import create_shared_llm_manager
from sandgraph.core.sg_workflow import SG_Workflow, WorkflowMode

# 创建LLM管理器
llm_manager = create_shared_llm_manager("trading_llm")

# 创建工作流
workflow = SG_Workflow("trading_workflow", WorkflowMode.TRADITIONAL, llm_manager)

# 添加节点和边
# ... 配置节点和边 ...

# 执行工作流
result = workflow.execute_full_workflow(max_steps=10)
```

## 5. 资源管理

### 5.1 资源类型

- **Energy**: 表示计算资源消耗
- **Tokens**: 表示LLM调用次数
- **Time**: 表示执行时间限制
- **Knowledge**: 表示知识库使用量

### 5.2 资源限制

每个节点都有其资源消耗限制：
```python
limits = NodeLimits(
    resource_cost={
        "energy": 5,
        "tokens": 3
    },
    max_visits=5  # 可选，限制节点访问次数
)
```

## 6. 错误处理

系统包含多层错误处理机制：

1. 节点执行错误处理
2. 资源限制检查
3. 交易执行异常处理
4. 数据验证和清理

## 7. 性能优化

### 7.1 缓存机制

- LLM响应缓存
- 市场数据缓存
- 策略结果缓存

### 7.2 并行处理

- 多节点并行执行
- 异步数据获取
- 批量交易处理

## 8. 扩展开发

### 8.1 添加新节点

```python
class CustomNode(EnhancedWorkflowNode):
    def __init__(self):
        super().__init__(
            "custom_node",
            NodeType.LLM,
            llm_func=self.custom_llm_func,
            condition=NodeCondition(),
            limits=NodeLimits(resource_cost={"energy": 5, "tokens": 3})
        )
    
    def custom_llm_func(self, prompt: str, context: Dict[str, Any] = {}) -> str:
        # 实现自定义处理逻辑
        pass
```

### 8.2 自定义交易引擎

```python
class CustomSandbox(Sandbox):
    def __init__(self, **kwargs):
        super().__init__("custom", "自定义交易沙盒")
        # 实现自定义交易逻辑
```

<!-- ## 9. 最佳实践

1. **资源管理**
   - 合理设置节点资源限制
   - 监控资源使用情况
   - 及时清理缓存数据

2. **错误处理**
   - 实现完整的错误处理机制
   - 记录详细的错误日志
   - 提供错误恢复机制

3. **性能优化**
   - 使用缓存减少重复计算
   - 优化数据获取和处理
   - 合理设置并行度

4. **安全考虑**
   - 验证交易参数
   - 限制交易规模
   - 实现风险控制

## 10. 常见问题

1. **Q: 如何选择合适的交易引擎？**
   A: Trading Gym适合快速开发和测试，Backtrader适合复杂策略和历史回测。

2. **Q: 如何处理节点执行失败？**
   A: 系统会自动记录错误并尝试恢复，可以通过日志查看详细信息。

3. **Q: 如何扩展系统功能？**
   A: 可以通过添加新节点或自定义交易引擎来扩展系统功能。

## 11. 更新日志

### v1.0.0
- 初始版本发布
- 支持基本交易功能
- 实现核心节点系统

### v1.1.0
- 添加资源管理
- 优化错误处理
- 改进性能

## 12. 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 13. 许可证

MIT License  -->