# Misinformation传播模拟系统

基于group chat simulation的misinformation传播规则系统，包含多种传播策略、信任机制、验证机制等。

## 系统概述

本系统模拟了misinformation在社交媒体网络中的传播过程，包括：

- **多种传播策略**：病毒式传播、定向传播、隐蔽传播、激进传播
- **Agent信念系统**：相信者、怀疑者、中立者、事实核查者
- **社交网络影响**：基于邻居信念的社交影响
- **验证机制**：事实核查、来源验证、同行验证
- **统计和分析**：传播效果统计、信念变化分析

## 核心组件

### 1. Misinformation类型 (MisinfoType)

- **CONSPIRACY**: 阴谋论 - 政府监控、选举操纵等
- **PSEUDOSCIENCE**: 伪科学 - 健康谣言、伪科学理论等
- **POLITICAL**: 政治谣言 - 政治相关的不实信息
- **HEALTH**: 健康谣言 - 医疗健康相关谣言
- **FINANCIAL**: 金融谣言 - 投资、股市相关谣言
- **SOCIAL**: 社会谣言 - 社会事件相关谣言

### 2. Agent信念类型 (AgentBelief)

- **BELIEVER**: 相信者 - 容易相信misinformation
- **SKEPTIC**: 怀疑者 - 对信息持怀疑态度
- **NEUTRAL**: 中立者 - 对信息持中立态度
- **FACT_CHECKER**: 事实核查者 - 主动验证信息真实性

### 3. 传播策略 (PropagationStrategy)

#### 病毒式传播 (VIRAL)
- 基于情感影响和传播性
- 社交网络放大效应
- 影响力放大

#### 定向传播 (TARGETED)
- 目标受众匹配
- 基于信念类型的匹配
- 关键词匹配

#### 隐蔽传播 (STEALTH)
- 降低怀疑者检测概率
- 基于内容可信度的隐蔽性
- 社交网络隐蔽性

#### 激进传播 (AGGRESSIVE)
- 高情感影响
- 社交压力
- 群体效应
- 降低怀疑阈值

## 传播规则

### 1. 传播概率计算

每种传播策略都有不同的概率计算公式：

```python
# 病毒式传播
base_prob = content.virality * content.emotional_impact
network_amplification = len(agent.social_network) * 0.1
influence_amplification = agent.influence_score * 0.2
propagation_prob = min(1.0, base_prob + network_amplification + influence_amplification)
```

### 2. 信念更新规则

#### 基于接触的信念更新
- 计算接触后的信念变化概率
- 基于当前信念的抵抗力
- 向believer方向变化

#### 基于社交影响的信念更新
- 统计邻居的信念分布
- 计算主流信念
- 社交影响概率

### 3. 验证机制

#### 事实核查验证
- 基于agent的怀疑程度进行验证
- 不同信念类型的验证准确率不同

#### 来源验证
- 基于来源可信度的验证
- 官方、已验证、未知、可疑来源

#### 同行验证
- 基于社交网络的同行验证
- 计算同行共识

## 使用方法

### 1. 基础模拟

```python
# 创建模拟器
simulation = OasisMisinfoSimulation(profile_path="user_data_36.json")

# 运行模拟
results = simulation.run_simulation(
    steps=50,
    propagation_strategy=PropagationStrategy.VIRAL
)

# 保存结果
simulation.save_results("misinfo_simulation_results.json")
```

### 2. 多策略对比

```python
strategies = [
    PropagationStrategy.VIRAL,
    PropagationStrategy.TARGETED,
    PropagationStrategy.STEALTH,
    PropagationStrategy.AGGRESSIVE
]

for strategy in strategies:
    results = simulation.run_simulation(
        steps=30,
        propagation_strategy=strategy
    )
    simulation.save_results(f"simulation_{strategy.value}.json")
```

## 输出结果

### 1. 统计信息

- **信念分布**: 各信念类型agent的百分比
- **平均影响力**: 所有agent的平均影响力分数
- **平均怀疑度**: 所有agent的平均怀疑程度
- **总接触次数**: 所有agent接触misinformation的总次数
- **平均接触次数**: 每个agent平均接触misinformation的次数

### 2. 传播记录

```json
{
  "source_agent": 1,
  "target_agent": 5,
  "content_id": "misinfo_001",
  "content_type": "conspiracy",
  "probability": 0.75,
  "strategy": "viral",
  "agent_belief": "neutral"
}
```

### 3. 信念变化记录

```json
{
  "agent_id": 5,
  "old_belief": "neutral",
  "new_belief": "believer",
  "reason": "exposure",
  "content_id": "misinfo_001"
}
```

## 配置参数

### 1. Agent配置

- **信念分布**: 各信念类型的初始分布比例
- **信任阈值**: 每个agent的信任阈值范围
- **怀疑程度**: 每个agent的怀疑程度范围
- **影响力分数**: 每个agent的影响力分数范围

### 2. 传播配置

- **传播概率**: 各种传播策略的概率计算参数
- **社交影响**: 社交网络影响的计算参数
- **验证准确率**: 不同信念类型的验证准确率

### 3. 内容配置

- **可信度**: misinformation内容的可信度
- **传播性**: misinformation内容的传播性
- **情感影响**: misinformation内容的情感影响
- **关键词**: 用于匹配的关键词列表

## 分析功能

### 1. 传播效果分析

- 不同策略的传播效果对比
- 不同信念类型的传播敏感性
- 社交网络结构对传播的影响

### 2. 信念变化分析

- 信念变化的趋势分析
- 不同因素对信念变化的影响
- 信念稳定的条件分析

### 3. 验证效果分析

- 不同验证方法的有效性
- 验证对传播的抑制作用
- 验证准确率的影响因素

## 扩展功能

### 1. 自定义传播规则

可以添加新的传播策略：

```python
def custom_propagation(self, agent, content, network):
    # 自定义传播逻辑
    return propagation_probability
```

### 2. 自定义验证机制

可以添加新的验证方法：

```python
def custom_verification(self, agent, content):
    # 自定义验证逻辑
    return verification_result
```

### 3. 自定义信念更新规则

可以添加新的信念更新逻辑：

```python
def custom_belief_update(self, agent, content):
    # 自定义信念更新逻辑
    return new_belief
```

## 文件结构

```
demo/
├── misinformation_simulation.py          # 基础misinformation传播模拟
├── oasis_misinfo_simulation.py          # 基于Oasis框架的模拟
├── README_misinfo_simulation.md         # 说明文档
└── results/                             # 结果文件目录
    ├── misinfo_simulation_results.json
    ├── oasis_misinfo_simulation_viral.json
    ├── oasis_misinfo_simulation_targeted.json
    ├── oasis_misinfo_simulation_stealth.json
    └── oasis_misinfo_simulation_aggressive.json
```

## 运行示例

```bash
# 运行基础模拟
python misinformation_simulation.py

# 运行Oasis框架模拟
python oasis_misinfo_simulation.py
```

## 注意事项

1. **数据隐私**: 确保使用的profile数据符合隐私保护要求
2. **结果解释**: 模拟结果仅供参考，不代表真实传播情况
3. **参数调优**: 根据实际需求调整各种参数
4. **扩展性**: 系统设计支持扩展新的传播规则和验证机制

## 未来改进

1. **真实数据集成**: 集成真实的社交媒体数据
2. **机器学习**: 使用机器学习优化传播规则
3. **可视化**: 添加传播过程的可视化功能
4. **实时模拟**: 支持实时misinformation传播模拟
5. **多平台支持**: 扩展到多个社交媒体平台 