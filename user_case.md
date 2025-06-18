# SandGraphX 用户案例：LLM社交网络模拟

## 用户输入

用户需要提供以下输入：

1. **OASIS 接口实现**
   ```python
   class YourOASIS:
       def get_network_state(self):
           # 返回当前网络状态
           return {
               "users": [...],  # 用户列表
               "connections": [...],  # 用户间连接
               "user_profiles": {...}  # 用户画像
           }
       
       def get_recent_posts(self):
           # 返回最近的发帖内容
           return {
               "user_id": ["post_content", ...],
               ...
           }
       
       def get_interactions(self):
           # 返回用户互动记录
           return [
               {
                   "from": "user_id",
                   "to": "user_id",
                   "type": "interaction_type",
                   "content": "interaction_content"
               },
               ...
           ]
   ```

2. **模拟参数**
   - 模拟步数
   - LLM 模型选择
   - 工作流配置

## SandGraphX 功能（黑盒）

SandGraphX 将自动处理以下任务：

1. **环境管理**
   - 维护社交网络状态
   - 追踪用户互动
   - 管理内容发布

2. **智能决策**
   - 分析网络状态
   - 生成行动建议
   - 优化网络结构

3. **内容生成**
   - 创建社交内容
   - 生成互动响应
   - 优化内容策略

4. **工作流执行**
   - 协调节点执行
   - 管理数据流转
   - 处理状态更新

## 使用示例

```python
# 1. 创建 OASIS 接口
class YourOASIS:
    def get_network_state(self):
        return {"users": ["user1", "user2"]}
    
    def get_recent_posts(self):
        return {"user1": ["Hello!"]}
    
    def get_interactions(self):
        return [{"from": "user1", "to": "user2", "type": "like"}]

# 2. 运行模拟
oasis = YourOASIS()
results = run_social_network_simulation(oasis, steps=10)
```

## 输出结果

每次模拟步骤将返回：

1. **网络状态更新**
   ```python
   {
       "network_state": {
           "users": [...],
           "connections": [...],
           "user_profiles": {...}
       }
   }
   ```

2. **新发帖内容**
   ```python
   {
       "new_posts": {
           "user_id": ["post_content", ...],
           ...
       }
   }
   ```

3. **互动记录**
   ```python
   {
       "interactions": [
           {
               "from": "user_id",
               "to": "user_id",
               "type": "interaction_type",
               "content": "interaction_content"
           },
           ...
       ]
   }
   ```

## 优势

1. **自动化管理**
   - 无需手动处理网络状态
   - 自动协调节点执行
   - 智能决策和优化

2. **灵活扩展**
   - 支持自定义 OASIS 接口
   - 可调整工作流配置
   - 可扩展节点类型

3. **智能优化**
   - LLM 驱动的决策
   - 自适应内容生成
   - 网络结构优化

## 注意事项

1. 确保 OASIS 接口实现完整
2. 合理设置模拟步数
3. 根据需求选择适当的 LLM 模型
4. 监控资源使用情况 