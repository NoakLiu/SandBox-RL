#!/usr/bin/env python3
"""
使用Camel和Oasis VLLM接口的示例
"""

import asyncio
import os
import sys
from typing import Dict, List, Any

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def test_camel_oasis_vllm():
    """测试Camel和Oasis VLLM接口"""
    print("🧪 测试Camel和Oasis VLLM接口...")
    print("=" * 50)
    
    try:
        # 导入Camel和Oasis
        from camel.models import ModelFactory
        from camel.types import ModelPlatformType
        import oasis
        from oasis import ActionType, LLMAction, ManualAction, generate_reddit_agent_graph
        
        print("✅ 成功导入Camel和Oasis")
        
        # 创建VLLM模型
        print("\n🔧 创建VLLM模型...")
        vllm_model_1 = ModelFactory.create(
            model_platform=ModelPlatformType.VLLM,
            model_type="qwen-2",
            url="http://localhost:8001/v1",
        )
        vllm_model_2 = ModelFactory.create(
            model_platform=ModelPlatformType.VLLM,
            model_type="qwen-2",
            url="http://localhost:8001/v1",
        )
        models = [vllm_model_1, vllm_model_2]
        
        print(f"✅ 创建了 {len(models)} 个VLLM模型")
        
        # 定义可用动作
        available_actions = [
            ActionType.CREATE_POST,
            ActionType.LIKE_POST,
            ActionType.REPOST,
            ActionType.FOLLOW,
            ActionType.DO_NOTHING,
            ActionType.QUOTE_POST,
        ]
        
        print(f"✅ 定义了 {len(available_actions)} 个可用动作")
        
        # 生成agent图
        print("\n🔧 生成agent图...")
        agent_graph = await generate_reddit_agent_graph(
            profile_path="user_data_36.json",
            model=models,
            available_actions=available_actions,
        )
        
        print(f"✅ 生成了agent图，包含 {len(list(agent_graph.get_agents()))} 个agents")
        
        # 给每个agent分配组
        import random
        trump_ratio = 0.5  # 50% Trump, 50% Biden
        agent_ids = [id for id, _ in agent_graph.get_agents()]
        trump_agents = set(random.sample(agent_ids, int(len(agent_ids) * trump_ratio)))
        
        for id, agent in agent_graph.get_agents():
            agent.group = "TRUMP" if id in trump_agents else "BIDEN"
        
        print(f"✅ 分配了agent组: TRUMP={len(trump_agents)}, BIDEN={len(agent_ids)-len(trump_agents)}")
        
        # 创建环境
        print("\n🔧 创建Oasis环境...")
        db_path = "camel_oasis_test.db"
        
        # 删除旧数据库
        if os.path.exists(db_path):
            os.remove(db_path)
        
        env = oasis.make(
            agent_graph=agent_graph,
            platform=oasis.DefaultPlatformType.TWITTER,
            database_path=db_path,
        )
        
        print("✅ 创建了Oasis环境")
        
        # 运行环境
        print("\n🔄 运行环境...")
        await env.reset()
        
        # 执行一些动作
        print("📝 执行手动动作...")
        actions_1 = {}
        actions_1[env.agent_graph.get_agent(0)] = ManualAction(
            action_type=ActionType.CREATE_POST,
            action_args={"content": "Multi-model training is amazing!"}
        )
        await env.step(actions_1)
        
        print("🤖 执行LLM动作...")
        actions_2 = {
            agent: LLMAction()
            for _, agent in env.agent_graph.get_agents([1, 3, 5, 7, 9])
        }
        await env.step(actions_2)
        
        # 模拟多模型训练场景
        print("\n🎯 模拟多模型训练场景...")
        for step in range(5):
            print(f"   Step {step + 1}:")
            
            # 使用VLLM模型生成响应
            for id, agent in agent_graph.get_agents():
                neighbors = agent.get_neighbors()
                neighbor_groups = [n.group for n in neighbors]
                
                prompt = (
                    f"You are a {agent.group} supporter. "
                    f"Your neighbors' groups: {neighbor_groups}. "
                    "Will you post/forward TRUMP or BIDEN message this round?"
                )
                
                # 使用VLLM模型生成响应
                try:
                    # 使用正确的Camel VLLM API格式
                    messages = [{"role": "user", "content": prompt}]
                    resp = await vllm_model_1.arun(messages)
                    print(f"     Agent {id} ({agent.group}): {resp[:50]}...")
                except Exception as e:
                    print(f"     Agent {id} ({agent.group}): VLLM调用失败 - {e}")
            
            # 统计组分布
            trump_count = sum(1 for _, agent in agent_graph.get_agents() if agent.group == "TRUMP")
            biden_count = sum(1 for _, agent in agent_graph.get_agents() if agent.group == "BIDEN")
            print(f"     TRUMP={trump_count}, BIDEN={biden_count}")
        
        # 关闭环境
        await env.close()
        
        print("\n✅ Camel和Oasis VLLM接口测试完成")
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保已安装Camel和Oasis: pip install camel-ai oasis")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

async def test_multi_model_integration():
    """测试多模型训练系统集成"""
    print("\n🧪 测试多模型训练系统集成...")
    print("=" * 50)
    
    try:
        # 导入多模型训练系统
        from demo.multi_model_single_env_simple import (
            MultiModelEnvironment, ModelConfig, ModelRole, TrainingMode, VLLMClient
        )
        
        print("✅ 成功导入多模型训练系统")
        
        # 测试VLLM客户端
        print("\n🔧 测试VLLM客户端...")
        vllm_client = VLLMClient("http://localhost:8001/v1", "qwen-2")
        
        test_prompt = "请简要说明多模型训练的优势。"
        print(f"📝 测试提示词: {test_prompt}")
        
        response = await vllm_client.generate(test_prompt, max_tokens=50)
        print(f"🤖 VLLM响应: {response}")
        
        # 测试环境初始化
        print("\n🔧 测试环境初始化...")
        env = MultiModelEnvironment(
            vllm_url="http://localhost:8001/v1",
            training_mode=TrainingMode.COOPERATIVE,
            max_models=3
        )
        
        print(f"✅ 环境初始化成功，VLLM可用性: {env.vllm_available}")
        
        # 测试模型添加
        print("\n🔧 测试模型添加...")
        config = ModelConfig(
            model_id="camel_test_model_001",
            model_name="qwen-2",
            role=ModelRole.LEADER,
            lora_rank=8,
            team_id="camel_test_team"
        )
        
        success = env.add_model(config)
        print(f"✅ 模型添加: {'成功' if success else '失败'}")
        
        print("\n✅ 多模型训练系统集成测试完成")
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """主函数"""
    print("🚀 Camel和Oasis VLLM接口测试")
    print("=" * 70)
    
    # 测试Camel和Oasis VLLM接口
    await test_camel_oasis_vllm()
    
    # 测试多模型训练系统集成
    await test_multi_model_integration()
    
    print("\n🎉 所有测试完成！")

if __name__ == "__main__":
    asyncio.run(main())
