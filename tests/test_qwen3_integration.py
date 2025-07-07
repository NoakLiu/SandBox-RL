#!/usr/bin/env python3
"""
Qwen3 14B Integration Test for SandGraphX
=========================================

This script tests the integration of Qwen3 14B model with SandGraphX framework.
It demonstrates basic usage, performance testing, and various application scenarios.
"""

import sys
import os
import time
import argparse
from typing import List, Dict, Any

# Add the parent directory to the path to import sandgraph modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_basic_integration():
    """Test basic integration of Qwen3 14B with SandGraphX"""
    print("🧪 Testing Qwen3 14B Basic Integration...")
    
    try:
        from sandgraph.core.llm_interface import create_shared_llm_manager
        
        # Create Qwen3 14B model manager
        llm_manager = create_shared_llm_manager(
            model_name="Qwen/Qwen3-14B-Instruct",
            backend="huggingface",
            temperature=0.7,
            max_tokens=1024
        )
        
        print("✅ Qwen3 14B model manager created successfully")
        
        # Register test node
        llm_manager.register_node("test_node", {
            "role": "测试助手",
            "temperature": 0.8,
            "max_length": 512
        })
        
        print("✅ Test node registered successfully")
        
        # Test basic generation
        test_prompt = "你好，请简单介绍一下自己"
        print(f"📝 Testing with prompt: {test_prompt}")
        
        start_time = time.time()
        response = llm_manager.generate_for_node("test_node", test_prompt)
        end_time = time.time()
        
        print(f"⏱️  Response time: {end_time - start_time:.2f} seconds")
        print(f"🤖 Response: {response.text[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic integration test failed: {e}")
        return False

def test_multiple_nodes():
    """Test multiple specialized nodes with Qwen3 14B"""
    print("\n🧪 Testing Multiple Specialized Nodes...")
    
    try:
        from sandgraph.core.llm_interface import create_shared_llm_manager
        
        llm_manager = create_shared_llm_manager(
            model_name="Qwen/Qwen3-14B-Instruct",
            backend="huggingface",
            temperature=0.7
        )
        
        # Register different specialized nodes
        nodes_config = {
            "chat_assistant": {
                "role": "智能对话助手",
                "temperature": 0.8,
                "max_length": 1024
            },
            "code_generator": {
                "role": "代码生成专家",
                "temperature": 0.3,
                "max_length": 2048
            },
            "reasoning_expert": {
                "role": "推理专家",
                "temperature": 0.5,
                "max_length": 1536
            }
        }
        
        for node_name, config in nodes_config.items():
            llm_manager.register_node(node_name, config)
            print(f"✅ Registered {node_name} node")
        
        # Test each node
        test_cases = [
            ("chat_assistant", "请介绍一下人工智能的发展历程"),
            ("code_generator", "请用Python实现一个快速排序算法"),
            ("reasoning_expert", "如果所有A都是B，所有B都是C，那么所有A都是C吗？")
        ]
        
        for node_name, prompt in test_cases:
            print(f"\n📝 Testing {node_name} with: {prompt}")
            start_time = time.time()
            response = llm_manager.generate_for_node(node_name, prompt)
            end_time = time.time()
            
            print(f"⏱️  Response time: {end_time - start_time:.2f} seconds")
            print(f"🤖 Response preview: {response.text[:150]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Multiple nodes test failed: {e}")
        return False

def test_workflow_integration():
    """Test Qwen3 14B integration with SandGraphX workflow"""
    print("\n🧪 Testing Workflow Integration...")
    
    try:
        from sandgraph.core.llm_interface import create_shared_llm_manager
        from sandgraph.core.sg_workflow import SG_Workflow, WorkflowMode, NodeType, EnhancedWorkflowNode, NodeCondition, NodeLimits
        
        # Create LLM manager with Qwen3 14B
        llm_manager = create_shared_llm_manager(
            model_name="Qwen/Qwen3-14B-Instruct",
            backend="huggingface",
            temperature=0.7
        )
        
        # Create workflow
        workflow = SG_Workflow("qwen3_test_workflow", WorkflowMode.TRADITIONAL, llm_manager)
        
        # Create a simple sandbox environment
        class TestSandbox:
            def __init__(self):
                self.state = {"user_count": 0, "engagement": 0.0}
            
            def execute(self, action):
                if action == "CREATE_POST":
                    self.state["user_count"] += 1
                    return self.state, 1.0, False
                elif action == "LIKE_POST":
                    self.state["engagement"] += 0.1
                    return self.state, 0.5, False
                return self.state, 0.2, False
            
            def get_state(self):
                return self.state
        
        # Add nodes to workflow
        env_node = EnhancedWorkflowNode(
            "environment", 
            NodeType.SANDBOX, 
            sandbox=TestSandbox(),
            condition=NodeCondition(),
            limits=NodeLimits()
        )
        
        decision_node = EnhancedWorkflowNode(
            "decision", 
            NodeType.LLM, 
            llm_func=lambda prompt: llm_manager.generate_for_node("decision", prompt).text,
            condition=NodeCondition(),
            limits=NodeLimits()
        )
        
        workflow.add_node(env_node)
        workflow.add_node(decision_node)
        
        # Connect nodes
        workflow.add_edge("environment", "decision")
        workflow.add_edge("decision", "environment")
        
        print("✅ Workflow created with Qwen3 14B")
        
        # Test workflow execution
        print("🚀 Executing workflow...")
        start_time = time.time()
        result = workflow.execute_full_workflow()
        end_time = time.time()
        
        print(f"⏱️  Workflow execution time: {end_time - start_time:.2f} seconds")
        print(f"📊 Workflow result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Workflow integration test failed: {e}")
        return False

def test_performance_benchmark():
    """Test performance benchmark with Qwen3 14B"""
    print("\n🧪 Testing Performance Benchmark...")
    
    try:
        from sandgraph.core.llm_interface import create_shared_llm_manager
        import psutil
        
        llm_manager = create_shared_llm_manager(
            model_name="Qwen/Qwen3-14B-Instruct",
            backend="huggingface",
            temperature=0.7
        )
        
        llm_manager.register_node("benchmark", {
            "role": "性能测试助手",
            "temperature": 0.5,
            "max_length": 512
        })
        
        # Test prompts
        test_prompts = [
            "请简单介绍一下自己",
            "什么是机器学习？",
            "Python和JavaScript有什么区别？",
            "如何提高编程技能？",
            "解释一下深度学习的基本概念"
        ]
        
        total_time = 0
        total_memory = 0
        successful_requests = 0
        
        print("📊 Running performance benchmark...")
        
        for i, prompt in enumerate(test_prompts, 1):
            try:
                start_time = time.time()
                start_memory = psutil.virtual_memory().used
                
                response = llm_manager.generate_for_node("benchmark", prompt)
                
                end_time = time.time()
                end_memory = psutil.virtual_memory().used
                
                request_time = end_time - start_time
                memory_used = (end_memory - start_memory) / 1024 / 1024  # MB
                
                total_time += request_time
                total_memory += memory_used
                successful_requests += 1
                
                print(f"  Request {i}: {request_time:.2f}s, {memory_used:.1f}MB")
                
            except Exception as e:
                print(f"  Request {i} failed: {e}")
        
        if successful_requests > 0:
            avg_time = total_time / successful_requests
            avg_memory = total_memory / successful_requests
            
            print(f"\n📈 Performance Results:")
            print(f"  Average response time: {avg_time:.2f} seconds")
            print(f"  Average memory usage: {avg_memory:.1f} MB")
            print(f"  Success rate: {successful_requests}/{len(test_prompts)} ({successful_requests/len(test_prompts)*100:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        return False

def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description="Test Qwen3 14B integration with SandGraphX")
    parser.add_argument("--test", choices=["basic", "nodes", "workflow", "performance", "all"], 
                       default="all", help="Test to run")
    
    args = parser.parse_args()
    
    print("🚀 Qwen3 14B Integration Test for SandGraphX")
    print("=" * 60)
    
    tests = {
        "basic": test_basic_integration,
        "nodes": test_multiple_nodes,
        "workflow": test_workflow_integration,
        "performance": test_performance_benchmark
    }
    
    if args.test == "all":
        test_functions = list(tests.values())
    else:
        test_functions = [tests[args.test]]
    
    results = []
    for test_func in test_functions:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Qwen3 14B integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 