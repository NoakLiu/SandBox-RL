"""
SandGraph 包安装配置
"""

from setuptools import setup, find_packages

setup(
    name="sandgraph",
    version="0.2.0",
    description="基于官方MCP协议的多智能体执行框架，用于沙盒任务模块和图工作流",
    long_description="""
SandGraph 是一个基于官方Model Context Protocol (MCP)的多智能体执行框架。

核心特性：
- 基于官方Anthropic MCP Python SDK
- 沙盒环境：遵循Game24bootcamp模式的任务环境
- 工作流图：支持复杂LLM-沙盒交互的DAG执行引擎
- 标准化通信：使用官方MCP协议进行LLM-沙盒通信
- 六种使用场景：从单一沙盒执行到复杂多阶段工作流

支持的使用场景：
1. UC1: 单沙盒执行 - 一个LLM调用一个沙盒
2. UC2: 并行映射归约 - 多个沙盒并行处理任务并聚合结果
3. UC3: 多智能体协作 - 多个LLM通过MCP协议合作
4. UC4: LLM辩论模式 - LLM之间的结构化辩论与判断
5. UC5: 复杂管道 - 多阶段工作流，涉及不同沙盒和LLM
6. UC6: 迭代交互 - 多轮LLM-沙盒对话与状态管理

集成官方MCP生态系统，支持连接到各种MCP服务器和工具。
    """.strip(),
    author="SandGraph Team",
    author_email="contact@sandgraph.dev",
    url="https://github.com/sandgraph/sandgraph",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.8",
    install_requires=[
        # 官方MCP SDK依赖
        "mcp[cli]>=1.0.0",  # 包含CLI工具的官方MCP SDK
        
        # 原有核心依赖
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "networkx>=2.8.0",
        
        # 用于沙盒实现的依赖
        "pandas>=1.3.0",
        "requests>=2.25.0",
        "typing-extensions>=4.0.0",
        
        # 异步和并发支持
        "asyncio-utils>=0.1.0",
        
        # JSON处理和验证
        "jsonschema>=4.0.0",
        
        # 日志和调试
        "rich>=12.0.0",
        "colorama>=0.4.4",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=1.0.0",
            "flake8>=5.0.0",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.17.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ],
        "web": [
            "fastapi>=0.95.0",
            "uvicorn>=0.20.0",
            "websockets>=10.0",
        ],
        "enterprise": [
            # 企业级功能
            "redis>=4.0.0",
            "sqlalchemy>=1.4.0",
            "alembic>=1.8.0",
            "prometheus-client>=0.14.0",
        ],
        "mcp-servers": [
            # 常用MCP服务器（如果有官方包的话）
            # 这些可能需要根据实际可用的包来调整
            "mcp-server-github",
            "mcp-server-filesystem", 
        ]
    },
    entry_points={
        "console_scripts": [
            "sandgraph=sandgraph.cli:main",
            "sandgraph-demo=sandgraph.examples:run_demo",
            "sandgraph-server=sandgraph.mcp_server:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="ai, llm, mcp, sandbox, workflow, multi-agent, anthropic",
    project_urls={
        "Documentation": "https://sandgraph.readthedocs.io/",
        "Source": "https://github.com/sandgraph/sandgraph",
        "Tracker": "https://github.com/sandgraph/sandgraph/issues",
        "MCP Official": "https://modelcontextprotocol.io/",
        "MCP Python SDK": "https://github.com/modelcontextprotocol/python-sdk",
    },
) 