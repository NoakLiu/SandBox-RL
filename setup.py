"""
SandGraph 包安装配置
"""

from setuptools import setup, find_packages

# 读取README.md内容
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sandgraph",
    version="0.2.0",
    author="SandGraph Team",
    author_email="team@sandgraph.ai",
    description="基于官方MCP协议的多智能体执行框架，集成InternBootcamp推理训练沙盒",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sandgraph/sandgraph",
    project_urls={
        "Bug Tracker": "https://github.com/sandgraph/sandgraph/issues",
        "Documentation": "https://sandgraph.readthedocs.io/",
        "Source Code": "https://github.com/sandgraph/sandgraph",
        "MCP Protocol": "https://modelcontextprotocol.io/",
        "InternBootcamp": "https://github.com/InternLM/InternBootcamp",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "typing-extensions>=4.0.0",
        "dataclasses;python_version<'3.7'",
        "mcp[cli]>=1.0.0",  # 官方MCP SDK
        "rich>=12.0.0",
        "colorama>=0.4.4",
        "anthropic>=0.3.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "networkx>=2.6.0",
        "matplotlib>=3.4.0",
        "gym>=0.21.0",
        "trading-gym>=0.1.0",
        "backtrader>=1.9.76.123",  # 添加 Backtrader 依赖
        "yfinance>=0.1.70",  # 用于获取Yahoo Finance数据
        "alpaca-trade-api>=2.0.0",  # 用于Alpaca交易API
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.15.0",
            "isort>=5.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.17.0",
        ],
        "examples": [
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
            "ipywidgets>=7.6.0",
        ],
        "mcp-servers": [
            "fastapi>=0.100.0",
            "uvicorn>=0.20.0",
            "websockets>=11.0.0",
            "httpx>=0.24.0",
        ],
        "internbootcamp": [
            "numpy>=1.20.0",
            "matplotlib>=3.5.0",
            "pandas>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sandgraph=sandgraph.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "sandgraph": ["*.txt", "*.md", "*.json"],
    },
    keywords=[
        "llm", "ai", "mcp", "sandbox", "workflow", "multi-agent", 
        "reasoning", "internbootcamp", "anthropic", "claude"
    ],
    zip_safe=False,
) 