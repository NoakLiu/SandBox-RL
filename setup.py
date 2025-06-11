"""
SandGraph 包安装配置
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sandgraph",
    version="0.1.0",
    author="SandGraph Team",
    author_email="dong.liu.dl2367@yale.edu",
    description="基于沙盒任务模块和图式工作流的多智能体执行框架",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sandgraph",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        # 核心依赖
        "numpy>=1.21.0",
        "typing-extensions>=4.0.0",
        
        # 可选依赖（用于可视化和扩展功能）
        "networkx>=2.6",
        "matplotlib>=3.5.0",
        "pydantic>=1.8.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.12",
            "black>=21.0",
            "isort>=5.0",
            "mypy>=0.910",
            "flake8>=3.9",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
        "viz": [
            "graphviz>=0.20",
            "plotly>=5.0",
            "dash>=2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sandgraph=sandgraph.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "sandgraph": [
            "examples/*.yaml",
            "configs/*.yaml",
        ],
    },
) 