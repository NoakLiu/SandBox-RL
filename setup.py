"""
Core SRL Package Installation Configuration
"""

from setuptools import setup, find_packages

# Read README.md content
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="core-srl",
    version="2.0.0",
    author="Core SRL Team",
    author_email="team@core-srl.ai",
    maintainer="Core SRL Team",
    maintainer_email="team@core-srl.ai",
    description="Advanced Multi-Model Reinforcement Learning Framework for Modern LLMs with Cooperative-Competitive Training",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/core-srl",
    project_urls={
        "Bug Tracker": "https://github.com/your-repo/core-srl/issues",
        "Documentation": "https://core-srl.readthedocs.io/",
        "Source Code": "https://github.com/your-repo/core-srl",
        "Examples": "https://github.com/your-repo/core-srl/tree/main/examples",
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
        # Core ML/RL dependencies
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        
        # Data processing
        "pandas>=1.3.0",
        "pyyaml>=6.0",
        "requests>=2.25.0",
        
        # Visualization and monitoring
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "networkx>=2.6.0",
        
        # System utilities
        "psutil>=5.8.0",
        "tqdm>=4.60.0",
        "aiofiles>=0.7.0",
        "typing-extensions>=4.0.0",
        "rich>=12.0.0",
        "colorama>=0.4.4",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=2.10.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.15.0",
            "isort>=5.0",
        ],
        "optimization": [
            "verl>=0.1.0",
            "areal>=0.1.0",
        ],
        "api": [
            "openai>=1.0.0",
            "anthropic>=0.3.0",
        ],
        "visualization": [
            "dash>=2.10.0",
            "kaleido>=0.2.1",
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
            "ipywidgets>=7.6.0",
        ],
        "distributed": [
            "ray>=2.0.0",
            "vllm>=0.2.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.17.0",
        ],
        "full": [
            # All optional dependencies
            "verl>=0.1.0",
            "areal>=0.1.0", 
            "openai>=1.0.0",
            "anthropic>=0.3.0",
            "dash>=2.10.0",
            "kaleido>=0.2.1",
            "ray>=2.0.0",
            "vllm>=0.2.0",
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "core-srl=core_srl.cli:main",
            "srl-train=core_srl.cli:train",
            "srl-monitor=core_srl.cli:monitor",
        ],
    },
    include_package_data=True,
    package_data={
        "core_srl": ["*.txt", "*.md", "*.json", "config/*.yaml", "templates/*.json"],
    },
    keywords=[
        "multi-model", "reinforcement-learning", "llm", "qwen3", "gpt4", "claude", 
        "cooperative", "competitive", "training", "modern-llms", "weight-updates",
        "verl", "areal", "kvcache", "distributed", "ai", "machine-learning"
    ],
    zip_safe=False,
)