#!/bin/bash
# SandGraph + MCP 快速安装脚本
#
# 使用 Conda 安装步骤：
# 1. 创建新的 conda 环境：
#    conda create -n sandgraph python=3.11
# 2. 激活环境：
#    conda activate sandgraph
# 3. 运行此脚本：
#    ./quick_install.sh

# 检查 Python 版本
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if (( $(echo "$python_version < 3.8" | python3 -c "import sys; print(float(sys.stdin.read()) < 3.8)") )); then
    echo "错误: 需要 Python 3.8 或更高版本，当前版本: $python_version"
    exit 1
fi

echo "🚀 开始安装 SandGraph + 官方MCP SDK..."

# 1. 安装基础依赖
echo "📦 安装基础依赖..."
pip install numpy pandas scipy networkx matplotlib || { echo "错误: 基础依赖安装失败"; exit 1; }

# 2. 安装官方MCP SDK
echo "📦 安装官方MCP SDK..."
pip install 'mcp[cli]' || { echo "错误: MCP SDK 安装失败"; exit 1; }

# 3. 安装交易相关依赖
echo "📦 安装交易相关依赖..."
pip install backtrader==1.9.76.123 mplfinance==0.12.10b0 yfinance==0.2.36 alpaca-trade-api==3.0.2 || { echo "错误: 交易依赖安装失败"; exit 1; }

# 4. 安装其他依赖
echo "📦 安装其他依赖..."
pip install anthropic==0.3.0 colorama==0.4.6 || { echo "错误: 其他依赖安装失败"; exit 1; }

# 4.1 安装 gymnasium (替代 gym)
echo "📦 安装 gymnasium..."
pip install gymnasium || { echo "错误: gymnasium 安装失败"; exit 1; }

# 4.2 安装 trading-gym
pip install trading-gym==0.1.8 || { echo "错误: trading-gym 安装失败"; exit 1; }

# 5. 安装SandGraph (开发版本)
echo "📦 安装SandGraph..."
pip install -e . || { echo "错误: SandGraph 安装失败"; exit 1; }

# 6. 安装 PyTorch 和相关依赖
echo "📦 安装 PyTorch 和相关依赖..."
pip install torch transformers accelerate tiktoken einops transformers_stream_generator || { echo "错误: PyTorch 和相关依赖安装失败"; exit 1; }

echo ""
echo "✅ 安装完成！"
echo ""
echo "🧪 运行以下命令验证安装："
echo "python -c \"from mcp.server.fastmcp import FastMCP; print('MCP SDK 安装成功')\""
echo "python -c \"from sandgraph import check_mcp_availability; print(check_mcp_availability())\""
echo ""
echo "🎯 下一步："
echo "1. 运行演示: python demo/sandbox_optimization.py"
echo "2. 启动MCP服务器: python demo/mcp_server_example.py"
echo "3. 查看文档: cat README.md" 