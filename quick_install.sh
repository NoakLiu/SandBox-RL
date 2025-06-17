#!/bin/bash
# SandGraph + MCP 快速安装脚本

# 检查 Python 版本
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if (( $(echo "$python_version < 3.8" | bc -l) )); then
    echo "错误: 需要 Python 3.8 或更高版本，当前版本: $python_version"
    exit 1
fi

echo "🚀 开始安装 SandGraph + 官方MCP SDK..."

# 1. 安装官方MCP SDK
echo "📦 安装官方MCP SDK..."
pip install 'mcp[cli]' || { echo "错误: MCP SDK 安装失败"; exit 1; }

# 2. 安装SandGraph (开发版本，因为我们在本地开发)
echo "📦 安装SandGraph..."
pip install -e . || { echo "错误: SandGraph 安装失败"; exit 1; }

# 3. 安装额外依赖
echo "📦 安装额外依赖..."
pip install numpy==1.24.3 scipy==1.10.1 networkx==3.1 pandas==2.0.3 || { echo "错误: 基础依赖安装失败"; exit 1; }
pip install backtrader==1.9.76.123 mplfinance==0.12.10b0 || { echo "错误: 交易依赖安装失败"; exit 1; }

echo ""
echo "✅ 安装完成！"
echo ""
echo "🧪 运行以下命令验证安装："
echo "python -c \"from mcp.server.fastmcp import FastMCP; print('MCP SDK 安装成功')\""
echo "python -c \"from sandgraph import check_mcp_availability; print(check_mcp_availability())\""
echo ""
echo "🎯 下一步："
echo "1. 运行演示: python demo.py"
echo "2. 启动MCP服务器: python mcp_server_example.py"
echo "3. 查看文档: cat README.md" 