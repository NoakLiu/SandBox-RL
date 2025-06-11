#!/bin/bash
# SandGraph + MCP 快速安装脚本
# 请在虚拟环境中运行此脚本

echo "🚀 开始安装 SandGraph + 官方MCP SDK..."

# 1. 安装官方MCP SDK
echo "📦 安装官方MCP SDK..."
pip install 'mcp[cli]'

# 2. 安装SandGraph (开发版本，因为我们在本地开发)
echo "📦 安装SandGraph..."
pip install -e .

# 3. 可选：安装额外依赖
echo "📦 安装额外依赖..."
pip install numpy scipy networkx pandas

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