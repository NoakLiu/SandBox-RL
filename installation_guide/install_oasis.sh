#!/bin/bash

echo "🏝️ Installing OASIS (Open Agent Social Interaction Simulations) for SandGraph"
echo "=" * 60

# 检查Python版本
python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+' | head -1)
echo "Python version: $python_version"

# 检查pip
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 not found. Please install pip first."
    exit 1
fi

echo "📦 Installing OASIS dependencies..."

# 安装OASIS
echo "Installing camel-oasis..."
pip3 install camel-oasis

# 安装其他依赖
echo "Installing additional dependencies..."
pip3 install camel-ai
pip3 install openai
pip3 install sqlite3

# 验证安装
echo "🔍 Verifying installation..."
python3 -c "
try:
    import oasis
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType, ModelType
    print('✅ OASIS installation successful!')
except ImportError as e:
    print(f'❌ OASIS installation failed: {e}')
    print('Please check the installation and try again.')
"

echo "📚 OASIS Documentation:"
echo "- GitHub: https://github.com/camel-ai/camel"
echo "- OASIS: https://github.com/camel-ai/camel/tree/main/camel/oasis"
echo "- Examples: https://github.com/camel-ai/camel/tree/main/examples/oasis"

echo "🚀 You can now run the OASIS API demo:"
echo "python demo/oasis_api_demo.py --steps 5" 