#!/bin/bash

# Comprehensive Misinformation Spread Demo Runner
# =============================================
# This script runs the comprehensive misinformation demo with optimal settings
# to demonstrate SandGraph LLM's ability to beat traditional rules and human users

echo "🚀 Comprehensive Misinformation Spread Demo"
echo "=========================================="
echo "Goal: SandGraph LLM beats rules and human users in misinformation spread"
echo ""

# Check if we're in the right directory
if [ ! -f "demo/comprehensive_misinformation_demo.py" ]; then
    echo "❌ Error: Please run this script from the SandGraphX root directory"
    exit 1
fi

# Check Python environment
echo "🔍 Checking Python environment..."
python --version
if [ $? -ne 0 ]; then
    echo "❌ Error: Python not found. Please activate your conda environment:"
    echo "   conda activate sandgraph"
    exit 1
fi

# Check if required packages are installed
echo "📦 Checking required packages..."
python -c "import wandb, torch, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Warning: Some optional packages not found. Installing..."
    pip install wandb torch numpy
fi

# Set up WanDB (optional)
echo "🔧 Setting up WanDB..."
if command -v wandb &> /dev/null; then
    echo "   WanDB CLI found. You can login with: wandb login"
    echo "   Or run without WanDB by removing --enable-wandb flag"
else
    echo "   WanDB CLI not found. Installing..."
    pip install wandb
fi

echo ""
echo "🎯 Demo Configuration:"
echo "   - Network Size: 1000 users"
echo "   - Simulation Steps: 50"
echo "   - AReaL KV Cache: 5000 entries"
echo "   - LLM Model: Mistral-7B"
echo "   - Monitoring: WanDB + TensorBoard"
echo ""

# Ask user for WanDB preference
read -p "🤔 Do you want to use WanDB monitoring? (y/n): " use_wandb

if [[ $use_wandb =~ ^[Yy]$ ]]; then
    echo "✅ Running with WanDB monitoring..."
    WANDB_FLAGS="--enable-wandb --wandb-project sandgraph-misinformation-competition"
else
    echo "✅ Running without WanDB monitoring..."
    WANDB_FLAGS=""
fi

echo ""
echo "🚀 Starting Comprehensive Misinformation Demo..."
echo ""

# Run the demo with optimal settings
python demo/comprehensive_misinformation_demo.py \
    --steps 50 \
    --num-users 1000 \
    --network-density 0.1 \
    --model-name "mistralai/Mistral-7B-Instruct-v0.2" \
    --kv-cache-size 5000 \
    --max-memory-gb 4.0 \
    --rollout-batch-size 16 \
    --posts-per-agent 3 \
    --enable-tensorboard \
    --log-interval 1.0 \
    $WANDB_FLAGS

# Check if demo completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Comprehensive Misinformation Demo completed successfully!"
    echo ""
    echo "📊 Results Summary:"
    echo "   - Check the console output above for detailed results"
    echo "   - Competition winner and performance metrics displayed"
    echo "   - Network dynamics and belief impact analysis shown"
    echo ""
    
    if [[ $use_wandb =~ ^[Yy]$ ]]; then
        echo "📈 WanDB Dashboard:"
        echo "   - Visit https://wandb.ai to view detailed metrics"
        echo "   - Project: sandgraph-misinformation-competition"
        echo "   - Real-time monitoring data and visualizations"
    fi
    
    echo "📁 Local Files:"
    echo "   - Logs: logs/ directory"
    echo "   - TensorBoard: tensorboard/ directory (if enabled)"
    echo "   - Cache: cache/ directory (AReaL KV cache data)"
    echo ""
    echo "🔍 Key Metrics to Look For:"
    echo "   - Misinformation spread percentage (target: >50%)"
    echo "   - SandGraph LLM vs Rules vs Human performance"
    echo "   - Belief impact and network engagement"
    echo "   - AReaL KV cache hit rate and efficiency"
    echo "   - LLM frozen & adaptive update performance"
    
else
    echo ""
    echo "❌ Demo failed. Please check the error messages above."
    echo ""
    echo "🔧 Troubleshooting:"
    echo "   1. Ensure all dependencies are installed: pip install -r requirements.txt"
    echo "   2. Check if LLM model is available: python -c 'from transformers import AutoModel'"
    echo "   3. Verify WanDB login if using monitoring: wandb login"
    echo "   4. Check available memory (recommended: 8GB+)"
    echo ""
fi

echo ""
echo "📚 For more information:"
echo "   - README.md: Complete documentation"
echo "   - demo/comprehensive_misinformation_demo.py: Demo source code"
echo "   - docs/: Detailed guides and tutorials"
echo "" 