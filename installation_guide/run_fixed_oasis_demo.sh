#!/bin/bash

# Fixed OASIS Social Network Demo Runner
# =====================================
# This script runs the fixed OASIS demo to resolve the network density and RL training issues

echo "🔧 Fixed OASIS Social Network Demo"
echo "=================================="
echo "Fixes applied:"
echo "- Network density calculation (was showing 0.000)"
echo "- User connections initialization"
echo "- RL training steps tracking (was showing 0)"
echo "- Monitoring system integration"
echo ""

# Check if we're in the right directory
if [ ! -f "demo/enhanced_oasis_social_demo_fixed.py" ]; then
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
python -c "import wandb, torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Warning: Some optional packages not found. Installing..."
    pip install wandb torch
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
echo "   - Steps: 10"
echo "   - Initial Users: 50"
echo "   - Initial Posts: 20"
echo "   - RL Algorithm: PPO"
echo "   - Monitoring: WanDB + TensorBoard"
echo ""

# Ask user for WanDB preference
read -p "🤔 Do you want to use WanDB monitoring? (y/n): " use_wandb

if [[ $use_wandb =~ ^[Yy]$ ]]; then
    echo "✅ Running with WanDB monitoring..."
    WANDB_FLAGS="--enable-wandb --wandb-project sandgraph-fixed-oasis"
else
    echo "✅ Running without WanDB monitoring..."
    WANDB_FLAGS=""
fi

echo ""
echo "🚀 Starting Fixed OASIS Demo..."
echo ""

# Run the fixed demo
python demo/enhanced_oasis_social_demo_fixed.py \
    --steps 10 \
    --enable-tensorboard \
    $WANDB_FLAGS

# Check if demo completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Fixed OASIS Demo completed successfully!"
    echo ""
    echo "📊 Expected Improvements:"
    echo "   - Network Density: Should be > 0.000 (was 0.000)"
    echo "   - RL Training Steps: Should be > 0 (was 0)"
    echo "   - Total Connections: Should show actual connections"
    echo "   - Better decision making and monitoring"
    echo ""
    
    if [[ $use_wandb =~ ^[Yy]$ ]]; then
        echo "📈 WanDB Dashboard:"
        echo "   - Visit https://wandb.ai to view detailed metrics"
        echo "   - Project: sandgraph-fixed-oasis"
        echo "   - Real-time monitoring data and visualizations"
    fi
    
    echo "📁 Local Files:"
    echo "   - Logs: logs/fixed_oasis_*.json"
    echo "   - TensorBoard: logs/fixed_oasis/ directory"
    echo "   - Alerts: logs/fixed_oasis_alerts.json"
    echo ""
    echo "🔍 Key Metrics to Verify:"
    echo "   - Network Density > 0.000"
    echo "   - RL Training Steps > 0"
    echo "   - Total Connections > 0"
    echo "   - Decision quality and reasoning"
    echo "   - Monitoring system functionality"
    
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
echo "   - demo/enhanced_oasis_social_demo_fixed.py: Fixed demo source code"
echo "   - demo/oasis_social_demo.py: Original OASIS demo"
echo "   - docs/: Detailed guides and tutorials"
echo "" 