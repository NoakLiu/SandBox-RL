#!/bin/bash

# 多模型训练系统启动脚本
# 适用于服务器环境

echo "🚀 启动多模型训练系统..."
echo "=================================================="

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "❌ Python未安装或不在PATH中"
    exit 1
fi

# 检查项目目录
if [ ! -f "demo/multi_model_single_env_simple.py" ]; then
    echo "❌ 未找到训练脚本，请确保在SandGraph项目根目录运行"
    exit 1
fi

# 创建必要的目录
echo "📁 创建目录..."
mkdir -p training_outputs checkpoints logs temp

# 设置环境变量
echo "🔧 设置环境变量..."
export PYTHONPATH="${PWD}:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32768
export TMPDIR="./temp"

# 显示系统信息
echo "💾 系统信息:"
echo "   当前目录: $(pwd)"
echo "   Python版本: $(python --version)"
echo "   内存信息: $(free -h | grep Mem | awk '{print $2}')"
echo "   磁盘信息: $(df -h . | tail -1 | awk '{print $4}') 可用"

# 检查GPU
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU信息:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | while read line; do
        echo "   $line"
    done
else
    echo "⚠️ 未检测到NVIDIA GPU"
fi

echo ""
echo "🚀 开始训练..."
echo "=================================================="

# 运行训练
python demo/run_training_server.py

# 检查运行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 训练完成！"
    echo "📁 输出目录: ./training_outputs"
    echo "💾 检查点目录: ./checkpoints"
    echo "📋 日志目录: ./logs"
    
    # 显示生成的文件
    echo ""
    echo "📊 生成的文件:"
    if [ -f "multi_model_training_simple_results.json" ]; then
        echo "   ✅ 训练结果: multi_model_training_simple_results.json"
    fi
    if [ -f "training_outputs/run_report.json" ]; then
        echo "   ✅ 运行报告: training_outputs/run_report.json"
    fi
    
    # 询问是否生成可视化
    echo ""
    read -p "是否生成可视化图表？(y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🎨 生成可视化图表..."
        python demo/multi_model_visualization.py
        if [ $? -eq 0 ]; then
            echo "✅ 可视化图表已生成到 ./visualization_outputs/"
        else
            echo "❌ 可视化生成失败"
        fi
    fi
    
else
    echo ""
    echo "❌ 训练失败！"
    echo "请检查日志文件: ./logs/training_runner.log"
    exit 1
fi

echo ""
echo "🏁 脚本执行完成"
