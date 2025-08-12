#!/bin/bash

# 多模型训练系统启动脚本 - 100次运行版本
# 适用于服务器环境，运行100次并计算平均值

echo "🚀 启动多模型训练系统 - 100次运行版本"
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
mkdir -p training_outputs checkpoints logs temp batch_results

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

# 运行参数
TOTAL_RUNS=100
SUCCESSFUL_RUNS=0
FAILED_RUNS=0
START_TIME=$(date +%s)

echo ""
echo "🔄 开始100次训练运行..."
echo "=================================================="

# 创建结果汇总文件
SUMMARY_FILE="batch_results/training_summary_$(date +%Y%m%d_%H%M%S).json"
echo "📊 结果汇总文件: $SUMMARY_FILE"

# 初始化汇总数据
cat > "$SUMMARY_FILE" << EOF
{
    "batch_info": {
        "total_runs": $TOTAL_RUNS,
        "start_time": "$(date -d @$START_TIME -Iseconds)",
        "successful_runs": 0,
        "failed_runs": 0
    },
    "runs": [],
    "averages": {}
}
EOF

# 运行100次训练
for ((i=1; i<=TOTAL_RUNS; i++)); do
    echo ""
    echo "🔄 运行 $i/$TOTAL_RUNS"
    echo "----------------------------------------"
    
    RUN_START_TIME=$(date +%s)
    
    # 运行训练
    python demo/run_training_server.py > "logs/run_${i}.log" 2>&1
    RUN_EXIT_CODE=$?
    
    RUN_END_TIME=$(date +%s)
    RUN_DURATION=$((RUN_END_TIME - RUN_START_TIME))
    
    if [ $RUN_EXIT_CODE -eq 0 ]; then
        SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
        echo "✅ 运行 $i 成功 (耗时: ${RUN_DURATION}秒)"
        
        # 收集运行结果
        if [ -f "multi_model_training_simple_results.json" ]; then
            # 备份结果文件
            cp "multi_model_training_simple_results.json" "batch_results/results_run_${i}.json"
            
            # 解析结果数据
            python -c "
import json
import sys

try:
    with open('multi_model_training_simple_results.json', 'r') as f:
        data = json.load(f)
    
    # 计算平均指标
    if isinstance(data, list) and len(data) > 0:
        avg_accuracy = sum(item.get('accuracy', 0) for item in data) / len(data)
        avg_efficiency = sum(item.get('efficiency', 0) for item in data) / len(data)
        avg_reward = sum(item.get('reward_earned', 0) for item in data) / len(data)
        total_tasks = sum(item.get('total_tasks', 0) for item in data)
        
        run_summary = {
            'run_number': $i,
            'status': 'success',
            'duration_seconds': $RUN_DURATION,
            'timestamp': '$(date -Iseconds)',
            'metrics': {
                'avg_accuracy': round(avg_accuracy, 4),
                'avg_efficiency': round(avg_efficiency, 4),
                'avg_reward': round(avg_reward, 2),
                'total_tasks': total_tasks,
                'model_count': len(data)
            }
        }
        
        print(json.dumps(run_summary))
    else:
        print(json.dumps({
            'run_number': $i,
            'status': 'success',
            'duration_seconds': $RUN_DURATION,
            'timestamp': '$(date -Iseconds)',
            'metrics': {'error': 'No valid data found'}
        }))
        
except Exception as e:
    print(json.dumps({
        'run_number': $i,
        'status': 'success',
        'duration_seconds': $RUN_DURATION,
        'timestamp': '$(date -Iseconds)',
        'metrics': {'error': str(e)}
    }))
" > "batch_results/run_${i}_summary.json"
            
            echo "   📊 平均准确率: $(jq -r '.metrics.avg_accuracy' batch_results/run_${i}_summary.json)"
            echo "   📊 平均效率: $(jq -r '.metrics.avg_efficiency' batch_results/run_${i}_summary.json)"
            echo "   📊 平均奖励: $(jq -r '.metrics.avg_reward' batch_results/run_${i}_summary.json)"
        fi
        
    else
        FAILED_RUNS=$((FAILED_RUNS + 1))
        echo "❌ 运行 $i 失败 (耗时: ${RUN_DURATION}秒)"
        
        # 记录失败信息
        cat > "batch_results/run_${i}_summary.json" << EOF
{
    "run_number": $i,
    "status": "failed",
    "duration_seconds": $RUN_DURATION,
    "timestamp": "$(date -Iseconds)",
    "error": "Exit code: $RUN_EXIT_CODE"
}
EOF
    fi
    
    # 更新汇总文件
    python -c "
import json

# 读取当前汇总
with open('$SUMMARY_FILE', 'r') as f:
    summary = json.load(f)

# 更新计数
summary['batch_info']['successful_runs'] = $SUCCESSFUL_RUNS
summary['batch_info']['failed_runs'] = $FAILED_RUNS

# 添加运行结果
try:
    with open('batch_results/run_${i}_summary.json', 'r') as f:
        run_data = json.load(f)
    summary['runs'].append(run_data)
except:
    summary['runs'].append({
        'run_number': $i,
        'status': 'error',
        'timestamp': '$(date -Iseconds)'
    })

# 保存更新后的汇总
with open('$SUMMARY_FILE', 'w') as f:
    json.dump(summary, f, indent=2)
"
    
    # 显示进度
    PROGRESS=$((i * 100 / TOTAL_RUNS))
    echo "📈 进度: $PROGRESS% ($SUCCESSFUL_RUNS 成功, $FAILED_RUNS 失败)"
    
    # 每10次运行后显示中间统计
    if [ $((i % 10)) -eq 0 ]; then
        echo ""
        echo "📊 中间统计 (前 $i 次运行):"
        echo "   成功率: $((SUCCESSFUL_RUNS * 100 / i))%"
        echo "   平均运行时间: $(( (RUN_END_TIME - START_TIME) / i ))秒"
        echo ""
    fi
    
    # 清理临时文件（保留日志）
    rm -f "multi_model_training_simple_results.json"
    rm -rf "temp"/*
    mkdir -p "temp"
done

# 计算最终统计
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo ""
echo "🎉 100次运行完成！"
echo "=================================================="
echo "📊 最终统计:"
echo "   总运行次数: $TOTAL_RUNS"
echo "   成功次数: $SUCCESSFUL_RUNS"
echo "   失败次数: $FAILED_RUNS"
echo "   成功率: $((SUCCESSFUL_RUNS * 100 / TOTAL_RUNS))%"
echo "   总耗时: ${TOTAL_DURATION}秒 ($(($TOTAL_DURATION / 60))分钟)"
echo "   平均每次运行: $((TOTAL_DURATION / TOTAL_RUNS))秒"

# 计算平均值
echo ""
echo "📈 计算平均值..."
python -c "
import json
import numpy as np

# 读取所有成功的运行
successful_runs = []
for i in range(1, $TOTAL_RUNS + 1):
    try:
        with open(f'batch_results/run_{i}_summary.json', 'r') as f:
            run_data = json.load(f)
            if run_data.get('status') == 'success' and 'metrics' in run_data:
                successful_runs.append(run_data['metrics'])
    except:
        continue

if successful_runs:
    # 计算平均值
    avg_accuracy = np.mean([run.get('avg_accuracy', 0) for run in successful_runs])
    avg_efficiency = np.mean([run.get('avg_efficiency', 0) for run in successful_runs])
    avg_reward = np.mean([run.get('avg_reward', 0) for run in successful_runs])
    avg_tasks = np.mean([run.get('total_tasks', 0) for run in successful_runs])
    avg_models = np.mean([run.get('model_count', 0) for run in successful_runs])
    
    # 计算标准差
    std_accuracy = np.std([run.get('avg_accuracy', 0) for run in successful_runs])
    std_efficiency = np.std([run.get('avg_efficiency', 0) for run in successful_runs])
    std_reward = np.std([run.get('avg_reward', 0) for run in successful_runs])
    
    averages = {
        'avg_accuracy': round(avg_accuracy, 4),
        'avg_efficiency': round(avg_efficiency, 4),
        'avg_reward': round(avg_reward, 2),
        'avg_tasks': round(avg_tasks, 1),
        'avg_models': round(avg_models, 1),
        'std_accuracy': round(std_accuracy, 4),
        'std_efficiency': round(std_efficiency, 4),
        'std_reward': round(std_reward, 2),
        'successful_runs_count': len(successful_runs)
    }
    
    print('📊 平均值统计:')
    print(f'   平均准确率: {averages[\"avg_accuracy\"]} ± {averages[\"std_accuracy\"]}')
    print(f'   平均效率: {averages[\"avg_efficiency\"]} ± {averages[\"std_efficiency\"]}')
    print(f'   平均奖励: {averages[\"avg_reward\"]} ± {averages[\"std_reward\"]}')
    print(f'   平均任务数: {averages[\"avg_tasks\"]}')
    print(f'   平均模型数: {averages[\"avg_models\"]}')
    print(f'   有效运行数: {averages[\"successful_runs_count\"]}')
    
    # 保存平均值到汇总文件
    with open('$SUMMARY_FILE', 'r') as f:
        summary = json.load(f)
    
    summary['averages'] = averages
    
    with open('$SUMMARY_FILE', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # 保存平均值到单独文件
    with open('batch_results/averages.json', 'w') as f:
        json.dump(averages, f, indent=2)
        
else:
    print('❌ 没有成功的运行数据')
"

# 生成可视化
echo ""
read -p "是否生成批量运行的可视化图表？(y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🎨 生成批量运行可视化图表..."
    
    # 创建批量可视化脚本
    cat > "batch_results/generate_batch_visualization.py" << 'EOF'
#!/usr/bin/env python3
"""
批量运行结果可视化脚本
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob

def load_batch_data():
    """加载批量运行数据"""
    runs = []
    for i in range(1, 101):  # 100次运行
        try:
            with open(f'batch_results/run_{i}_summary.json', 'r') as f:
                run_data = json.load(f)
                if run_data.get('status') == 'success':
                    runs.append(run_data)
        except:
            continue
    return runs

def create_batch_visualizations():
    """创建批量运行可视化"""
    runs = load_batch_data()
    
    if not runs:
        print("❌ 没有找到有效的运行数据")
        return
    
    # 提取数据
    accuracies = [run['metrics']['avg_accuracy'] for run in runs]
    efficiencies = [run['metrics']['avg_efficiency'] for run in runs]
    rewards = [run['metrics']['avg_reward'] for run in runs]
    durations = [run['duration_seconds'] for run in runs]
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('100次训练运行结果分析', fontsize=16, fontweight='bold')
    
    # 1. 准确率分布
    axes[0, 0].hist(accuracies, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(np.mean(accuracies), color='red', linestyle='--', label=f'平均值: {np.mean(accuracies):.4f}')
    axes[0, 0].set_title('准确率分布')
    axes[0, 0].set_xlabel('准确率')
    axes[0, 0].set_ylabel('频次')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 效率分布
    axes[0, 1].hist(efficiencies, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].axvline(np.mean(efficiencies), color='red', linestyle='--', label=f'平均值: {np.mean(efficiencies):.4f}')
    axes[0, 1].set_title('效率分布')
    axes[0, 1].set_xlabel('效率')
    axes[0, 1].set_ylabel('频次')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 奖励趋势
    axes[1, 0].plot(rewards, marker='o', alpha=0.6, color='orange')
    axes[1, 0].axhline(np.mean(rewards), color='red', linestyle='--', label=f'平均值: {np.mean(rewards):.2f}')
    axes[1, 0].set_title('奖励趋势')
    axes[1, 0].set_xlabel('运行次数')
    axes[1, 0].set_ylabel('奖励')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 运行时间分布
    axes[1, 1].hist(durations, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1, 1].axvline(np.mean(durations), color='red', linestyle='--', label=f'平均值: {np.mean(durations):.1f}秒')
    axes[1, 1].set_title('运行时间分布')
    axes[1, 1].set_xlabel('运行时间(秒)')
    axes[1, 1].set_ylabel('频次')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('batch_results/batch_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ 批量运行可视化图表已生成: batch_results/batch_analysis.png")

if __name__ == "__main__":
    create_batch_visualizations()
EOF
    
    python batch_results/generate_batch_visualization.py
    
    if [ $? -eq 0 ]; then
        echo "✅ 批量运行可视化图表已生成到 batch_results/"
    else
        echo "❌ 批量可视化生成失败"
    fi
fi

echo ""
echo "📁 结果文件位置:"
echo "   汇总报告: $SUMMARY_FILE"
echo "   平均值: batch_results/averages.json"
echo "   各次运行结果: batch_results/results_run_*.json"
echo "   运行日志: logs/run_*.log"
echo "   可视化图表: batch_results/batch_analysis.png"

echo ""
echo "🏁 100次运行脚本执行完成"
