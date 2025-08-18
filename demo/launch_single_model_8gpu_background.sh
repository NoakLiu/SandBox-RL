#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="/cpfs04/shared/kilab/hf-hub/Qwen2.5-7B-Instruct"
PORT=8001
LOG_FILE="vllm_single_8gpu.log"

# 设置环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 状态变量
VLLM_PID=""
IS_RUNNING=false

# 清理函数
cleanup() {
    echo -e "\n${YELLOW}🛑 清理资源...${NC}"
    if [ ! -z "$VLLM_PID" ]; then
        kill $VLLM_PID 2>/dev/null || true
    fi
    exit 0
}

# 设置信号处理
trap cleanup SIGINT SIGTERM

# 显示GPU状态
show_gpu_status() {
    echo -e "\n${CYAN}📊 GPU状态:${NC}"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | while IFS=, read -r gpu_id name mem_used mem_total util temp; do
            mem_percent=$((mem_used * 100 / mem_total))
            echo -e "   GPU ${gpu_id}: ${name} | 内存: ${mem_used}MB/${mem_total}MB (${mem_percent}%) | 利用率: ${util}% | 温度: ${temp}°C"
        done
    else
        echo "   nvidia-smi 不可用"
    fi
}

# 检查vLLM状态
check_vllm_status() {
    if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✅${NC}"
        return 0
    else
        echo -e "${RED}❌${NC}"
        return 1
    fi
}

# 检查LoRA状态
check_lora_status() {
    echo -e "\n${PURPLE}🔧 LoRA状态:${NC}"
    if curl -s "http://localhost:${PORT}/v1/lora/adapters" > /dev/null 2>&1; then
        lora_data=$(curl -s "http://localhost:${PORT}/v1/lora/adapters" 2>/dev/null || echo '{"data": []}')
        lora_count=$(echo "$lora_data" | jq '.data | length' 2>/dev/null || echo "0")
        echo -e "   已加载LoRA数量: ${lora_count}"
        
        if [ "$lora_count" -gt 0 ]; then
            echo -e "   已加载的LoRA:"
            echo "$lora_data" | jq -r '.data[].name' 2>/dev/null | while read lora_name; do
                echo -e "     - ${lora_name}"
            done
        fi
    else
        echo -e "   ${YELLOW}⚠️ LoRA API不可用${NC}"
    fi
}

# 显示实时状态
show_realtime_status() {
    clear
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║                单模型+8GPU vLLM 实时状态监控                ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
    
    echo -e "\n${CYAN}📡 服务信息:${NC}"
    echo -e "   端口: ${PORT}"
    echo -e "   模型: $(basename ${MODEL_PATH})"
    echo -e "   张量并行: 8 GPU"
    echo -e "   进程ID: ${VLLM_PID:-未启动}"
    echo -e "   日志文件: ${LOG_FILE}"
    
    echo -e "\n${CYAN}🔍 健康状态:${NC}"
    echo -n "   vLLM服务: "
    if check_vllm_status; then
        IS_RUNNING=true
    else
        IS_RUNNING=false
    fi
    
    if [ "$IS_RUNNING" = true ]; then
        check_lora_status
    fi
    
    show_gpu_status
    
    echo -e "\n${CYAN}📈 系统资源:${NC}"
    if command -v free &> /dev/null; then
        mem_info=$(free -h | grep Mem)
        echo -e "   内存: ${mem_info}"
    fi
    
    if command -v df &> /dev/null; then
        disk_info=$(df -h . | tail -1)
        echo -e "   磁盘: ${disk_info}"
    fi
    
    echo -e "\n${YELLOW}⏰ 更新时间: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo -e "${YELLOW}💡 按 Ctrl+C 退出监控${NC}"
}

# 监控循环
monitor_loop() {
    while true; do
        show_realtime_status
        sleep 5
    done
}

# 主启动流程
main() {
    echo -e "${BLUE}🚀 启动单模型+8GPU分布式vLLM实例 (后台模式)...${NC}"
    
    # 停止现有的vLLM进程
    echo -e "${YELLOW}🛑 停止现有vLLM进程...${NC}"
    pkill -f "vllm serve" || true
    sleep 3
    
    # 清理编译缓存
    echo -e "${YELLOW}🗑️ 清理编译缓存...${NC}"
    rm -rf ~/.cache/vllm/torch_compile_cache || true
    rm -rf ~/.cache/torch/compiled_cache || true
    
    echo -e "\n${CYAN}📡 启动单模型+8GPU vLLM实例...${NC}"
    echo -e "   模型: ${MODEL_PATH}"
    echo -e "   端口: ${PORT}"
    echo -e "   张量并行: 8"
    echo -e "   日志文件: ${LOG_FILE}"
    
    # 启动单模型+8GPU实例（后台运行）
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    vllm serve "${MODEL_PATH}" \
      --host 0.0.0.0 \
      --port "${PORT}" \
      --served-model-name qwen-2 \
      --tensor-parallel-size 8 \
      --dtype bfloat16 \
      --max-model-len 8192 \
      --gpu-memory-utilization 0.4 \
      --max-num-seqs 128 \
      --enable-lora \
      --max-lora-rank 64 \
      --max-loras 16 \
      --disable-compilation \
      > "${LOG_FILE}" 2>&1 &
    
    VLLM_PID=$!
    echo -e "${GREEN}✅ 单模型+8GPU启动完成 (PID: ${VLLM_PID})${NC}"
    
    # 等待启动
    echo -e "\n${YELLOW}⏳ 等待实例启动...${NC}"
    for i in {1..30}; do
        echo -n "."
        if check_vllm_status > /dev/null 2>&1; then
            echo -e "\n${GREEN}✅ vLLM实例已就绪！${NC}"
            break
        fi
        sleep 2
    done
    
    if [ $i -eq 30 ]; then
        echo -e "\n${RED}❌ vLLM启动超时，请检查日志: ${LOG_FILE}${NC}"
        echo -e "${YELLOW}🔍 查看日志: tail -f ${LOG_FILE}${NC}"
        exit 1
    fi
    
    echo -e "\n${GREEN}🎉 单模型+8GPU分布式vLLM启动完成！${NC}"
    echo -e "${CYAN}📊 配置信息:${NC}"
    echo -e "   - 端口: http://localhost:${PORT}"
    echo -e "   - 张量并行: 8 GPU"
    echo -e "   - LoRA支持: 启用"
    echo -e "   - 最大LoRA数: 16"
    echo -e "   - 最大LoRA rank: 64"
    echo -e "\n${CYAN}🔧 关键特性:${NC}"
    echo -e "   - 单进程管理8个GPU"
    echo -e "   - 支持8个LoRA独立热更新"
    echo -e "   - 张量并行加速推理"
    echo -e "   - 禁用编译避免缓存问题"
    echo -e "\n${YELLOW}📝 日志文件: ${LOG_FILE}${NC}"
    echo -e "${YELLOW}🔍 查看日志: tail -f ${LOG_FILE}${NC}"
    
    echo -e "\n${BLUE}🔄 启动实时状态监控...${NC}"
    echo -e "${YELLOW}💡 按 Ctrl+C 退出监控${NC}"
    
    # 启动监控循环
    monitor_loop
}

# 运行主函数
main
