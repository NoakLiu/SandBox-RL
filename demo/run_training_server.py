#!/usr/bin/env python3
"""
服务器环境多模型训练运行脚本
配置大内存和本地存储，保存运行信息和检查点
"""

import os
import sys
import json
import time
import logging
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
import gc

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/training_runner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_environment():
    """设置运行环境"""
    logger.info("🔧 设置运行环境...")
    
    # 创建必要的目录
    directories = [
        "./training_outputs",
        "./checkpoints", 
        "./logs",
        "./temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 创建目录: {directory}")
    
    # 设置环境变量
    env_vars = {
        "PYTHONPATH": f"{os.getcwd()}:{os.environ.get('PYTHONPATH', '')}",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",  # 使用所有GPU
        "OMP_NUM_THREADS": "16",  # 设置线程数
        "MKL_NUM_THREADS": "16",
        "NUMEXPR_NUM_THREADS": "16",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:32768",  # 32GB内存限制
        "TMPDIR": "./temp"  # 使用本地临时目录
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        logger.info(f"   设置环境变量: {key}={value}")
    
    # 清理内存
    gc.collect()
    
    logger.info("✅ 环境设置完成")

def check_system_resources():
    """检查系统资源"""
    logger.info("💾 检查系统资源...")
    
    try:
        import psutil
        
        # 内存信息
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        
        logger.info(f"   内存: {available_gb:.1f}GB 可用 / {memory_gb:.1f}GB 总计")
        
        # CPU信息
        cpu_count = psutil.cpu_count()
        logger.info(f"   CPU: {cpu_count} 核心")
        
        # 磁盘信息
        disk = psutil.disk_usage('.')
        disk_gb = disk.total / (1024**3)
        free_gb = disk.free / (1024**3)
        
        logger.info(f"   磁盘: {free_gb:.1f}GB 可用 / {disk_gb:.1f}GB 总计")
        
        # GPU信息
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                logger.info(f"   GPU: {gpu_count} 个GPU可用")
                
                for i in range(gpu_count):
                    device = torch.cuda.get_device_properties(i)
                    memory_gb = device.total_memory / (1024**3)
                    logger.info(f"     GPU {i}: {device.name}, {memory_gb:.1f}GB")
            else:
                logger.warning("   GPU: 不可用")
        except ImportError:
            logger.warning("   GPU: PyTorch未安装")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 系统资源检查失败: {e}")
        return False

def run_training():
    """运行训练"""
    logger.info("🚀 启动多模型训练...")
    
    # 设置环境
    setup_environment()
    
    # 检查系统资源
    if not check_system_resources():
        logger.error("❌ 系统资源不足，退出训练")
        return False
    
    # 创建训练脚本
    script_content = '''#!/usr/bin/env python3
"""
多模型训练脚本
"""

import os
import sys
import json
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """主训练函数"""
    logger.info("🚀 开始多模型训练...")
    
    try:
        # 导入训练模块
        from demo.multi_model_single_env_simple import main as training_main
        
        # 运行训练
        import asyncio
        asyncio.run(training_main())
        
        logger.info("✅ 训练完成")
        
    except Exception as e:
        logger.error(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    script_path = "./temp/run_training.py"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    # 设置执行权限
    os.chmod(script_path, 0o755)
    
    logger.info(f"📝 训练脚本已创建: {script_path}")
    
    # 运行训练
    try:
        logger.info("📋 开始执行训练...")
        logger.info("=" * 80)
        
        # 使用subprocess运行训练
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # 实时输出日志
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # 等待进程完成
        return_code = process.wait()
        
        logger.info("=" * 80)
        
        if return_code == 0:
            logger.info("✅ 训练成功完成")
            return True
        else:
            logger.error(f"❌ 训练失败，返回码: {return_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ 训练执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def save_checkpoint():
    """保存检查点"""
    logger.info("💾 保存检查点...")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = f"./checkpoints/checkpoint_{timestamp}"
    
    try:
        # 创建检查点目录
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # 保存运行配置
        config = {
            "timestamp": datetime.now().isoformat(),
            "checkpoint_dir": checkpoint_dir,
            "environment": dict(os.environ)
        }
        
        config_file = f"{checkpoint_dir}/config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # 复制训练结果
        result_files = list(Path(".").glob("*.json"))
        for result_file in result_files:
            if "training" in result_file.name or "model" in result_file.name:
                shutil.copy2(result_file, checkpoint_dir)
        
        # 复制日志文件
        log_files = list(Path("./logs").glob("*.log"))
        for log_file in log_files:
            shutil.copy2(log_file, checkpoint_dir)
        
        logger.info(f"✅ 检查点已保存: {checkpoint_dir}")
        return checkpoint_dir
        
    except Exception as e:
        logger.error(f"❌ 保存检查点失败: {e}")
        return None

def cleanup():
    """清理临时文件"""
    logger.info("🧹 清理临时文件...")
    
    try:
        # 清理临时脚本
        temp_script = "./temp/run_training.py"
        if os.path.exists(temp_script):
            os.remove(temp_script)
            logger.info(f"   删除临时脚本: {temp_script}")
        
        # 清理临时目录
        temp_dir = "./temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"   清理临时目录: {temp_dir}")
        
        logger.info("✅ 清理完成")
        
    except Exception as e:
        logger.error(f"❌ 清理失败: {e}")

def main():
    """主函数"""
    print("🚀 服务器环境多模型训练运行器")
    print("=" * 60)
    
    start_time = datetime.now()
    
    try:
        # 运行训练
        success = run_training()
        
        if success:
            print("\n🎉 训练成功完成！")
            
            # 保存检查点
            checkpoint_dir = save_checkpoint()
            if checkpoint_dir:
                print(f"💾 检查点已保存: {checkpoint_dir}")
            
            # 显示输出目录
            print(f"📁 输出目录: ./training_outputs")
            print(f"📋 日志目录: ./logs")
            
        else:
            print("\n❌ 训练失败！")
            
            # 即使失败也保存检查点
            checkpoint_dir = save_checkpoint()
            if checkpoint_dir:
                print(f"💾 失败检查点已保存: {checkpoint_dir}")
            
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
        
        # 保存中断检查点
        checkpoint_dir = save_checkpoint()
        if checkpoint_dir:
            print(f"💾 中断检查点已保存: {checkpoint_dir}")
            
    except Exception as e:
        print(f"\n❌ 运行器错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        # 清理
        cleanup()
        
        # 显示运行时间
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"\n⏱️ 总运行时间: {duration}")

if __name__ == "__main__":
    main()
