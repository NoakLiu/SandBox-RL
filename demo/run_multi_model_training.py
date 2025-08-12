#!/usr/bin/env python3
"""
多模型训练系统运行脚本
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
from typing import Dict, Any, Optional
import psutil
import gc

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultiModelTrainingRunner:
    """多模型训练系统运行器"""
    
    def __init__(self, 
                 output_dir: str = "./training_outputs",
                 checkpoint_dir: str = "./checkpoints",
                 log_dir: str = "./logs",
                 memory_limit_gb: int = 32,
                 enable_gpu: bool = True):
        
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.memory_limit_gb = memory_limit_gb
        self.enable_gpu = enable_gpu
        
        # 创建目录
        self.ensure_directories()
        
        # 运行配置
        self.run_config = {
            "start_time": datetime.now().isoformat(),
            "memory_limit_gb": memory_limit_gb,
            "enable_gpu": enable_gpu,
            "output_dir": str(self.output_dir),
            "checkpoint_dir": str(self.checkpoint_dir),
            "log_dir": str(self.log_dir)
        }
        
        # 性能监控
        self.performance_metrics = {
            "memory_usage": [],
            "cpu_usage": [],
            "disk_usage": [],
            "checkpoints": [],
            "training_steps": []
        }
    
    def ensure_directories(self):
        """确保所有必要的目录存在"""
        directories = [self.output_dir, self.checkpoint_dir, self.log_dir]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 确保目录存在: {directory}")
    
    def check_system_resources(self) -> Dict[str, Any]:
        """检查系统资源"""
        try:
            # 内存信息
            memory = psutil.virtual_memory()
            memory_info = {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_gb": memory.used / (1024**3),
                "percent": memory.percent
            }
            
            # CPU信息
            cpu_info = {
                "count": psutil.cpu_count(),
                "percent": psutil.cpu_percent(interval=1),
                "freq": psutil.cpu_freq().current if psutil.cpu_freq() else 0
            }
            
            # 磁盘信息
            disk = psutil.disk_usage(self.output_dir)
            disk_info = {
                "total_gb": disk.total / (1024**3),
                "used_gb": disk.used / (1024**3),
                "free_gb": disk.free / (1024**3),
                "percent": disk.percent
            }
            
            # GPU信息（如果可用）
            gpu_info = self.get_gpu_info()
            
            system_info = {
                "memory": memory_info,
                "cpu": cpu_info,
                "disk": disk_info,
                "gpu": gpu_info,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"💾 系统资源检查完成:")
            logger.info(f"   内存: {memory_info['available_gb']:.1f}GB 可用 / {memory_info['total_gb']:.1f}GB 总计")
            logger.info(f"   CPU: {cpu_info['count']} 核心, {cpu_info['percent']:.1f}% 使用率")
            logger.info(f"   磁盘: {disk_info['free_gb']:.1f}GB 可用 / {disk_info['total_gb']:.1f}GB 总计")
            
            return system_info
            
        except Exception as e:
            logger.error(f"❌ 系统资源检查失败: {e}")
            return {}
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """获取GPU信息"""
        try:
            # 尝试导入GPU相关库
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_info = {
                    "available": True,
                    "count": gpu_count,
                    "devices": []
                }
                
                for i in range(gpu_count):
                    device = torch.cuda.get_device_properties(i)
                    gpu_info["devices"].append({
                        "id": i,
                        "name": device.name,
                        "memory_total_gb": device.total_memory / (1024**3),
                        "memory_allocated_gb": torch.cuda.memory_allocated(i) / (1024**3)
                    })
                
                logger.info(f"🎮 GPU信息: {gpu_count} 个GPU可用")
                return gpu_info
            else:
                logger.warning("⚠️ GPU不可用")
                return {"available": False}
                
        except ImportError:
            logger.warning("⚠️ PyTorch未安装，无法获取GPU信息")
            return {"available": False}
        except Exception as e:
            logger.error(f"❌ GPU信息获取失败: {e}")
            return {"available": False}
    
    def optimize_environment(self):
        """优化运行环境"""
        logger.info("🔧 优化运行环境...")
        
        # 设置环境变量
        env_vars = {
            "PYTHONPATH": f"{os.getcwd()}:{os.environ.get('PYTHONPATH', '')}",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3" if self.enable_gpu else "",
            "OMP_NUM_THREADS": str(psutil.cpu_count()),
            "MKL_NUM_THREADS": str(psutil.cpu_count()),
            "NUMEXPR_NUM_THREADS": str(psutil.cpu_count()),
            "TOKENIZERS_PARALLELISM": "false"
        }
        
        # 设置内存限制
        if self.memory_limit_gb > 0:
            env_vars["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{self.memory_limit_gb * 1024}"
        
        # 应用环境变量
        for key, value in env_vars.items():
            if value:
                os.environ[key] = value
                logger.info(f"   设置环境变量: {key}={value}")
        
        # 清理内存
        gc.collect()
        
        logger.info("✅ 环境优化完成")
    
    def create_training_script(self) -> str:
        """创建训练脚本"""
        script_content = f'''#!/usr/bin/env python3
"""
自动生成的多模型训练脚本
生成时间: {datetime.now().isoformat()}
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

# 添加项目路径
sys.path.insert(0, "{os.getcwd()}")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('{self.log_dir}/training.log'),
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
        logger.error(f"❌ 训练失败: {{e}}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        
        script_path = self.output_dir / "run_training.py"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # 设置执行权限
        os.chmod(script_path, 0o755)
        
        logger.info(f"📝 训练脚本已创建: {script_path}")
        return str(script_path)
    
    def monitor_performance(self, process: subprocess.Popen):
        """监控训练过程性能"""
        logger.info("📊 开始性能监控...")
        
        while process.poll() is None:
            try:
                # 获取进程信息
                process_info = psutil.Process(process.pid)
                
                # 内存使用
                memory_usage = process_info.memory_info()
                memory_gb = memory_usage.rss / (1024**3)
                
                # CPU使用
                cpu_percent = process_info.cpu_percent()
                
                # 磁盘使用
                disk_usage = psutil.disk_usage(self.output_dir)
                disk_gb = disk_usage.used / (1024**3)
                
                # 记录指标
                self.performance_metrics["memory_usage"].append({
                    "timestamp": datetime.now().isoformat(),
                    "memory_gb": memory_gb
                })
                
                self.performance_metrics["cpu_usage"].append({
                    "timestamp": datetime.now().isoformat(),
                    "cpu_percent": cpu_percent
                })
                
                self.performance_metrics["disk_usage"].append({
                    "timestamp": datetime.now().isoformat(),
                    "disk_gb": disk_gb
                })
                
                # 每30秒记录一次
                if len(self.performance_metrics["memory_usage"]) % 30 == 0:
                    logger.info(f"📈 性能监控: 内存={memory_gb:.1f}GB, CPU={cpu_percent:.1f}%, 磁盘={disk_gb:.1f}GB")
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ 性能监控错误: {e}")
                break
    
    def save_checkpoint(self, checkpoint_name: str = None):
        """保存检查点"""
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(exist_ok=True)
        
        # 保存配置
        config_file = checkpoint_path / "run_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.run_config, f, indent=2, ensure_ascii=False)
        
        # 保存性能指标
        metrics_file = checkpoint_path / "performance_metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.performance_metrics, f, indent=2, ensure_ascii=False)
        
        # 复制训练结果
        result_files = list(self.output_dir.glob("*.json"))
        for result_file in result_files:
            shutil.copy2(result_file, checkpoint_path)
        
        # 复制日志文件
        log_files = list(self.log_dir.glob("*.log"))
        for log_file in log_files:
            shutil.copy2(log_file, checkpoint_path)
        
        logger.info(f"💾 检查点已保存: {checkpoint_path}")
        
        # 记录检查点信息
        self.performance_metrics["checkpoints"].append({
            "name": checkpoint_name,
            "timestamp": datetime.now().isoformat(),
            "path": str(checkpoint_path)
        })
        
        return checkpoint_path
    
    def run_training(self) -> bool:
        """运行训练"""
        logger.info("🚀 启动多模型训练系统...")
        
        # 检查系统资源
        system_info = self.check_system_resources()
        self.run_config["system_info"] = system_info
        
        # 优化环境
        self.optimize_environment()
        
        # 创建训练脚本
        script_path = self.create_training_script()
        
        # 设置工作目录
        work_dir = os.getcwd()
        
        try:
            # 启动训练进程
            logger.info(f"📝 执行训练脚本: {script_path}")
            
            process = subprocess.Popen(
                [sys.executable, script_path],
                cwd=work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # 启动性能监控（在后台线程中）
            import threading
            monitor_thread = threading.Thread(
                target=self.monitor_performance, 
                args=(process,),
                daemon=True
            )
            monitor_thread.start()
            
            # 实时输出日志
            logger.info("📋 训练日志:")
            logger.info("=" * 80)
            
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
                
                # 保存最终检查点
                self.save_checkpoint("final_checkpoint")
                
                # 保存运行报告
                self.save_run_report()
                
                return True
            else:
                logger.error(f"❌ 训练失败，返回码: {return_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ 训练执行失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def save_run_report(self):
        """保存运行报告"""
        report = {
            "run_config": self.run_config,
            "performance_metrics": self.performance_metrics,
            "end_time": datetime.now().isoformat(),
            "status": "completed"
        }
        
        report_file = self.output_dir / "run_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 运行报告已保存: {report_file}")
    
    def cleanup(self):
        """清理临时文件"""
        logger.info("🧹 清理临时文件...")
        
        # 清理临时脚本
        temp_script = self.output_dir / "run_training.py"
        if temp_script.exists():
            temp_script.unlink()
            logger.info(f"   删除临时脚本: {temp_script}")
        
        logger.info("✅ 清理完成")

def main():
    """主函数"""
    print("🚀 多模型训练系统运行器")
    print("=" * 60)
    
    # 配置参数
    config = {
        "output_dir": "./training_outputs",
        "checkpoint_dir": "./checkpoints", 
        "log_dir": "./logs",
        "memory_limit_gb": 32,  # 32GB内存限制
        "enable_gpu": True
    }
    
    # 创建运行器
    runner = MultiModelTrainingRunner(**config)
    
    try:
        # 运行训练
        success = runner.run_training()
        
        if success:
            print("\n🎉 训练成功完成！")
            print(f"📁 输出目录: {runner.output_dir}")
            print(f"💾 检查点目录: {runner.checkpoint_dir}")
            print(f"📋 日志目录: {runner.log_dir}")
        else:
            print("\n❌ 训练失败！")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
        runner.save_checkpoint("interrupted_checkpoint")
    except Exception as e:
        print(f"\n❌ 运行器错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # 清理
        runner.cleanup()

if __name__ == "__main__":
    main()
