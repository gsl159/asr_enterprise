"""
资源监控器
监控 GPU 和系统资源使用情况
"""
import torch
from core.logger import get_logger

logger = get_logger("resource_monitor")


class ResourceMonitor:
    """
    资源监控
    安全地获取 GPU 内存信息
    """

    @staticmethod
    def gpu_memory():
        """
        获取所有 GPU 的内存使用情况
        
        Returns:
            dict: GPU 内存信息，格式 {device: {allocated_MB, reserved_MB}}
        """
        try:
            if not torch.cuda.is_available():
                return {}

            result = {}

            for i in range(torch.cuda.device_count()):
                try:
                    allocated = torch.cuda.memory_allocated(i) / 1024**2
                    reserved = torch.cuda.memory_reserved(i) / 1024**2

                    result[f"cuda:{i}"] = {
                        "allocated_MB": round(allocated, 2),
                        "reserved_MB": round(reserved, 2)
                    }
                except Exception as e:
                    logger.warning(f"Failed to get memory for cuda:{i}: {str(e)}")
                    result[f"cuda:{i}"] = {"error": str(e)}

            return result
        
        except Exception as e:
            logger.error(f"GPU memory query failed: {str(e)}")
            return {}
