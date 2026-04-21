"""
ASR 基础模型类
所有具体模型实现必须继承此类
"""
from abc import ABC, abstractmethod
import torch

from core.logger import get_logger

logger = get_logger("base_model")


class BaseASRModel(ABC):
    """
    ASR 模型基类
    
    特点：
    - 抽象基类，定义模型接口
    - 自动加载模型到指定设备
    - 支持模型卸载和显存清理
    """

    def __init__(self, model_name: str, device: str):
        """
        初始化模型
        
        Args:
            model_name: 模型名称标识
            device: 设备标识，如 'cuda:0' 或 'cpu'
        """
        self.model_name = model_name
        self.device = device
        self.model = None

        logger.info(f"Loading model: {model_name} on {device}")
        
        # 调用子类实现的加载方法
        self.load_model()
        
        logger.info(f"✓ {model_name} loaded on {device}")

    @abstractmethod
    def load_model(self):
        """
        加载模型到指定设备
        子类必须实现此方法
        
        示例：
            self.model = SomeModel()
            self.model.to(self.device)
        """
        pass

    @abstractmethod
    def infer(self, audio_path: str) -> str:
        """
        执行推理
        
        Args:
            audio_path: 音频文件路径
        
        Returns:
            str: 识别文本
        
        示例：
            result = self.model.transcribe(audio_path)
            return result['text']
        """
        pass

    def cleanup(self):
        """
        释放模型和GPU显存
        应用关闭时调用
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"{self.model_name} GPU memory cleared")