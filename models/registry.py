"""
模型注册表
支持模型的动态注册和获取
新增模型无需修改此文件
"""

# 全局模型注册表：{模型名称 -> 模型类}
MODEL_REGISTRY = {}


def register_model(name: str):
    """
    模型注册装饰器
    
    使用方式：
        @register_model("my_model")
        class MyModel(BaseASRModel):
            ...
    
    Args:
        name: 模型注册名称
    
    Returns:
        装饰器函数
    """
    def wrapper(cls):
        # 将模型类注册到全局表
        MODEL_REGISTRY[name] = cls
        print(f"✓ Model registered: {name} -> {cls.__name__}")
        return cls
    
    return wrapper


def get_model_class(name: str):
    """
    从注册表获取模型类
    
    Args:
        name: 模型名称
    
    Returns:
        type: 模型类
    
    Raises:
        ValueError: 模型未注册
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{name}' not registered. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )
    
    return MODEL_REGISTRY[name]