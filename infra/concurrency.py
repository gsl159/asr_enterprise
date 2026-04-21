"""
并发限流器 - 优化版
支持异步协程和 Future 对象
"""
import asyncio


class RequestLimiter:
    """
    请求限流器
    使用信号量限制并发数量
    """

    def __init__(self, max_concurrent: int):
        """
        初始化限流器
        
        Args:
            max_concurrent: 最大并发请求数
        """
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def run(self, coro_or_future):
        """
        运行异步任务或 Future，限制并发
        
        Args:
            coro_or_future: 协程对象或 Future 对象
        
        Returns:
            任务结果
        """
        async with self.semaphore:
            # 支持协程和 Future
            if asyncio.iscoroutine(coro_or_future):
                return await coro_or_future
            else:
                # Future 对象直接等待
                return await coro_or_future
