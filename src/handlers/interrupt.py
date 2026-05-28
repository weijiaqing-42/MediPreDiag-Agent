from typing import Optional, Any
import asyncio
import logging

logger = logging.getLogger(__name__)


class InterruptHandler:
    CANCEL_KEYWORDS = {"取消", "重新开始", "算了", "不用了", "退出", "停止"}

    def __init__(self):
        self.timeout_threshold: int = 15
        self.max_retry: int = 3

    def check_cancellation(self, message: str) -> bool:
        return any(kw in message for kw in self.CANCEL_KEYWORDS)

    async def execute_with_timeout(self, coro, timeout: int = None):
        timeout = timeout or self.timeout_threshold
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Operation timed out after {timeout}s")
            return None

    async def execute_with_retry(self, coro_func, *args, **kwargs):
        last_error = None
        for attempt in range(self.max_retry):
            try:
                return await coro_func(*args, **kwargs)
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1}/{self.max_retry} failed: {e}")
                if attempt < self.max_retry - 1:
                    await asyncio.sleep(1.0 * (attempt + 1))
        raise last_error

    def build_fallback_response(self, failed_node: str) -> str:
        fallbacks = {
            "intent_classifier": "抱歉，我没有理解您的意思。请描述您的症状或问题。",
            "symptom_extractor": "抱歉，症状分析暂时不可用。建议您前往医院咨询专业医生。",
            "rag_retriever": "知识库暂不可用，但我会尽力为您提供帮助。",
            "location_recommender": "暂时无法查询附近医疗机构。请拨打120求助或使用地图软件搜索。",
            "severity_evaluator": "无法评估病情严重程度。如果您感到严重不适，请立即就医。",
        }
        return fallbacks.get(failed_node, "服务暂时不可用，请稍后重试。")


interrupt_handler = InterruptHandler()