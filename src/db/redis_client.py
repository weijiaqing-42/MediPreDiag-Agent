import json
import logging
from typing import Optional, List
import redis.asyncio as aioredis
from src.config import settings
from src.memory.fallback import fallback_store

logger = logging.getLogger(__name__)


class RedisClient:
    def __init__(self):
        self._pool: Optional[aioredis.Redis] = None
        self._is_connected: bool = False
        self._fallback = fallback_store

    async def connect(self):
        try:
            self._pool = aioredis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                password=settings.redis_password or None,
                db=settings.redis_db,
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=3,
            )
            await self._pool.ping()
            self._is_connected = True
            logger.info("Redis connected successfully")
        except Exception as e:
            self._is_connected = False
            self._pool = None
            logger.warning(f"Redis unavailable, using in-memory fallback: {e}")

    async def disconnect(self):
        if self._pool:
            try:
                await self._pool.close()
            except Exception:
                pass
        self._is_connected = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected and self._pool is not None

    async def _redis_get(self, key: str) -> Optional[str]:
        if self.is_connected:
            try:
                return await self._pool.get(key)
            except Exception as e:
                logger.warning(f"Redis get failed, switching to fallback: {e}")
                self._is_connected = False
        return None

    async def _redis_setex(self, key: str, ttl: int, value: str) -> bool:
        if self.is_connected:
            try:
                await self._pool.setex(key, ttl, value)
                return True
            except Exception as e:
                logger.warning(f"Redis setex failed, switching to fallback: {e}")
                self._is_connected = False
        return False

    async def _redis_delete(self, key: str) -> bool:
        if self.is_connected:
            try:
                await self._pool.delete(key)
                return True
            except Exception as e:
                logger.warning(f"Redis delete failed, switching to fallback: {e}")
                self._is_connected = False
        return False

    async def get_session_history(self, session_id: str) -> List[dict]:
        key = f"session:{session_id}:history"
        data = await self._redis_get(key)
        if data is not None:
            return json.loads(data)
        if not self.is_connected:
            result = await self._fallback.get_json(key)
            return result if result is not None else []
        return []

    async def append_session_history(self, session_id: str, entry: dict):
        key = f"session:{session_id}:history"
        history = await self.get_session_history(session_id)
        history.append(entry)
        serialized = json.dumps(history, ensure_ascii=False)
        if not await self._redis_setex(key, settings.session_ttl, serialized):
            await self._fallback.setex_json(key, settings.session_ttl, history)

    async def set_interrupt_state(self, session_id: str, state: dict):
        key = f"session:{session_id}:interrupt"
        serialized = json.dumps(state, ensure_ascii=False)
        if not await self._redis_setex(key, settings.session_ttl, serialized):
            await self._fallback.setex_json(key, settings.session_ttl, state)

    async def get_interrupt_state(self, session_id: str) -> Optional[dict]:
        key = f"session:{session_id}:interrupt"
        data = await self._redis_get(key)
        if data is not None:
            return json.loads(data)
        if not self.is_connected:
            return await self._fallback.get_json(key)
        return None

    async def clear_interrupt_state(self, session_id: str):
        key = f"session:{session_id}:interrupt"
        if not await self._redis_delete(key):
            await self._fallback.delete(key)

    async def clear_session(self, session_id: str):
        keys = [
            f"session:{session_id}:history",
            f"session:{session_id}:interrupt",
        ]
        for key in keys:
            if not await self._redis_delete(key):
                await self._fallback.delete(key)

    async def set_cancel_flag(self, session_id: str):
        key = f"session:{session_id}:cancel"
        if not await self._redis_setex(key, 300, "1"):
            await self._fallback.setex(key, 300, b"1")
        logger.info(f"Cancel flag set for session {session_id}")

    async def check_cancel_flag(self, session_id: str) -> bool:
        key = f"session:{session_id}:cancel"
        data = await self._redis_get(key)
        if data is not None:
            return data == "1"
        if not self.is_connected:
            result = await self._fallback.get(key)
            return result is not None
        return False

    async def clear_cancel_flag(self, session_id: str):
        key = f"session:{session_id}:cancel"
        if not await self._redis_delete(key):
            await self._fallback.delete(key)


redis_client = RedisClient()