import asyncio
import time
import json
from typing import Optional, List


class MemoryFallback:
    def __init__(self, default_ttl: int = 1800):
        self._store: dict[str, dict] = {}
        self._lock = asyncio.Lock()
        self._default_ttl = default_ttl

    def _now(self) -> float:
        return time.monotonic()

    def _make_entry(self, value, ttl: int = None) -> dict:
        ttl = ttl if ttl is not None else self._default_ttl
        return {"value": value, "expires_at": self._now() + ttl}

    def _is_expired(self, entry: dict) -> bool:
        return self._now() > entry["expires_at"]

    async def _cleanup_expired(self):
        expired = [k for k, v in self._store.items() if self._is_expired(v)]
        for k in expired:
            del self._store[k]

    async def get(self, key: str) -> Optional[bytes]:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if self._is_expired(entry):
                del self._store[key]
                return None
            return entry["value"] if isinstance(entry["value"], bytes) else entry["value"].encode("utf-8")

    async def setex(self, key: str, ttl: int, value):
        async with self._lock:
            if isinstance(value, str):
                value = value.encode("utf-8")
            self._store[key] = self._make_entry(value, ttl)

    async def delete(self, key: str):
        async with self._lock:
            self._store.pop(key, None)

    async def clear_all(self):
        async with self._lock:
            self._store.clear()

    async def keys(self, pattern: str = "*") -> List[str]:
        async with self._lock:
            await self._cleanup_expired()
            return list(self._store.keys())

    async def get_json(self, key: str) -> Optional[list]:
        data = await self.get(key)
        if data is None:
            return None
        try:
            return json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return None

    async def setex_json(self, key: str, ttl: int, value):
        await self.setex(key, ttl, json.dumps(value, ensure_ascii=False))

    @property
    def connected(self) -> bool:
        return True


fallback_store = MemoryFallback()