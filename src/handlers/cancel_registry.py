import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class CancelRegistry:
    def __init__(self):
        self._cancelled: set[str] = set()
        self._tasks: dict[str, asyncio.Task] = {}

    def set(self, session_id: str):
        self._cancelled.add(session_id)
        logger.info(f"Cancel flag set for session {session_id}")

    def is_cancelled(self, session_id: str) -> bool:
        return session_id in self._cancelled

    def clear(self, session_id: str):
        self._cancelled.discard(session_id)
        self._tasks.pop(session_id, None)

    def clear_all(self):
        self._cancelled.clear()
        for task in self._tasks.values():
            task.cancel()
        self._tasks.clear()

    def register_task(self, session_id: str, task: asyncio.Task):
        self._tasks[session_id] = task
        logger.debug(f"Task registered for session {session_id}")

    def unregister_task(self, session_id: str):
        self._tasks.pop(session_id, None)

    def cancel_session(self, session_id: str) -> bool:
        self._cancelled.add(session_id)

        task = self._tasks.get(session_id)
        if task is not None and not task.done():
            task.cancel()
            logger.info(f"Task cancelled for session {session_id}")
            return True

        logger.info(f"No active task found for session {session_id}")
        return False


cancel_registry = CancelRegistry()