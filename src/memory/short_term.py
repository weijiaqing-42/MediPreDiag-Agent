from src.db.redis_client import redis_client


class ShortTermMemory:
    async def load(self, session_id: str) -> list[dict]:
        return await redis_client.get_session_history(session_id)

    async def save(self, session_id: str, entry: dict):
        await redis_client.append_session_history(session_id, entry)

    async def clear(self, session_id: str):
        await redis_client.clear_session(session_id)

    async def get_context_str(self, session_id: str, max_turns: int = 10) -> str:
        history = await self.load(session_id)
        recent = history[-max_turns:] if len(history) > max_turns else history
        lines = []
        for entry in recent:
            role = "用户" if entry.get("role") == "user" else "助手"
            lines.append(f"{role}: {entry.get('content', '')}")
        return "\n".join(lines)


short_term_memory = ShortTermMemory()