from contextlib import asynccontextmanager
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from src.config import settings
from src.routes.chat import router as chat_router
from src.routes.ws import router as ws_router
from src.db.redis_client import redis_client
from src.db.milvus_client import milvus_client
from src.db.mysql import init_mysql_schema


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await asyncio.wait_for(redis_client.connect(), timeout=5.0)
    except Exception as e:
        print(f"Warning: Redis not available: {e}")
    try:
        milvus_client.init_collection()
    except Exception as e:
        print(f"Warning: Milvus not available: {e}")
    try:
        init_mysql_schema()
    except Exception as e:
        print(f"Warning: MySQL not available: {e}")
    yield
    try:
        await redis_client.disconnect()
    except Exception:
        pass
    try:
        milvus_client.disconnect()
    except Exception:
        pass


app = FastAPI(
    title="MediPreDiag-Agent",
    description="Multi-Agent Collaborative Medical Pre-Diagnosis System",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(ws_router)

import os
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def root():
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok", "service": "MediPreDiag-Agent", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.app_host,
        port=settings.app_port,
        log_level=settings.log_level.lower(),
    )