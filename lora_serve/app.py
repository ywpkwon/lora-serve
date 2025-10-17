
import asyncio
from fastapi import FastAPI
from .core.config import settings
from .api.routes import api_router
from .metrics.prometheus import metrics_app

def create_app() -> FastAPI:
    app = FastAPI(title="LoRAServe")
    app.include_router(api_router, prefix="/v1")
    app.mount("/metrics", metrics_app)
    return app

app = create_app()

@app.on_event("startup")
async def on_startup():
    # TODO: warm up models if desired
    await asyncio.sleep(0)
