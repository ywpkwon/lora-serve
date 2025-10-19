import os
import asyncio
from fastapi import FastAPI
from .core.config import settings
from .core.logging import configure_logging
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
    configure_logging(level=os.getenv("LORASERVE_LOGLEVEL", "INFO"))
    await asyncio.sleep(0)
