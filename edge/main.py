from __future__ import annotations

import logging

import uvicorn

from edge.runtime import EdgeRuntime, create_app
from edge.settings import settings

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
runtime = EdgeRuntime()
app = create_app(runtime)

if __name__ == "__main__":
    uvicorn.run("edge.main:app", host="0.0.0.0", port=settings.edge_api_port, reload=False)
