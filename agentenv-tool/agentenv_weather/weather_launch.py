"""
Entrypoint for the weather agent environment.
"""

import argparse

import uvicorn

from .weather_utils import debug_flg


def launch():
    """entrypoint for `weather` commond"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    uvicorn.run(
        "agentenv_weather:app",
        host=args.host,
        port=args.port,
        reload=debug_flg,
        workers=args.workers,
    )
