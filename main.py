"""Точка входа: `python main.py`."""

import uvicorn

from src import config
from src.app import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level=config.LOG_LEVEL.lower(), ws="websockets")
