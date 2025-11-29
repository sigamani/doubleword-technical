"""Main entry point for the offline batch inference service."""
from ray import logger
import uvicorn

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
from api.routes import app
import logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting Uvicorn server with FastAPI app on 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)