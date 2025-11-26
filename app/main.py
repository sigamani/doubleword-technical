"""
Main application entry point with proper initialization
"""

import logging
import uvicorn
from app.api.routes import app
from app.core.config import get_config

logger = logging.getLogger(__name__)

def main():
    """Main entry point"""
    config = get_config()
    
    logger.info("Starting Ray Data vLLM Batch Inference Server")
    logger.info(f"Configuration loaded: {config.model.name}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level=config.monitoring.log_level,
        access_log=True
    )

if __name__ == "__main__":
    main()