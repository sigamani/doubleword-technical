"""
Application context and dependency injection
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AppContext:
    """Application-wide context and state"""
    config: Dict[str, Any]
    processor: Any
    metrics: Any
    monitor: Any
    redis_client: Any
    
    def __init__(self):
        self.config = {}
        self.processor = None
        self.metrics = None
        self.monitor = None
        self.redis_client = None

# Global context instance
_context: Optional[AppContext] = None

def get_context() -> AppContext:
    """Get application context (dependency injection)"""
    global _context
    if _context is None:
        _context = AppContext()
    return _context

def set_context(context: AppContext):
    """Set application context"""
    global _context
    _context = context