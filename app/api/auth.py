import logging
import time
import jwt
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TokenManager:
    """JWT token management for authentication"""
    
    def __init__(self, secret: str, algorithm: str = "HS256"):
        self.secret = secret
        self.algorithm = algorithm
        self.token_blacklist = set()
    
    def generate_token(self, user_id: str, tier: str = "basic", expires_hours: int = 24) -> str:
        """Generate JWT token with user info and tier"""
        expires_at = datetime.utcnow() + timedelta(hours=expires_hours)
        payload = {
            "user_id": user_id,
            "tier": tier,
            "exp": expires_at,
            "iat": datetime.utcnow()
        }
        return jwt.encode(payload, self.secret, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify JWT token and return payload"""
        try:
            if token in self.token_blacklist:
                return None
            
            payload = jwt.decode(token, self.secret, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
    
    def revoke_token(self, token: str):
        """Add token to blacklist"""
        self.token_blacklist.add(token)

class RateLimiter:
    """Tier-based rate limiting"""
    
    # Rate limits per hour
    TIER_LIMITS = {
        "free": 100,
        "basic": 500,
        "premium": 2000,
        "enterprise": 10000
    }
    
    def __init__(self):
        self.user_requests = {}  # user_id -> [(timestamp, count), ...]
    
    def check_rate_limit(self, user_id: str, tier: str) -> bool:
        """Check if user is within rate limits"""
        current_time = time.time()
        hour_ago = current_time - 3600
        
        # Clean old requests
        if user_id in self.user_requests:
            self.user_requests[user_id] = [
                (ts, count) for ts, count in self.user_requests[user_id] 
                if ts > hour_ago
            ]
        else:
            self.user_requests[user_id] = []
        
        # Count current hour requests
        current_requests = sum(count for ts, count in self.user_requests[user_id])
        limit = self.TIER_LIMITS.get(tier, 500)
        
        if current_requests >= limit:
            logger.warning(f"Rate limit exceeded for user {user_id}, tier {tier}")
            return False
        
        # Record this request
        self.user_requests[user_id].append((current_time, 1))
        return True

# Global instances
token_manager = TokenManager("default-secret")
rate_limiter = RateLimiter()