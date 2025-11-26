"""
Authentication and authorization components
"""

import logging
import os
import jwt
from typing import Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TokenManager:
    """JWT-based token management system"""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.blacklisted_tokens = set()
    
    def generate_token(self, user_id: str, tier: str, expires_hours: int = 24) -> str:
        """Generate JWT token with tier-based expiration"""
        now = datetime.utcnow()
        exp = now + timedelta(hours=expires_hours)
        
        payload = {
            "user_id": user_id,
            "tier": tier,
            "exp": exp,
            "iat": now
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        logger.info(f"Generated token for user {user_id}, tier {tier}, expires {exp}")
        return token
    
    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            
            # Check if token is blacklisted
            if token in self.blacklisted_tokens:
                logger.warning(f"Blacklisted token used: {token[:10]}...")
                return None
            
            # Check expiration
            if datetime.utcnow() > datetime.fromtimestamp(payload["exp"]):
                logger.warning(f"Expired token used: {token[:10]}...")
                return None
            
            logger.info(f"Token verified for user {payload['user_id']}, tier {payload['tier']}")
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token signature expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    def revoke_token(self, token: str) -> bool:
        """Revoke a token by adding to blacklist"""
        self.blacklisted_tokens.add(token)
        logger.info(f"Token revoked: {token[:10]}...")
        return True

class RateLimiter:
    """Tier-based rate limiting"""
    
    def __init__(self):
        self.requests = {}
    
    def check_rate_limit(self, user_id: str, tier: str) -> bool:
        """Check if user exceeds rate limit"""
        now = datetime.utcnow()
        limits = {
            "free": {"requests_per_minute": 10, "requests_per_hour": 100},
            "basic": {"requests_per_minute": 30, "requests_per_hour": 500},
            "premium": {"requests_per_minute": 100, "requests_per_hour": 2000},
            "enterprise": {"requests_per_minute": 1000, "requests_per_hour": 10000}
        }
        
        user_limit = limits.get(tier, limits["basic"])
        
        # Clean old requests
        cutoff = now - timedelta(hours=1)
        self.requests[user_id] = [
            req_time for req_time in self.requests.get(user_id, [])
            if req_time > cutoff
        ]
        
        # Count recent requests
        recent_requests = [
            req_time for req_time in self.requests.get(user_id, [])
            if req_time > now - timedelta(minutes=1)
        ]
        
        if len(recent_requests) >= user_limit["requests_per_minute"]:
            logger.warning(f"Rate limit exceeded for user {user_id}, tier {tier}")
            return False
        
        # Add current request
        self.requests.setdefault(user_id, []).append(now)
        return True

# Initialize with environment variables
token_manager = TokenManager(os.getenv("JWT_SECRET", "default-secret-key"))
rate_limiter = RateLimiter()