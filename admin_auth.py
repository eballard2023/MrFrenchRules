"""
Admin Authentication Module
Simple authentication system for admin panel
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict
import jwt
import os
from supabase_client import supabase_client

class AdminAuth:
    def __init__(self):
        # JWT secret key - use environment variable or default
        self.jwt_secret = os.getenv("JWT_SECRET_KEY", "ai-coach-jwt-secret-change-in-production")
        self.token_expiry_hours = 24
    
    def _hash_password(self, password: str) -> str:
        """Hash password with salt"""
        salt = "ai_coach_salt"  # In production, use random salt per user
        return hashlib.sha256((password + salt).encode()).hexdigest()
    
    def authenticate(self, email: str, password: str) -> Optional[Dict]:
        """Authenticate admin user against database"""
        print(f"🔐 AUTH: Attempting login for {email}")
        
        # Use database authentication
        user_data = supabase_client.authenticate_admin(email, password)
        
        if user_data:
            # Generate JWT token
            token = self._generate_token(user_data["email"], user_data["name"])
            print(f"✅ AUTH: Generated token for {email}")
            return {
                "token": token,
                "user": user_data
            }
        
        print(f"❌ AUTH: Failed for {email}")
        return None
    
    def _generate_token(self, email: str, name: str) -> str:
        """Generate JWT token"""
        payload = {
            "email": email,
            "name": name,
            "exp": datetime.utcnow() + timedelta(hours=self.token_expiry_hours),
            "iat": datetime.utcnow()
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")
    
    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            return {
                "email": payload["email"],
                "name": payload["name"]
            }
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

# Global auth instance
admin_auth = AdminAuth()