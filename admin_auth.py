"""
Admin Authentication Module
Simple authentication system for admin panel
"""

from datetime import datetime, timedelta
from typing import Optional, Dict
import jwt
import os
from supabase_client import supabase_client


class AdminAuth:
    def __init__(self):
        env = os.getenv("ENV", "development")
        self.jwt_secret = os.getenv("JWT_SECRET_KEY") or ("dev-insecure-jwt" if env != "production" else None)
        if not self.jwt_secret:
            raise RuntimeError("JWT_SECRET_KEY must be set in production")
        self.token_expiry_hours = 24

    def authenticate(self, email: str, password: str) -> Optional[Dict]:
        """Authenticate admin user against database"""
        # Avoid logging sensitive data in production
        if os.getenv("ENV", "development") != "production":
            print(f"🔐 AUTH: Attempting login for {email}")
        
        # Use database authentication
        user_data = supabase_client.authenticate_admin(email, password)
        
        if user_data:
            token = self._generate_token(user_data["email"], user_data["name"])
            if os.getenv("ENV", "development") != "production":
                print(f"✅ AUTH: Generated token for {email}")
            return {
                "token": token,
                "user": user_data
            }
        if os.getenv("ENV", "development") != "production":
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