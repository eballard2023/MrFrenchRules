"""
Admin Authentication Module
Simple authentication system for admin panel
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict
import jwt

class AdminAuth:
    def __init__(self):
        # In production, store these in database
        # For now, hardcoded admin credentials
        self.admin_users = {
            "admin@aicoach.com": {
                "password_hash": self._hash_password("admin123"),
                "name": "Admin User"
            }
        }
        
        # JWT secret key - in production, use environment variable
        self.jwt_secret = "your-secret-key-change-in-production"
        self.token_expiry_hours = 24
    
    def _hash_password(self, password: str) -> str:
        """Hash password with salt"""
        salt = "ai_coach_salt"  # In production, use random salt per user
        return hashlib.sha256((password + salt).encode()).hexdigest()
    
    def authenticate(self, email: str, password: str) -> Optional[Dict]:
        """Authenticate admin user"""
        print(f"Auth attempt - Email: {email}, Password: {password}")
        print(f"Available users: {list(self.admin_users.keys())}")
        
        if email not in self.admin_users:
            print(f"Email {email} not found in users")
            return None
        
        user = self.admin_users[email]
        password_hash = self._hash_password(password)
        expected_hash = user["password_hash"]
        
        print(f"Password hash: {password_hash}")
        print(f"Expected hash: {expected_hash}")
        print(f"Hashes match: {password_hash == expected_hash}")
        
        if password_hash == user["password_hash"]:
            # Generate JWT token
            token = self._generate_token(email, user["name"])
            print(f"Generated token: {token[:20]}...")
            return {
                "token": token,
                "user": {
                    "email": email,
                    "name": user["name"]
                }
            }
        
        print("Password hash mismatch")
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