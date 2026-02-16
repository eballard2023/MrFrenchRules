"""
User Authentication Module
Simple JWT-based auth for non-admin application users (experts).
"""

from datetime import datetime, timedelta
from typing import Optional, Dict

import jwt
import os
import bcrypt

from supabase_client import supabase_client


class UserAuth:
    def __init__(self):
        # Reuse the same JWT secret key as admin auth
        env = os.getenv("ENV", "development")
        self.jwt_secret = os.getenv("JWT_SECRET_KEY") or ("dev-insecure-jwt" if env != "production" else None)
        if not self.jwt_secret:
            raise RuntimeError("JWT_SECRET_KEY must be set in production")
        self.token_expiry_hours = 24

    def _generate_token(self, user_id: int, email: str, name: str, role: str) -> str:
        """Generate JWT token with role."""
        payload = {
            "sub": user_id,
            "email": email,
            "name": name,
            "role": role,
            "exp": datetime.utcnow() + timedelta(hours=self.token_expiry_hours),
            "iat": datetime.utcnow(),
        }
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")

    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify JWT token and return user info including role."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            # Return standardized user object
            return {
                "id": payload.get("sub"),
                "email": payload.get("email"),
                "name": payload.get("name"),
                "role": payload.get("role", "user")
            }
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def register(self, email: str, password: str, name: str) -> Optional[Dict]:
        """
        Create or update a regular user (default role='user').
        """
        if not supabase_client.connected:
            supabase_client.connect()
        if not supabase_client.connected:
            return None

        conn = None
        try:
            password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
            conn = supabase_client.get_connection()
            with conn.cursor() as cur:
                # Default role is 'user' for public registration
                cur.execute(
                    """
                    INSERT INTO users (email, password_hash, name, role)
                    VALUES (%s, %s, %s, 'user')
                    ON CONFLICT (email) DO UPDATE SET
                        password_hash = EXCLUDED.password_hash,
                        name = EXCLUDED.name,
                        is_active = TRUE
                    RETURNING id, email, name, role, created_at;
                    """,
                    (email, password_hash, name),
                )
                row = cur.fetchone()
                conn.commit()

                user_data = {
                    "id": row[0],
                    "email": row[1],
                    "name": row[2],
                    "role": row[3],
                    "created_at": row[4],
                }
                token = self._generate_token(user_data["id"], user_data["email"], user_data["name"], user_data["role"])
                return {
                    "token": token,
                    "user": {
                        "id": user_data["id"],
                        "email": user_data["email"],
                        "name": user_data["name"],
                        "role": user_data["role"]
                    },
                }
        except Exception as e:
            if os.getenv("ENV", "development") != "production":
                print(f"❌ USER REGISTER ERROR: {e}")
            if conn:
                conn.rollback()
            return None
        finally:
            if conn:
                supabase_client.put_connection(conn)

    def authenticate(self, email: str, password: str) -> Optional[Dict]:
        """Authenticate user against users table (unified auth)."""
        if not supabase_client.connected:
            supabase_client.connect()
        if not supabase_client.connected:
            return None

        conn = None
        try:
            conn = supabase_client.get_connection()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, password_hash, name, role
                    FROM users
                    WHERE email = %s AND is_active = TRUE;
                    """,
                    (email,),
                )
                row = cur.fetchone()

                if not row:
                    return None

                user_id, stored_hash, name, role = row
                if not stored_hash:
                    return None

                if not bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8")):
                    return None

                token = self._generate_token(user_id, email, name, role)
                return {
                    "token": token,
                    "user": {
                        "id": user_id,
                        "email": email,
                        "name": name,
                        "role": role
                    },
                }
        except Exception as e:
            if os.getenv("ENV", "development") != "production":
                print(f"❌ AUTH ERROR: {e}")
            return None
        finally:
            if conn:
                supabase_client.put_connection(conn)


# Global auth instance
# basic aliasing for now, will replace usage in main.py
user_auth = UserAuth()
auth_service = user_auth

