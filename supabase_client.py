import psycopg2
import psycopg2.extensions
import os
import socket
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure DNS to use Google's public DNS
try:
    import dns.resolver
    dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
    dns.resolver.default_resolver.nameservers = ['8.8.8.8', '8.8.4.4']
except ImportError:
    pass  # dnspython not installed

# Database configuration from environment (using working parameters)
DB_HOST = os.getenv("host")
DB_NAME = os.getenv("dbname")
DB_USER = os.getenv("user")
DB_PASSWORD = os.getenv("password")
DB_PORT = os.getenv("port", "6543")

print(f"DB_HOST: {DB_HOST}")
print(f"DB_NAME: {DB_NAME}")
print(f"DB_USER: {DB_USER}")
print(f"DB_PASSWORD: {'***' if DB_PASSWORD else 'None'}")
print(f"DB_PORT: {DB_PORT}")

class SupabaseClient:
    def __init__(self):
        self.connection = None
    
    def connect(self):
        """Establishes a connection to the PostgreSQL database."""
        try:
            self.connection = psycopg2.connect(
                user=DB_USER,
                password=DB_PASSWORD,
                host=DB_HOST,
                port=DB_PORT,
                dbname=DB_NAME
            )
            self.connected = True
            print("Connection to Supabase PostgreSQL successful!")
            
            # Create admin_users table if it doesn't exist
            self._create_admin_table()
            
            return True
        except Exception as e:
            self.connected = False
            print(f"Error connecting to the database: {e}")
            return False
    
    def _create_admin_table(self):
        """Create admin_users table and insert default admin"""
        try:
            with self.connection.cursor() as cur:
                # Create admin table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS admin_users (
                        id SERIAL PRIMARY KEY,
                        email VARCHAR(255) UNIQUE NOT NULL,
                        password_hash VARCHAR(255) NOT NULL,
                        name VARCHAR(255) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_active BOOLEAN DEFAULT TRUE
                    );
                """)
                
                # Create sessions table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS interview_sessions (
                        session_id VARCHAR(50) PRIMARY KEY,
                        expert_name VARCHAR(255) NOT NULL,
                        expert_email VARCHAR(255) NOT NULL,
                        expertise_area VARCHAR(255) NOT NULL,
                        conversation_history JSONB DEFAULT '[]',
                        current_question_index INTEGER DEFAULT 0,
                        is_complete BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Create interview_rules table with proper defaults
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS interview_rules (
                        id SERIAL PRIMARY KEY,
                        session_id VARCHAR(50) NOT NULL,
                        expert_name VARCHAR(255) NOT NULL,
                        expertise_area VARCHAR(255) NOT NULL,
                        rule_text TEXT NOT NULL,
                        completed BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expert_email VARCHAR(255)
                    );
                """)
                
                # Always recreate admin user from env
                import bcrypt
                admin_email = os.getenv("ADMIN_EMAIL", "admin@coachai.com")
                admin_password = os.getenv("ADMIN_PASSWORD", "admin123")
                password_hash = bcrypt.hashpw(admin_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                
                cur.execute("""
                    INSERT INTO admin_users (email, password_hash, name)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (email) DO UPDATE SET
                    password_hash = EXCLUDED.password_hash,
                    name = EXCLUDED.name
                """, (admin_email, password_hash, "Admin User"))
                
                print(f"✅ Admin user created/updated: {admin_email} / {admin_password}")
                
                self.connection.commit()
                
        except Exception as e:
            print(f"❌ Error creating admin table: {e}")
            if self.connection:
                self.connection.rollback()
    
    def insert_rule(self, session_id: str, expert_name: str, expertise_area: str, rule_text: str, expert_email: str = None):
        """Inserts a new rule into the interview_rules table."""
        if not self.connection or self.connection.closed:
            self.connect()
            
        if not self.connection:
            return None
            
        try:
            with self.connection.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO interview_rules (session_id, expert_name, expertise_area, rule_text)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id;
                    """,
                    (session_id, expert_name, expertise_area, rule_text)
                )
                rule_id = cur.fetchone()[0]
                self.connection.commit()
                return rule_id
        except Exception as error:
            if self.connection:
                self.connection.rollback()
            return None
    
    def get_rule_by_session_id(self, session_id: str):
        """Retrieves a rule by its session_id."""
        print(f"🔍 DB GET: Looking for rule with session_id {session_id}")
        
        # Reconnect if connection is lost
        if not self.connection or self.connection.closed:
            print("🔄 DB RECONNECT: Reconnecting to database")
            self.connect()
        
        if not self.connection:
            print("❌ DB GET FAILED: No database connection")
            return None
            
        try:
            with self.connection.cursor() as cur:
                cur.execute(
                    """
                    SELECT * FROM interview_rules WHERE session_id = %s LIMIT 1;
                    """,
                    (session_id,)
                )
                row = cur.fetchone()
                if row:
                    print(f"✅ DB GET SUCCESS: Found rule for session {session_id}")
                    return row
                else:
                    print(f"⚠️ DB GET: No rule found for session {session_id}")
                return None
        except (Exception, psycopg2.DatabaseError) as error:
            print(f"❌ DB GET ERROR: {error}")
            # Try to reconnect on error
            try:
                self.connect()
            except:
                pass
            return None
    
    def get_all_rules(self):
        """Retrieves all rules from the database."""
        print("🔍 DB SELECT: Getting all rules")
        
        if not self.connection:
            print("❌ DB SELECT FAILED: No database connection")
            return []
            
        try:
            with self.connection.cursor() as cur:
                cur.execute("SELECT * FROM interview_rules ORDER BY created_at DESC;")
                rows = cur.fetchall()
                print(f"✅ DB SELECT SUCCESS: Found {len(rows)} rules")
                return rows
        except (Exception, psycopg2.DatabaseError) as error:
            print(f"❌ DB SELECT ERROR: {error}")
            return []
    
    def update_rule_status(self, session_id: str, completed: bool):
        """Updates the completed status of a rule by session_id."""
        print(f"🔄 DB UPDATE: Setting session {session_id} completed = {completed}")
        
        # Reconnect if connection is lost
        if not self.connection or self.connection.closed:
            print("🔄 DB RECONNECT: Reconnecting to database")
            self.connect()
        
        if not self.connection:
            print("❌ DB UPDATE FAILED: No database connection")
            return False
            
        try:
            with self.connection.cursor() as cur:
                cur.execute(
                    """
                    UPDATE interview_rules SET completed = %s WHERE session_id = %s;
                    """,
                    (completed, session_id)
                )
                self.connection.commit()
                updated = cur.rowcount > 0
                if updated:
                    print(f"✅ DB UPDATE SUCCESS: {cur.rowcount} rows updated")
                else:
                    print("⚠️ DB UPDATE: No rows matched the session_id")
                return updated
        except (Exception, psycopg2.DatabaseError) as error:
            print(f"❌ DB UPDATE ERROR: {error}")
            # Try to reconnect on error
            try:
                self.connect()
                self.connection.rollback()
            except:
                pass
            return False
    
    async def update_rule_completed(self, rule_id: int, completed: bool):
        """Updates the completed status of a rule by ID."""
        print(f"🔄 DB UPDATE: Setting rule {rule_id} completed = {completed}")
        
        # Reconnect if connection is lost
        if not self.connection or self.connection.closed:
            print("🔄 DB RECONNECT: Reconnecting to database")
            self.connect()
        
        if not self.connection:
            print("❌ DB UPDATE FAILED: No database connection")
            return False
            
        try:
            with self.connection.cursor() as cur:
                cur.execute(
                    """
                    UPDATE interview_rules SET completed = %s WHERE id = %s;
                    """,
                    (completed, rule_id)
                )
                self.connection.commit()
                updated = cur.rowcount > 0
                if updated:
                    print(f"✅ DB UPDATE SUCCESS: {cur.rowcount} rows updated")
                else:
                    print("⚠️ DB UPDATE: No rows matched the rule_id")
                return updated
        except (Exception, psycopg2.DatabaseError) as error:
            print(f"❌ DB UPDATE ERROR: {error}")
            # Try to reconnect on error
            try:
                self.connect()
                self.connection.rollback()
            except:
                pass
            return False
    
    async def save_interview_rule(self, session_id: str, expert_name: str, expertise_area: str, rule_text: str, expert_email: str = None):
        """Save interview rule to database."""
        return self.insert_rule(session_id, expert_name, expertise_area, rule_text)
    
    async def get_all_rules(self):
        """Get all rules from database."""
        if not self.connection or self.connection.closed:
            self.connect()
        
        if not self.connection:
            return []
        
        try:
            with self.connection.cursor() as cur:
                cur.execute("SELECT * FROM interview_rules ORDER BY created_at DESC;")
                rows = cur.fetchall()
                return [{
                    'id': row[0],
                    'session_id': row[1], 
                    'expert_name': row[2],
                    'expertise_area': row[3],
                    'rule_text': row[5],
                    'completed': row[4],
                    'created_at': row[6],
                    'expert_email': row[7] if len(row) > 7 else None
                } for row in rows]
        except Exception as e:
            try:
                self.connect()
            except:
                pass
            return []
    
    async def get_rules_by_session(self, session_id: str):
        """Get rules for a specific session."""
        print(f"🔍 ASYNC GET SESSION: Looking for rules in session {session_id}")
        
        if not self.connection:
            print("❌ ASYNC GET SESSION FAILED: No database connection")
            return []
        try:
            with self.connection.cursor() as cur:
                cur.execute(
                    "SELECT * FROM interview_rules WHERE session_id = %s ORDER BY created_at DESC;",
                    (session_id,)
                )
                rows = cur.fetchall()
                rules = [{
                    'id': row[0],
                    'session_id': row[1], 
                    'expert_name': row[2],
                    'expertise_area': row[3],
                    'rule_text': row[5],  # Fixed: rule_text is in column 5
                    'completed': row[4],  # Fixed: completed is in column 4
                    'created_at': row[6],
                    'expert_email': row[7] if len(row) > 7 else None
                } for row in rows]
                print(f"✅ ASYNC GET SESSION SUCCESS: Found {len(rules)} rules for session {session_id}")
                return rules
        except Exception as e:
            print(f"❌ ASYNC GET SESSION ERROR: {e}")
            return []
    
    async def save_session(self, session_id: str, expert_name: str, expert_email: str, expertise_area: str):
        """Save session to database"""
        if not self.connection:
            return False
        try:
            with self.connection.cursor() as cur:
                cur.execute("""
                    INSERT INTO interview_sessions (session_id, expert_name, expert_email, expertise_area)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (session_id) DO UPDATE SET
                    expert_name = EXCLUDED.expert_name,
                    expert_email = EXCLUDED.expert_email,
                    expertise_area = EXCLUDED.expertise_area;
                """, (session_id, expert_name, expert_email, expertise_area))
                self.connection.commit()
                return True
        except Exception as e:
            print(f"❌ SAVE SESSION ERROR: {e}")
            return False
    
    async def update_session(self, session_id: str, conversation_history: list, question_index: int, is_complete: bool):
        """Update session conversation and status"""
        # Reconnect if connection is closed or in bad state
        if not self.connection or self.connection.closed:
            print("🔄 UPDATE SESSION: Reconnecting to database")
            self.connect()
        
        if not self.connection:
            return False
        
        try:
            # Check if we're in a bad transaction state and rollback if needed
            if self.connection.get_transaction_status() == psycopg2.extensions.TRANSACTION_STATUS_INERROR:
                print("🔄 UPDATE SESSION: Rolling back aborted transaction")
                self.connection.rollback()
            
            import json
            with self.connection.cursor() as cur:
                cur.execute("""
                    UPDATE interview_sessions SET 
                    conversation_history = %s,
                    current_question_index = %s,
                    is_complete = %s
                    WHERE session_id = %s;
                """, (json.dumps(conversation_history), question_index, is_complete, session_id))
                self.connection.commit()
                return True
        except Exception as e:
            print(f"❌ UPDATE SESSION ERROR: {e}")
            # Try to recover by reconnecting
            try:
                self.connection.rollback()
                self.connect()
            except:
                pass
            return False
    
    async def get_session(self, session_id: str):
        """Get session from database"""
        # Reconnect if connection is closed or in bad state
        if not self.connection or self.connection.closed:
            print("🔄 GET SESSION: Reconnecting to database")
            self.connect()
        
        if not self.connection:
            print(f"❌ GET SESSION ERROR: No database connection for session {session_id}")
            return None
        
        try:
            # Check if we're in a bad transaction state and rollback if needed
            if self.connection.get_transaction_status() == psycopg2.extensions.TRANSACTION_STATUS_INERROR:
                print("🔄 GET SESSION: Rolling back aborted transaction")
                self.connection.rollback()
            
            with self.connection.cursor() as cur:
                cur.execute("SELECT * FROM interview_sessions WHERE session_id = %s;", (session_id,))
                row = cur.fetchone()
                if row:
                    return {
                        'session_id': row[0],
                        'expert_name': row[1],
                        'expert_email': row[2],
                        'expertise_area': row[3],
                        'conversation_history': row[4] or [],
                        'current_question_index': row[5] or 0,
                        'is_complete': row[6] or False,
                        'created_at': row[7]
                    }
                return None
        except Exception as e:
            print(f"❌ GET SESSION ERROR: {e}")
            # Try to recover by reconnecting
            try:
                self.connect()
            except:
                pass
            return None
    
    async def get_all_sessions(self):
        """Get all sessions from database"""
        if not self.connection or self.connection.closed:
            self.connect()
        
        if not self.connection:
            return []
        
        try:
            with self.connection.cursor() as cur:
                cur.execute("SELECT * FROM interview_sessions ORDER BY created_at DESC;")
                rows = cur.fetchall()
                return [{
                    'session_id': row[0],
                    'expert_name': row[1],
                    'expert_email': row[2],
                    'expertise_area': row[3],
                    'conversation_history': row[4] or [],
                    'current_question_index': row[5] or 0,
                    'is_complete': row[6] or False,
                    'created_at': row[7]
                } for row in rows]
        except Exception as e:
            try:
                self.connect()
            except:
                pass
            return []
    
    async def get_max_session_id(self):
        """Get the highest session_id to avoid conflicts"""
        if not self.connection or self.connection.closed:
            self.connect()
        
        if not self.connection:
            return 0
        try:
            with self.connection.cursor() as cur:
                cur.execute("SELECT COALESCE(MAX(CAST(session_id AS INTEGER)), 0) FROM interview_sessions WHERE session_id ~ '^[0-9]+$';")
                result = cur.fetchone()
                return result[0] if result else 0
        except Exception as e:
            print(f"❌ GET MAX SESSION ID ERROR: {e}")
            return 0
    
    async def get_stats(self):
        """Get dashboard statistics."""
        print("📊 ASYNC STATS: Getting dashboard statistics")
        
        if not self.connection:
            print("❌ ASYNC STATS FAILED: No database connection")
            return {
                "total_interviews": 0,
                "pending_tasks": 0,
                "approved_tasks": 0,
                "rejected_tasks": 0
            }
        try:
            with self.connection.cursor() as cur:
                # Get total interviews (unique sessions)
                cur.execute("SELECT COUNT(DISTINCT session_id) FROM interview_rules;")
                total_interviews = cur.fetchone()[0] or 0
                
                # Get pending tasks
                cur.execute("SELECT COUNT(*) FROM interview_rules WHERE completed = FALSE;")
                pending_tasks = cur.fetchone()[0] or 0
                
                # Get approved tasks
                cur.execute("SELECT COUNT(*) FROM interview_rules WHERE completed = TRUE;")
                approved_tasks = cur.fetchone()[0] or 0
                
                stats = {
                    "total_interviews": total_interviews,
                    "pending_tasks": pending_tasks,
                    "approved_tasks": approved_tasks,
                    "rejected_tasks": 0  # Not tracking rejected separately
                }
                
                print(f"✅ ASYNC STATS SUCCESS: {stats}")
                return stats
        except Exception as e:
            print(f"❌ ASYNC STATS ERROR: {e}")
            return {
                "total_interviews": 0,
                "pending_tasks": 0,
                "approved_tasks": 0,
                "rejected_tasks": 0
            }
    
    def authenticate_admin(self, email: str, password: str):
        """Authenticate admin user against database"""
        print(f"🔐 DB AUTH: Checking credentials for {email}")
        
        if not self.connection or self.connection.closed:
            print("🔄 DB AUTH: Reconnecting to database")
            self.connect()
        
        if not self.connection:
            print("❌ DB AUTH: No database connection")
            return None
        
        try:
            with self.connection.cursor() as cur:
                cur.execute(
                    "SELECT password_hash, name FROM admin_users WHERE email = %s AND is_active = TRUE",
                    (email,)
                )
                result = cur.fetchone()
                
                if not result:
                    print(f"❌ DB AUTH: User {email} not found")
                    return None
                
                stored_hash, name = result
                
                # Verify password with bcrypt
                import bcrypt
                if bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8')):
                    print(f"✅ DB AUTH: Success for {email}")
                    return {
                        "email": email,
                        "name": name
                    }
                else:
                    print(f"❌ DB AUTH: Invalid password for {email}")
                    return None
                    
        except Exception as e:
            print(f"❌ DB AUTH ERROR: {e}")
            return None
    
    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            print("🔌 Database connection closed.")

# Global Supabase client instance
supabase_client = SupabaseClient()
supabase_client.connected = False
print("💾 Supabase client initialized")