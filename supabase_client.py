import psycopg2
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
            return True
        except Exception as e:
            self.connected = False
            print(f"Error connecting to the database: {e}")
            return False
    
    def insert_rule(self, session_id: str, expert_name: str, expertise_area: str, rule_text: str):
        """Inserts a new rule into the interview_rules table."""
        print(f"🔄 DB INSERT: Adding rule for session {session_id}")
        print(f"   Expert: {expert_name}")
        print(f"   Area: {expertise_area}")
        print(f"   Rule: {rule_text[:100]}...")
        
        if not self.connection:
            print("❌ DB INSERT FAILED: No database connection")
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
                print(f"✅ DB INSERT SUCCESS: Rule saved with ID {rule_id}")
                return rule_id
        except (Exception, psycopg2.DatabaseError) as error:
            print(f"❌ DB INSERT ERROR: {error}")
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
        """Updates the completed status of a rule."""
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
    
    async def save_interview_rule(self, session_id: str, expert_name: str, expertise_area: str, rule_text: str):
        """Save interview rule to database."""
        print(f"💾 ASYNC SAVE: Saving rule for session {session_id}")
        result = self.insert_rule(session_id, expert_name, expertise_area, rule_text)
        if result:
            print(f"✅ ASYNC SAVE SUCCESS: Rule ID {result}")
        else:
            print("❌ ASYNC SAVE FAILED")
        return result
    
    async def get_all_rules(self):
        """Get all rules from database."""
        print("🔍 ASYNC GET: Fetching all rules")
        
        if not self.connection:
            print("❌ ASYNC GET FAILED: No database connection")
            return []
        try:
            with self.connection.cursor() as cur:
                cur.execute("SELECT * FROM interview_rules ORDER BY created_at DESC;")
                rows = cur.fetchall()
                # Convert to dict format
                rules = [{
                    'id': row[0],
                    'session_id': row[1], 
                    'expert_name': row[2],
                    'expertise_area': row[3],
                    'rule_text': row[4],
                    'completed': row[5],
                    'created_at': row[6]
                } for row in rows]
                print(f"✅ ASYNC GET SUCCESS: Retrieved {len(rules)} rules")
                return rules
        except Exception as e:
            print(f"❌ ASYNC GET ERROR: {e}")
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
                    'rule_text': row[4],
                    'completed': row[5],
                    'created_at': row[6]
                } for row in rows]
                print(f"✅ ASYNC GET SESSION SUCCESS: Found {len(rules)} rules for session {session_id}")
                return rules
        except Exception as e:
            print(f"❌ ASYNC GET SESSION ERROR: {e}")
            return []
    
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
    
    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            print("🔌 Database connection closed.")

# Global Supabase client instance
supabase_client = SupabaseClient()
supabase_client.connected = False
print("💾 Supabase client initialized")