import psycopg2
import psycopg2.extensions
import os
import socket
import json
from typing import List, Dict, Optional, Tuple
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
        self.connected = False
        self.documents_ready = False
        self.documents_error = None
    
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

            # Ensure document schema exists (idempotent). Try first; don't fail startup.
            try:
                self.ensure_document_schema()
                self.documents_ready = True
                self.documents_error = None
            except Exception as e:
                self.documents_ready = False
                self.documents_error = str(e)
                print(f"⚠️ Document schema not ready: {e}")
            
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
    
    # Document Management Methods
    async def create_document(self, expert_name: str, session_id: str, title: str, 
                            file_path: str, doc_type: str, file_size_bytes: int) -> int:
        """Create a new document record and return its ID"""
        try:
            with self.connection.cursor() as cur:
                cur.execute("""
                    INSERT INTO documents (expert_name, session_id, title, file_path, doc_type, file_size_bytes)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (expert_name, session_id, title, file_path, doc_type, file_size_bytes))
                
                doc_id = cur.fetchone()[0]
                self.connection.commit()
                return doc_id
        except Exception as e:
            print(f"❌ Error creating document: {e}")
            self.connection.rollback()
            raise
    
    async def update_document_status(self, doc_id: int, status: str):
        """Update document processing status"""
        try:
            with self.connection.cursor() as cur:
                cur.execute("""
                    UPDATE documents SET upload_status = %s, updated_at = NOW()
                    WHERE id = %s
                """, (status, doc_id))
                self.connection.commit()
        except Exception as e:
            print(f"❌ Error updating document status: {e}")
            self.connection.rollback()
            raise
    
    async def update_document_page_count(self, doc_id: int, page_count: int):
        """Update document page count"""
        try:
            with self.connection.cursor() as cur:
                cur.execute("""
                    UPDATE documents SET page_count = %s, updated_at = NOW()
                    WHERE id = %s
                """, (page_count, doc_id))
                self.connection.commit()
        except Exception as e:
            print(f"❌ Error updating document page count: {e}")
            self.connection.rollback()
            raise
    
    async def create_doc_chunk(self, doc_id: int, chunk_index: int, content: str, 
                             content_length: int, page_number: Optional[int] = None, 
                             slide_number: Optional[int] = None, embedding: List[float] = None) -> int:
        """Create a document chunk with embedding"""
        try:
            with self.connection.cursor() as cur:
                cur.execute("""
                    INSERT INTO doc_chunks (doc_id, chunk_index, content, content_length, 
                                          page_number, slide_number, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (doc_id, chunk_index, content, content_length, page_number, slide_number, embedding))
                
                chunk_id = cur.fetchone()[0]
                self.connection.commit()
                return chunk_id
        except Exception as e:
            print(f"❌ Error creating document chunk: {e}")
            self.connection.rollback()
            raise
    
    async def search_similar_chunks(self, query_embedding: List[float], 
                                  session_id: Optional[str] = None, limit: int = 5) -> List[Dict]:
        """Search for similar document chunks using vector similarity"""
        try:
            with self.connection.cursor() as cur:
                if session_id:
                    # Search within specific session's documents
                    cur.execute("""
                        SELECT dc.id, dc.content, dc.page_number, dc.slide_number,
                               d.title, d.doc_type, d.session_id,
                               1 - (dc.embedding <=> %s::vector) AS similarity
                        FROM doc_chunks dc
                        JOIN documents d ON dc.doc_id = d.id
                        WHERE d.session_id = %s AND d.upload_status = 'completed'
                        ORDER BY dc.embedding <=> %s::vector
                        LIMIT %s
                    """, (query_embedding, session_id, query_embedding, limit))
                else:
                    # Search across all documents
                    cur.execute("""
                        SELECT dc.id, dc.content, dc.page_number, dc.slide_number,
                               d.title, d.doc_type, d.session_id,
                               1 - (dc.embedding <=> %s::vector) AS similarity
                        FROM doc_chunks dc
                        JOIN documents d ON dc.doc_id = d.id
                        WHERE d.upload_status = 'completed'
                        ORDER BY dc.embedding <=> %s::vector
                        LIMIT %s
                    """, (query_embedding, query_embedding, limit))
                
                results = cur.fetchall()
                
                chunks = []
                for row in results:
                    chunks.append({
                        'id': row[0],
                        'content': row[1],
                        'page_number': row[2],
                        'slide_number': row[3],
                        'document_title': row[4],
                        'doc_type': row[5],
                        'session_id': row[6],
                        'similarity': float(row[7])
                    })
                
                return chunks
        except Exception as e:
            print(f"❌ Error searching similar chunks: {e}")
            return []
    
    async def get_documents_by_session(self, session_id: str) -> List[Dict]:
        """Get all documents for a session"""
        try:
            with self.connection.cursor() as cur:
                cur.execute("""
                    SELECT id, title, doc_type, file_size_bytes, page_count, 
                           upload_status, created_at
                    FROM documents
                    WHERE session_id = %s
                    ORDER BY created_at DESC
                """, (session_id,))
                
                results = cur.fetchall()
                
                documents = []
                for row in results:
                    documents.append({
                        'id': row[0],
                        'title': row[1],
                        'doc_type': row[2],
                        'file_size_bytes': row[3],
                        'page_count': row[4],
                        'upload_status': row[5],
                        'created_at': row[6].isoformat() if row[6] else None
                    })
                
                return documents
        except Exception as e:
            print(f"❌ Error getting documents: {e}")
            return []
    
    async def get_all_documents(self) -> List[Dict]:
        """Get all documents for admin panel"""
        try:
            with self.connection.cursor() as cur:
                cur.execute("""
                    SELECT d.id, d.title, d.doc_type, d.expert_name, d.session_id,
                           d.file_size_bytes, d.page_count, d.upload_status, d.created_at,
                           COUNT(dc.id) as chunk_count
                    FROM documents d
                    LEFT JOIN doc_chunks dc ON d.id = dc.doc_id
                    GROUP BY d.id, d.title, d.doc_type, d.expert_name, d.session_id,
                             d.file_size_bytes, d.page_count, d.upload_status, d.created_at
                    ORDER BY d.created_at DESC
                """)
                
                results = cur.fetchall()
                
                documents = []
                for row in results:
                    documents.append({
                        'id': row[0],
                        'title': row[1],
                        'doc_type': row[2],
                        'expert_name': row[3],
                        'session_id': row[4],
                        'file_size_bytes': row[5],
                        'page_count': row[6],
                        'upload_status': row[7],
                        'created_at': row[8].isoformat() if row[8] else None,
                        'chunk_count': row[9]
                    })
                
                return documents
        except Exception as e:
            print(f"❌ Error getting all documents: {e}")
            return []
    
    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            print("🔌 Database connection closed.")

    # -------------------- Optional Document Schema --------------------
    def ensure_document_schema(self):
        """Create pgvector extension and document tables if they do not exist."""
        if not self.connection or self.connection.closed:
            self.connect()
        if not self.connection:
            raise RuntimeError("No database connection")

        try:
            with self.connection.cursor() as cur:
                # Enable pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                # Documents table
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        id SERIAL PRIMARY KEY,
                        expert_name VARCHAR(255) NOT NULL,
                        session_id VARCHAR(100),
                        title VARCHAR(500) NOT NULL,
                        file_url TEXT,
                        file_path TEXT,
                        doc_type VARCHAR(50) NOT NULL,
                        file_size_bytes INTEGER,
                        page_count INTEGER,
                        upload_status VARCHAR(50) DEFAULT 'processing',
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    );
                    """
                )

                # Doc chunks table (1536 dims for text-embedding-3-small/ada-002)
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS doc_chunks (
                        id SERIAL PRIMARY KEY,
                        doc_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                        chunk_index INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        content_length INTEGER NOT NULL,
                        page_number INTEGER,
                        slide_number INTEGER,
                        embedding vector(1536),
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                    """
                )

                # Indexes
                cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_session_id ON documents(session_id);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_expert_name ON documents(expert_name);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_doc_chunks_doc_id ON doc_chunks(doc_id);")
                # Create ANN index (ivfflat supports <= 2000 dims)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_doc_chunks_embedding ON doc_chunks USING ivfflat (embedding vector_cosine_ops);")

                # Force correct embedding dimension - drop and recreate if needed
                try:
                    # Check current dimension
                    cur.execute("""
                        SELECT atttypmod 
                        FROM pg_attribute a
                        JOIN pg_class c ON a.attrelid = c.oid 
                        WHERE c.relname = 'doc_chunks' AND a.attname = 'embedding'
                    """)
                    result = cur.fetchone()
                    
                    if result and result[0] != 1536:
                        print(f"⚠️ doc_chunks has wrong embedding dimension ({result[0]}), recreating with 1536...")
                        # Clean up related documents that will lose their chunks
                        cur.execute("DELETE FROM documents WHERE upload_status = 'completed';")
                        # Drop table and recreate with correct dimensions
                        cur.execute("DROP TABLE IF EXISTS doc_chunks CASCADE;")
                        cur.execute(
                            """
                            CREATE TABLE doc_chunks (
                                id SERIAL PRIMARY KEY,
                                doc_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                                chunk_index INTEGER NOT NULL,
                                content TEXT NOT NULL,
                                content_length INTEGER NOT NULL,
                                page_number INTEGER,
                                slide_number INTEGER,
                                embedding vector(1536),
                                created_at TIMESTAMPTZ DEFAULT NOW()
                            );
                            """
                        )
                        cur.execute("CREATE INDEX IF NOT EXISTS idx_doc_chunks_doc_id ON doc_chunks(doc_id);")
                        cur.execute("CREATE INDEX IF NOT EXISTS idx_doc_chunks_embedding ON doc_chunks USING ivfflat (embedding vector_cosine_ops);")
                        print("✅ doc_chunks recreated with vector(1536)")
                except Exception as e:
                    print(f"⚠️ Could not verify/fix embedding dimension: {e}")

                self.connection.commit()
                print("✅ Document schema ensured (documents, doc_chunks)")
        except Exception as e:
            if self.connection:
                self.connection.rollback()
            # Fail fast: document feature is mandatory
            raise RuntimeError(f"Document schema setup failed: {e}")

# Global Supabase client instance
supabase_client = SupabaseClient()
supabase_client.connected = False
print("💾 Supabase client initialized")