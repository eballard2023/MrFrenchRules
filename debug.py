from supabase_client import supabase_client
import bcrypt

supabase_client.connect()

# 1) Read current hash
conn = supabase_client.connection_pool.getconn()
cur = conn.cursor()
cur.execute("SELECT email, is_active, password_hash FROM admin_users WHERE email=%s", ("admin@coachai.com",))
row = cur.fetchone()
print("Row:", row)
supabase_client.connection_pool.putconn(conn)

# 2) Check bcrypt match
stored_hash = row[2] if row else None
print("Bcrypt matches?:", bool(stored_hash and bcrypt.checkpw("admin123".encode("utf-8"), stored_hash.encode("utf-8"))))