import socket
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

DB_HOST = os.getenv("SUPABASE_HOST")
DB_NAME = os.getenv("SUPABASE_DB_NAME")
DB_USER = os.getenv("SUPABASE_USER")
DB_PASSWORD = os.getenv("SUPABASE_PASSWORD")
DB_PORT = os.getenv("SUPABASE_PORT", "5432")

print(f"Testing connection to: {DB_HOST}")

# Test 1: Basic socket connection
try:
    print("\n1. Testing socket connection...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10)
    result = sock.connect_ex((DB_HOST, int(DB_PORT)))
    if result == 0:
        print("✅ Socket connection successful")
    else:
        print(f"❌ Socket connection failed: {result}")
    sock.close()
except Exception as e:
    print(f"❌ Socket test failed: {e}")

# Test 2: DNS resolution
try:
    print("\n2. Testing DNS resolution...")
    addr_info = socket.getaddrinfo(DB_HOST, DB_PORT, socket.AF_INET)
    ipv4_addr = addr_info[0][4][0]
    print(f"✅ Resolved to IPv4: {ipv4_addr}")
except Exception as e:
    print(f"❌ DNS resolution failed: {e}")

# Test 3: Direct psycopg2 connection
try:
    print("\n3. Testing psycopg2 connection...")
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT,
        sslmode='require',
        connect_timeout=10
    )
    print("✅ psycopg2 connection successful")
    conn.close()
except Exception as e:
    print(f"❌ psycopg2 connection failed: {e}")

# Test 4: Connection string method
try:
    print("\n4. Testing connection string...")
    conn_string = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=require"
    conn = psycopg2.connect(conn_string)
    print("✅ Connection string method successful")
    conn.close()
except Exception as e:
    print(f"❌ Connection string failed: {e}")