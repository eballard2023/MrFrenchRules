import uvicorn
from main import app

if __name__ == "__main__":
    print("🚀 Starting AI Coach Interview System...")
    print("🌐 Binding to: 0.0.0.0:8003")
    print("🔗 Interview Interface: http://localhost:8003")
    print("🔗 Admin Panel: http://localhost:8003/admin")
    print("🔑 Admin Credentials: admin@aicoach.com / admin123")
    uvicorn.run(app, host="0.0.0.0", port=8003, log_level="info")