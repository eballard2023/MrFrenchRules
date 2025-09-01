#!/usr/bin/env python3
"""
AI Coach Interview System - Startup Script
Simple script to start the FastAPI server
"""

import uvicorn

if __name__ == "__main__":
    print("🚀 Starting AI Coach Interview System...")
    print("📱 Open your browser and go to: http://localhost:8000")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 50)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
