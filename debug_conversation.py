#!/usr/bin/env python3
"""
Debug Conversation History Script
Use this to check if conversation history is being saved correctly
"""

import asyncio
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def debug_conversation():
    """Debug conversation history saving"""
    print("🔍 Debugging Conversation History...")
    print("=" * 50)
    
    # Get the base URL from environment or use default
    base_url = os.getenv("APP_URL", "http://localhost:8000")
    print(f"🌐 Testing against: {base_url}")
    
    try:
        # 1. Start a new interview
        print("\n1️⃣ Starting new interview...")
        response = requests.post(f"{base_url}/start_interview")
        if response.status_code == 200:
            data = response.json()
            session_id = data["session_id"]
            print(f"✅ Interview started - Session ID: {session_id}")
            print(f"   Initial message: {data['message'][:100]}...")
        else:
            print(f"❌ Failed to start interview: {response.status_code}")
            return
        
        # 2. Send a test message
        print("\n2️⃣ Sending test message...")
        test_message = "Hello, I'm a test user. I work in child psychology."
        response = requests.post(f"{base_url}/chat", json={
            "message": test_message,
            "session_id": session_id
        })
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Message sent successfully")
            print(f"   AI Response: {data['message'][:100]}...")
            print(f"   Question number: {data['question_number']}")
        else:
            print(f"❌ Failed to send message: {response.status_code}")
            print(f"   Error: {response.text}")
            return
        
        # 3. Check conversation history
        print("\n3️⃣ Checking conversation history...")
        response = requests.get(f"{base_url}/session/{session_id}/conversation")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Conversation retrieved - Source: {data.get('source', 'unknown')}")
            print(f"   Total messages: {len(data['conversation'])}")
            print(f"   Is complete: {data['is_complete']}")
            
            # Show each message
            for i, msg in enumerate(data['conversation']):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')[:80]
                print(f"   {i+1}. [{role.upper()}]: {content}...")
        else:
            print(f"❌ Failed to get conversation: {response.status_code}")
            print(f"   Error: {response.text}")
        
        # 4. Check database status
        print("\n4️⃣ Checking database status...")
        response = requests.get(f"{base_url}/database/status")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Database status: {data['message']}")
            print(f"   Connected: {data['connected']}")
            if data.get('collections'):
                print(f"   Interviews in DB: {data['collections']['interviews']}")
                print(f"   Rules in DB: {data['collections']['rules']}")
        else:
            print(f"❌ Failed to get database status: {response.status_code}")
        
        print("\n" + "=" * 50)
        print("🔍 Debug complete!")
        
    except Exception as e:
        print(f"❌ Error during debug: {e}")
        import traceback
        print(f"❌ Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(debug_conversation())
