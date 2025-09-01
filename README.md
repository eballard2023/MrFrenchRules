# AI Coach Interview System

A FastAPI-based chatbot system that conducts structured interviews with behavioral experts to extract actionable rules and best practices. The system converts conversational responses into structured JSON format for AI coaching systems.

## 🎯 Purpose

This system conducts comprehensive interviews with subject matter experts (SMEs) to extract behavioral rules, processes, and best practices. The conversational responses are then converted into structured JSON format that can be used as a "super-script" for AI coaching systems like Mr. French.

## ✨ Features

- **Intelligent Interviewing**: AI conducts natural, one-question-at-a-time interviews
- **Structured Rule Extraction**: Converts conversations into actionable JSON rules
- **MongoDB Integration**: Persistent storage of interviews and extracted rules
- **Modern Web Interface**: Clean, responsive chat interface
- **Background Processing**: Non-blocking rule extraction for better performance

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- MongoDB instance
- OpenAI API key

### 1. Environment Setup

Create a `.env` file with your configuration:

```bash
OPENAI_API_KEY=your_openai_api_key_here
MONGODB_CONNECTION_STRING=mongodb://localhost:27017
MONGODB_DATABASE_NAME=ai_coach_interviews
MONGODB_INTERVIEWS_COLLECTION=interviews
MONGODB_RULES_COLLECTION=rules
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the Application

```bash
python run.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Access the Interface

Open your browser and navigate to: `http://localhost:8000`

## 🏗️ System Architecture

### Core Components

1. **FastAPI Backend** (`main.py`)
   - Interview session management
   - OpenAI GPT-4o mini integration
   - Rule extraction and processing
   - RESTful API endpoints

2. **Database Layer** (`database.py`)
   - MongoDB connection management
   - Data models and validation
   - CRUD operations for interviews and rules

3. **Web Interface** (`templates/interview.html`)
   - Modern, responsive chat interface
   - Real-time conversation with AI interviewer
   - Session management and status updates

### Interview Flow

1. **Session Creation**: Generate unique session ID and initialize interview
2. **Conversation**: AI asks structured questions one at a time
3. **Response Processing**: Expert responses are stored and analyzed
4. **Rule Extraction**: Background processing converts conversations to JSON rules
5. **Storage**: Rules saved to both local files and MongoDB

## 📊 Data Models

### Interview Session
```json
{
  "session_id": "1",
  "started_at": "2025-08-29T21:19:30.364+00:00",
  "completed_at": "2025-08-29T21:25:15.123+00:00",
  "conversation_history": [...],
  "questions_asked": 23,
  "is_complete": true,
  "status": "completed"
}
```

### Extracted Rules
```json
{
  "if": {
    "event": "frustrated_with_homework",
    "context": "child is struggling with academic work",
    "user_type": "child"
  },
  "then": {
    "action": "suggest_break",
    "response": "Let's take a 5-minute break to clear your mind",
    "duration": "5_minutes",
    "tone": "calm"
  },
  "priority": "high",
  "category": "crisis_management"
}
```

## 🔌 API Endpoints

- `GET /` - Main interview interface
- `POST /start_interview` - Start new interview session
- `POST /chat_with_interviewer` - Send message to AI interviewer
- `POST /submit_interview` - Complete interview and extract rules
- `GET /database/status` - Check MongoDB connection
- `GET /database/rules` - Retrieve all extracted rules
- `GET /database/rules/session/{session_id}` - Get rules for specific session

## 📁 Project Structure

```
├── main.py                 # FastAPI application and core logic
├── database.py            # Database models and operations
├── requirements.txt       # Python dependencies
├── run.py              # Startup script
├── .gitignore            # Git ignore rules
├── README.md             # This file
├── example_mr_french_rules.json  # Example rule format
└── templates/
    └── interview.html    # Web interface template
```
