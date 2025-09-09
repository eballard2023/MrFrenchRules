# AI Coach Interview System

A FastAPI-based chatbot system that conducts structured interviews with behavioral experts to extract actionable rules and best practices. The system converts conversational responses into structured rules for AI coaching systems like Mr. French.

## 🎯 Purpose

This system conducts comprehensive interviews with subject matter experts (SMEs) to extract behavioral rules, processes, and best practices. The conversational responses and uploaded documents are then converted into actionable rules that can be used to train Mr. French, the conversational AI family assistant.

## ✨ Features

- **Intelligent Interviewing**: AI conducts natural, conversational interviews
- **Document Integration**: Upload research papers, PDFs, and documents for AI context
- **Structured Rule Extraction**: Converts conversations and documents into actionable rules
- **Supabase Integration**: PostgreSQL database for interviews and rules storage
- **ChromaDB Vector Storage**: Document embeddings and conversation history
- **Admin Dashboard**: Review and approve extracted rules
- **Modern Web Interface**: Clean, responsive chat interface with document upload

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Supabase account and database
- OpenAI API key

### 1. Environment Setup

Create a `.env` file with your configuration:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Supabase Database Configuration
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
user=your_db_user
password=your_db_password
host=your_db_host
port=5432
dbname=your_db_name

# Admin Configuration
ADMIN_EMAIL=your_admin_email
ADMIN_PASSWORD=your_admin_password
JWT_SECRET_KEY=your_jwt_secret
PASSWORD_SALT=your_password_salt

# Environment
ENV=development
PORT=8000
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the Application

**Development:**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Production:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 4. Access the Interface

Open your browser and navigate to: `http://localhost:8000`

## 🏗️ System Architecture

### Core Components

1. **FastAPI Backend** (`main.py`)
   - Interview session management
   - OpenAI GPT-4o mini integration
   - Document processing and embedding
   - Rule extraction from conversations and documents
   - RESTful API endpoints

2. **Database Layer** (`supabase_client.py`)
   - PostgreSQL connection pooling
   - Interview sessions and rules storage
   - Admin authentication and management

3. **Vector Database** (`chroma_client.py`)
   - Document embeddings storage
   - Conversation history management
   - Similarity search capabilities

4. **Document Processing** (`document_processor.py`)
   - PDF, DOCX, PPTX, TXT file parsing
   - Text chunking and embedding generation
   - Integration with ChromaDB

5. **Web Interface** (`templates/`)
   - Modern, responsive chat interface
   - Document upload functionality
   - Admin dashboard for rule management

### Interview Flow

1. **Session Creation**: Generate unique session ID and initialize interview
2. **Document Upload**: Experts can upload research papers and documents
3. **Conversation**: AI conducts natural interviews with document context
4. **Response Processing**: Expert responses and document content are stored
5. **Rule Extraction**: AI extracts rules from both conversation and documents
6. **Storage**: Rules saved to Supabase, documents to ChromaDB

## 📊 Data Models

### Interview Session
```json
{
  "session_id": "unique_session_id",
  "expert_name": "Dr. Smith",
  "started_at": "2025-01-15T10:30:00Z",
  "completed_at": "2025-01-15T11:00:00Z",
  "conversation_history": [...],
  "current_question_index": 15,
  "is_complete": true
}
```

### Extracted Rules
```json
{
  "rule_text": "Mr. French should use calm, reassuring language when a child is in the red zone",
  "expert_name": "Dr. Smith",
  "session_id": "unique_session_id",
  "created_at": "2025-01-15T11:00:00Z",
  "is_approved": false
}
```

### Document Chunks
```json
{
  "id": "chunk_id",
  "session_id": "unique_session_id",
  "title": "Behavioral Strategies.pdf",
  "content": "Positive reinforcement works best when...",
  "chunk_index": 0,
  "embedding": [0.1, 0.2, ...]
}
```

## 🔌 API Endpoints

### Interview Endpoints
- `GET /` - Main interview interface
- `POST /start_interview` - Start new interview session
- `POST /chat_with_interviewer` - Send message to AI interviewer
- `POST /submit_interview/{session_id}` - Complete interview and extract rules

### Document Endpoints
- `POST /upload-doc` - Upload document (PDF, DOCX, PPTX, TXT)
- `GET /documents/{session_id}` - Get documents for session

### Admin Endpoints
- `GET /admin/login` - Admin login page
- `POST /admin/login` - Admin authentication
- `GET /admin/dashboard` - Admin dashboard
- `POST /admin/approve-rule/{rule_id}` - Approve rule
- `POST /admin/reject-rule/{rule_id}` - Reject rule

### Database Endpoints
- `GET /database/status` - Check database connection
- `GET /database/rules` - Retrieve all extracted rules
- `GET /database/rules/session/{session_id}` - Get rules for specific session

## 📁 Project Structure

```
├── main.py                    # FastAPI application and core logic
├── supabase_client.py         # PostgreSQL database operations
├── chroma_client.py           # ChromaDB vector database operations
├── document_processor.py      # Document parsing and processing
├── admin_auth.py              # Admin authentication logic
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (create this)
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
├── MrFrench.png               # Favicon
└── templates/
    ├── interview.html         # Main interview interface
    ├── admin_login.html       # Admin login page
    ├── admin_dashboard.html   # Admin dashboard
    └── start.html             # Landing page
```

## 🚀 Deployment

### Development
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Environment Variables
Make sure to set `ENV=production` in your `.env` file for production deployment to disable debug logs and FastAPI docs.
