-- Enable pgvector extension for embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table - stores metadata about uploaded files
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    expert_name VARCHAR(255) NOT NULL,
    session_id VARCHAR(100),
    title VARCHAR(500) NOT NULL,
    file_url TEXT,
    file_path TEXT,
    doc_type VARCHAR(50) NOT NULL, -- 'pdf', 'docx', 'pptx', 'txt'
    file_size_bytes INTEGER,
    page_count INTEGER,
    upload_status VARCHAR(50) DEFAULT 'processing', -- 'processing', 'completed', 'error'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Document chunks table - stores text chunks with embeddings
CREATE TABLE IF NOT EXISTS doc_chunks (
    id SERIAL PRIMARY KEY,
    doc_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL, -- order within the document
    content TEXT NOT NULL,
    content_length INTEGER NOT NULL,
    page_number INTEGER, -- for PDFs/PPTX
    slide_number INTEGER, -- for PPTX
    embedding vector(1536), -- OpenAI 1536-dim models (3-small / ada-002)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Force correct embedding dimension - recreate table if needed
DO $$ 
DECLARE
    current_dim INTEGER;
BEGIN
    -- Check if doc_chunks exists and get its embedding dimension
    SELECT atttypmod INTO current_dim
    FROM pg_attribute a
    JOIN pg_class c ON a.attrelid = c.oid 
    WHERE c.relname = 'doc_chunks' AND a.attname = 'embedding';
    
    -- If wrong dimension, drop and recreate
    IF current_dim IS NOT NULL AND current_dim != 1536 THEN
        RAISE NOTICE 'doc_chunks has wrong embedding dimension (%), recreating with 1536...', current_dim;
        
        DROP TABLE IF EXISTS doc_chunks CASCADE;
        
        CREATE TABLE doc_chunks (
            id SERIAL PRIMARY KEY,
            doc_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            content_length INTEGER NOT NULL,
            page_number INTEGER,
            slide_number INTEGER,
            embedding vector(1536),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        -- Recreate indexes
        CREATE INDEX IF NOT EXISTS idx_doc_chunks_doc_id ON doc_chunks(doc_id);
        CREATE INDEX IF NOT EXISTS idx_doc_chunks_embedding ON doc_chunks USING ivfflat (embedding vector_cosine_ops);
        
        RAISE NOTICE 'doc_chunks recreated with vector(1536)';
    END IF;
    
EXCEPTION WHEN others THEN
    -- Continue if there are any issues
    RAISE NOTICE 'Could not verify/fix embedding dimension: %', SQLERRM;
END $$;

-- Update rules table to track document sources
ALTER TABLE interview_rules 
ADD COLUMN IF NOT EXISTS source VARCHAR(50) DEFAULT 'conversation', -- 'conversation', 'document', 'both'
ADD COLUMN IF NOT EXISTS doc_id INTEGER REFERENCES documents(id) ON DELETE SET NULL;

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_documents_session_id ON documents(session_id);
CREATE INDEX IF NOT EXISTS idx_documents_expert_name ON documents(expert_name);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_doc_id ON doc_chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_embedding ON doc_chunks USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_rules_source ON interview_rules(source);
CREATE INDEX IF NOT EXISTS idx_rules_doc_id ON interview_rules(doc_id);

-- Update function for updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for documents table
DROP TRIGGER IF EXISTS update_documents_updated_at ON documents;
CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
