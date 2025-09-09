-- Script to remove document tables from Supabase
-- Run this to clean up old document tables since we're now using ChromaDB only

-- Drop the tables in reverse dependency order
DROP TABLE IF EXISTS doc_chunks CASCADE;
DROP TABLE IF EXISTS documents CASCADE;

-- Also clean up any interview_rules columns that referenced documents
ALTER TABLE interview_rules 
DROP COLUMN IF EXISTS doc_id,
DROP COLUMN IF EXISTS source CASCADE;

-- Remove any indexes that might still exist
DROP INDEX IF EXISTS idx_documents_session_id;
DROP INDEX IF EXISTS idx_documents_expert_name;
DROP INDEX IF EXISTS idx_doc_chunks_doc_id;
DROP INDEX IF EXISTS idx_doc_chunks_embedding;
DROP INDEX IF EXISTS idx_rules_source;
DROP INDEX IF EXISTS idx_rules_doc_id;

PRINT '✅ Document tables and related columns removed successfully';
