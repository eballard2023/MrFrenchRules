-- Create the interview_rules table
CREATE TABLE IF NOT EXISTS interview_rules (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(50) NOT NULL,
    expert_name VARCHAR(255) DEFAULT 'Expert User',
    expertise_area VARCHAR(255) DEFAULT 'General',
    rule_text TEXT NOT NULL,
    completed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);