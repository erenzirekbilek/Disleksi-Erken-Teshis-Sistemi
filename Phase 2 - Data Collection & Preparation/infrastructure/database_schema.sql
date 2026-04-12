"""
PostgreSQL Database Schema for Dyslexia Early Detection System
"""

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================
-- STUDENTS TABLE (Anonymized)
-- ============================================
CREATE TABLE students (
    student_hash VARCHAR(64) PRIMARY KEY,
    age INTEGER CHECK (age >= 5 AND age <= 18),
    grade VARCHAR(20),
    gender VARCHAR(20),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    
    CONSTRAINT valid_grade CHECK (grade IN ('K', '1', '2', '3', '4', '5', '6', '7', '8'))
);

COMMENT ON TABLE students IS 'Anonymized student records with hash-based IDs';
COMMENT ON COLUMN students.student_hash IS 'SHA-256 hash of original student ID';

-- Index for fast lookups
CREATE INDEX idx_students_age ON students(age);
CREATE INDEX idx_students_grade ON students(grade);
CREATE INDEX idx_students_active ON students(is_active) WHERE is_active = TRUE;

-- ============================================
-- SPEECH SAMPLES TABLE
-- ============================================
CREATE TABLE speech_samples (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    student_hash VARCHAR(64) NOT NULL REFERENCES students(student_hash) ON DELETE CASCADE,
    audio_path VARCHAR(500) NOT NULL,
    audio_format VARCHAR(20) DEFAULT 'wav',
    duration_seconds FLOAT,
    passage_id VARCHAR(20) NOT NULL,
    recording_date DATE,
    recording_quality VARCHAR(20) DEFAULT 'good',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT valid_format CHECK (audio_format IN ('wav', 'mp3', 'flac', 'ogg')),
    CONSTRAINT valid_quality CHECK (recording_quality IN ('excellent', 'good', 'fair', 'poor'))
);

COMMENT ON TABLE speech_samples IS 'Speech audio samples for analysis';

CREATE INDEX idx_speech_student ON speech_samples(student_hash);
CREATE INDEX idx_speech_date ON speech_samples(recording_date);
CREATE INDEX idx_speech_passage ON speech_samples(passage_id);

-- ============================================
-- SPEECH LABELS TABLE
-- ============================================
CREATE TABLE speech_labels (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    sample_id UUID NOT NULL REFERENCES speech_samples(id) ON DELETE CASCADE,
    
    phonological VARCHAR(20) NOT NULL,
    phonological_notes TEXT,
    
    fluency_wpm FLOAT,
    fluency_repetitions INTEGER,
    fluency VARCHAR(20) NOT NULL,
    fluency_notes TEXT,
    
    pronunciation VARCHAR(20) NOT NULL,
    pronunciation_notes TEXT,
    
    overall_risk VARCHAR(20) NOT NULL,
    overall_score FLOAT NOT NULL,
    
    annotator_id VARCHAR(100),
    annotation_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_verified BOOLEAN DEFAULT FALSE,
    verification_date TIMESTAMP WITH TIME ZONE,
    
    CONSTRAINT valid_phonological CHECK (phonological IN ('normal', 'mild', 'moderate', 'severe')),
    CONSTRAINT valid_fluency CHECK (fluency IN ('normal', 'mild', 'moderate', 'severe')),
    CONSTRAINT valid_pronunciation CHECK (pronunciation IN ('normal', 'mild', 'moderate', 'severe')),
    CONSTRAINT valid_overall CHECK (overall_risk IN ('low', 'medium', 'high')),
    CONSTRAINT valid_score CHECK (overall_score >= 0 AND overall_score <= 1)
);

COMMENT ON TABLE speech_labels IS 'Labels for speech samples';

CREATE INDEX idx_speech_labels_sample ON speech_labels(sample_id);
CREATE INDEX idx_speech_labels_risk ON speech_labels(overall_risk);

-- ============================================
-- HANDWRITING SAMPLES TABLE
-- ============================================
CREATE TABLE handwriting_samples (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    student_hash VARCHAR(64) NOT NULL REFERENCES students(student_hash) ON DELETE CASCADE,
    image_path VARCHAR(500) NOT NULL,
    image_format VARCHAR(20) DEFAULT 'png',
    task_type VARCHAR(50) NOT NULL,
    dpi INTEGER,
    writing_date DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT valid_task CHECK (task_type IN ('copying', 'free_writing', 'dictation', 'essay'))
);

COMMENT ON TABLE handwriting_samples IS 'Handwriting image samples for analysis';

CREATE INDEX idx_handwriting_student ON handwriting_samples(student_hash);
CREATE INDEX idx_handwriting_task ON handwriting_samples(task_type);
CREATE INDEX idx_handwriting_date ON handwriting_samples(writing_date);

-- ============================================
-- HANDWRITING LABELS TABLE
-- ============================================
CREATE TABLE handwriting_labels (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    sample_id UUID NOT NULL REFERENCES handwriting_samples(id) ON DELETE CASCADE,
    
    letter_reversals VARCHAR(20) NOT NULL,
    letter_reversals_boxes JSONB,
    letter_reversals_notes TEXT,
    
    spacing_irregularity VARCHAR(20) NOT NULL,
    spacing_boxes JSONB,
    spacing_notes TEXT,
    
    character_misplacement VARCHAR(20) NOT NULL,
    misplacement_boxes JSONB,
    misplacement_notes TEXT,
    
    baseline_adherence VARCHAR(20) NOT NULL,
    baseline_notes TEXT,
    
    size_consistency VARCHAR(20) NOT NULL,
    size_notes TEXT,
    
    overall_risk VARCHAR(20) NOT NULL,
    overall_score FLOAT NOT NULL,
    
    annotator_id VARCHAR(100),
    annotation_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_verified BOOLEAN DEFAULT FALSE,
    
    CONSTRAINT valid_severity CHECK (letter_reversals IN ('none', 'mild', 'moderate', 'severe')),
    CONSTRAINT valid_overall CHECK (overall_risk IN ('low', 'medium', 'high'))
);

COMMENT ON TABLE handwriting_labels IS 'Labels for handwriting samples';

CREATE INDEX idx_handwriting_labels_sample ON handwriting_labels(sample_id);

-- ============================================
-- TEXT SAMPLES TABLE
-- ============================================
CREATE TABLE text_samples (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    student_hash VARCHAR(64) NOT NULL REFERENCES students(student_hash) ON DELETE CASCADE,
    text_content TEXT NOT NULL,
    prompt_id VARCHAR(20) NOT NULL,
    word_count INTEGER,
    writing_date DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT positive_words CHECK (word_count >= 0)
);

COMMENT ON TABLE text_samples IS 'Text samples for analysis';

CREATE INDEX idx_text_student ON text_samples(student_hash);
CREATE INDEX idx_text_prompt ON text_samples(prompt_id);
CREATE INDEX idx_text_words ON text_samples(word_count);

-- ============================================
-- TEXT LABELS TABLE
-- ============================================
CREATE TABLE text_labels (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    sample_id UUID NOT NULL REFERENCES text_samples(id) ON DELETE CASCADE,
    
    spelling_error_rate FLOAT NOT NULL,
    spelling VARCHAR(20) NOT NULL,
    spelling_notes TEXT,
    
    grammar_score FLOAT NOT NULL,
    grammar VARCHAR(20) NOT NULL,
    grammar_notes TEXT,
    
    sentence_complexity VARCHAR(20) NOT NULL,
    complexity_score FLOAT NOT NULL,
    complexity_notes TEXT,
    
    flesch_reading_ease FLOAT NOT NULL,
    reading_ease VARCHAR(20) NOT NULL,
    reading_ease_notes TEXT,
    
    vocabulary_diversity FLOAT NOT NULL,
    vocabulary VARCHAR(20) NOT NULL,
    vocabulary_notes TEXT,
    
    overall_risk VARCHAR(20) NOT NULL,
    overall_score FLOAT NOT NULL,
    
    annotator_id VARCHAR(100),
    annotation_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_verified BOOLEAN DEFAULT FALSE,
    
    CONSTRAINT valid_spelling CHECK (spelling IN ('normal', 'mild', 'moderate', 'severe')),
    CONSTRAINT valid_grammar CHECK (grammar IN ('normal', 'mild', 'moderate', 'severe')),
    CONSTRAINT valid_overall CHECK (overall_risk IN ('low', 'medium', 'high'))
);

COMMENT ON TABLE text_labels IS 'Labels for text samples';

CREATE INDEX idx_text_labels_sample ON text_labels(sample_id);

-- ============================================
-- ANALYSIS RESULTS TABLE
-- ============================================
CREATE TABLE analysis_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    student_hash VARCHAR(64) NOT NULL REFERENCES students(student_hash) ON DELETE CASCADE,
    
    -- Modality scores
    speech_score FLOAT,
    handwriting_score FLOAT,
    text_score FLOAT,
    
    -- Individual scores
    speech_risk VARCHAR(20),
    handwriting_risk VARCHAR(20),
    text_risk VARCHAR(20),
    
    -- Combined result
    combined_score FLOAT NOT NULL,
    combined_risk VARCHAR(20) NOT NULL,
    
    -- Detailed features
    speech_features JSONB,
    handwriting_features JSONB,
    text_features JSONB,
    
    -- Model info
    model_version VARCHAR(50),
    analysis_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT valid_risk CHECK (combined_risk IN ('low', 'medium', 'high')),
    CONSTRAINT valid_score CHECK (combined_score >= 0 AND combined_score <= 1)
);

COMMENT ON TABLE analysis_results IS 'Combined analysis results for students';

CREATE INDEX idx_results_student ON analysis_results(student_hash);
CREATE INDEX idx_results_risk ON analysis_results(combined_risk);
CREATE INDEX idx_results_date ON analysis_results(analysis_date);

-- ============================================
-- AUDIT LOG TABLE
-- ============================================
CREATE TABLE audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(100),
    action VARCHAR(100) NOT NULL,
    target_type VARCHAR(50),
    target_hash VARCHAR(64),
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT valid_action CHECK (action IN (
        'CREATE', 'READ', 'UPDATE', 'DELETE', 'EXPORT', 'LOGIN', 'LOGOUT'
    ))
);

COMMENT ON TABLE audit_log IS 'Audit trail for all data access and modifications';

CREATE INDEX idx_audit_user ON audit_log(user_id);
CREATE INDEX idx_audit_target ON audit_log(target_hash);
CREATE INDEX idx_audit_date ON audit_log(created_at);
CREATE INDEX idx_audit_action ON audit_log(action);

-- ============================================
-- USER ACCOUNTS TABLE
-- ============================================
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL DEFAULT 'teacher',
    full_name VARCHAR(255),
    organization VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    last_login TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT valid_role CHECK (role IN ('admin', 'teacher', 'parent', 'psychologist'))
);

COMMENT ON TABLE users IS 'User accounts for system access';

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_role ON users(role);
CREATE INDEX idx_users_active ON users(is_active) WHERE is_active = TRUE;

-- ============================================
-- USER STUDENT MAPPING (for parent access)
-- ============================================
CREATE TABLE user_student_access (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    student_hash VARCHAR(64) NOT NULL REFERENCES students(student_hash) ON DELETE CASCADE,
    access_level VARCHAR(20) DEFAULT 'read',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT valid_access CHECK (access_level IN ('read', 'write', 'admin')),
    UNIQUE(user_id, student_hash)
);

COMMENT ON TABLE user_student_access IS 'Maps users to students they can access';

CREATE INDEX idx_access_user ON user_student_access(user_id);
CREATE INDEX idx_access_student ON user_student_access(student_hash);

-- ============================================
-- SCHOOL / INSTITUTION TABLE
-- ============================================
CREATE TABLE institutions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    code VARCHAR(50) UNIQUE NOT NULL,
    address TEXT,
    contact_email VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE institutions IS 'Educational institutions/schools';

CREATE INDEX idx_institutions_code ON institutions(code);

-- ============================================
-- INSTITUTION STUDENT LINK
-- ============================================
CREATE TABLE institution_students (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    institution_id UUID NOT NULL REFERENCES institutions(id) ON DELETE CASCADE,
    student_hash VARCHAR(64) NOT NULL REFERENCES students(student_hash) ON DELETE CASCADE,
    enrolled_date DATE,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT valid_status CHECK (status IN ('active', 'graduated', 'transferred', 'withdrawn')),
    UNIQUE(institution_id, student_hash)
);

CREATE INDEX idx_inst_students_inst ON institution_students(institution_id);
CREATE INDEX idx_inst_students_student ON institution_students(student_hash);

-- ============================================
-- FUNCTIONS AND TRIGGERS
-- ============================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for students
CREATE TRIGGER update_students_timestamp
    BEFORE UPDATE ON students
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- Trigger for users
CREATE TRIGGER update_users_timestamp
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- Function to log data access
CREATE OR REPLACE FUNCTION log_data_access(
    p_user_id VARCHAR,
    p_action VARCHAR,
    p_target_type VARCHAR,
    p_target_hash VARCHAR,
    p_details JSONB DEFAULT NULL
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO audit_log (user_id, action, target_type, target_hash, details)
    VALUES (p_user_id, p_action, p_target_type, p_target_hash, p_details);
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- VIEWS
-- ============================================

-- View for student summary with all modality scores
CREATE OR REPLACE VIEW v_student_summary AS
SELECT 
    s.student_hash,
    s.age,
    s.grade,
    s.created_at as enrolled_date,
    COALESCE(sl.overall_score, 0) as speech_score,
    COALESCE(hl.overall_score, 0) as handwriting_score,
    COALESCE(tl.overall_score, 0) as text_score,
    COALESCE(sl.overall_risk, 'low') as speech_risk,
    COALESCE(hl.overall_risk, 'low') as handwriting_risk,
    COALESCE(tl.overall_risk, 'low') as text_risk
FROM students s
LEFT JOIN (
    SELECT student_hash, overall_score, overall_risk, 
           ROW_NUMBER() OVER (PARTITION BY student_hash ORDER BY created_at DESC) as rn
    FROM speech_labels
) sl ON s.student_hash = sl.student_hash AND sl.rn = 1
LEFT JOIN (
    SELECT student_hash, overall_score, overall_risk,
           ROW_NUMBER() OVER (PARTITION BY student_hash ORDER BY created_at DESC) as rn
    FROM handwriting_labels
) hl ON s.student_hash = hl.student_hash AND hl.rn = 1
LEFT JOIN (
    SELECT student_hash, overall_score, overall_risk,
           ROW_NUMBER() OVER (PARTITION BY student_hash ORDER BY created_at DESC) as rn
    FROM text_labels
) tl ON s.student_hash = tl.student_hash AND tl.rn = 1
WHERE s.is_active = TRUE;

-- View for risk distribution
CREATE OR REPLACE VIEW v_risk_distribution AS
SELECT 
    combined_risk,
    COUNT(*) as count,
    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() as percentage
FROM analysis_results
GROUP BY combined_risk;

-- ============================================
-- ROW LEVEL SECURITY (Optional)
-- ============================================

-- Enable RLS on sensitive tables
ALTER TABLE students ENABLE ROW LEVEL SECURITY;
ALTER TABLE speech_samples ENABLE ROW LEVEL SECURITY;
ALTER TABLE handwriting_samples ENABLE ROW LEVEL SECURITY;
ALTER TABLE text_samples ENABLE ROW LEVEL SECURITY;
ALTER TABLE analysis_results ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only see their accessible students
CREATE POLICY access_own_students ON students
    USING (student_hash IN (
        SELECT student_hash FROM user_student_access WHERE user_id = current_user_id()
    ));

-- ============================================
-- PERMISSIONS
-- ============================================

-- Grant standard permissions
GRANT SELECT ON ALL TABLES IN SCHEMA public TO dyslexia_app;
GRANT INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO dyslexia_app;

-- Grant usage on sequences
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO dyslexia_app;

-- ============================================
-- METADATA
-- ============================================

COMMENT ON SCHEMA public IS 'Dyslexia Early Detection System Database Schema';
COMMENT ON COLUMN students.student_hash IS 'Anonymized student identifier (SHA-256)';

-- Version: 1.0
-- Last Updated: 2026-04-12