# Phase 2: Data Collection & Preparation

**Duration:** 4-6 weeks  
**Start:** Week 3  
**End:** Week 8  
**Dependencies:** Phase 1 Complete  
**Key Stakeholders:** Data Engineering Lead, Clinical Advisor, Data Collection Coordinator

---

## 2.1 Overview

Phase 2 focuses on building a clean, labeled, multi-modal dataset. This phase is critical as model performance depends directly on data quality.

**Goals:**
- Collect 500+ samples per modality (speech, handwriting, text)
- Apply preprocessing pipelines for each data type
- Establish data infrastructure (storage, versioning, anonymization)

---

## 2.2 Week 3-5: Speech Data Collection

### Task 2.1.1 — Design Speech Collection Protocol

**Description:** Create standardized procedures for speech data collection.

**Specifications:**

| Parameter | Value |
|-----------|-------|
| Audio Format | WAV / MP3 |
| Sample Rate | 48 kHz |
| Bit Depth | 16-bit |
| Noise Floor | < -40 dB |
| Environment | Quiet room, controlled acoustics |

**Reading Passages:**
- 3 difficulty levels (Easy, Medium, Hard)
- Age-appropriate content
- Standardized prompts across all participants

**Deliverables:**
- Speech collection protocol document
- Standardized reading passage set

**Owner:** Data Collection Coordinator  
**Due:** Week 3

---

### Task 2.1.2 — Execute Speech Recording Sessions

**Description:** Conduct speech recording sessions with target demographic.

**Target:**
- 500+ samples across target age groups (5-12)
- Balanced representation across age, gender, socioeconomic status
- Diverse accent/dialect representation (if applicable)

**Recording Setup:**
- Professional microphone
- Pop filter
- Acoustic treatment
- Real-time monitoring

**Metadata to Capture:**
- Age, grade level
- Native language/dialect
- Recording date/time
- Reading passage ID
- Duration

**Deliverables:**
- Raw audio files (500+)
- Metadata spreadsheet

**Owner:** Data Collection Coordinator  
**Due:** Week 4

---

### Task 2.1.3 — Apply Audio Preprocessing

**Description:** Clean and normalize audio recordings.

**Preprocessing Pipeline:**

1. **Noise Reduction**
   - Spectral Gating / Wiener Filter
   - Manual review for residual noise

2. **Voice Activity Detection (VAD)**
   - Trim silence from beginning/end
   - Preserve meaningful pauses

3. **Audio Normalization**
   - Target: -20 dB LUFS
   - Peak normalization to -1 dB

4. **Format Standardization**
   - Convert all to WAV (48kHz, 16-bit)
   - Metadata embedding

**Commands (librosa example):**
```python
import librosa
import numpy as np

def preprocess_audio(audio_path):
    y, sr = librosa.load(audio_path)
    # Noise reduction
    y = noise_reduction(y)
    # VAD
    y, _ = librosa.effects.trim(y, top_db=20)
    # Normalize
    y = librosa.util.normalize(y)
    return y, sr
```

**Deliverables:**
- Preprocessed audio files (500+)
- Preprocessing log

**Owner:** Audio Technician  
**Due:** Week 5

---

### Task 2.1.4 — Speech Data Labeling

**Description:** Label speech samples with dyslexia indicators.

**Labeling Criteria:**

| Indicator | Description | Labels |
|-----------|-------------|--------|
| Phonological Processing | Pause frequency, sound omissions | Normal, Mild, Moderate, Severe |
| Fluency | Words per minute, repetition count | Normal, Mild, Moderate, Severe |
| Pronunciation | Accuracy scores | Normal, Mild, Moderate, Severe |

**Labeling Process:**
1. Expert annotation (clinical advisor)
2. Secondary review for consistency
3. Inter-rater reliability check (Cohen's Kappa > 0.8)

**Deliverables:**
- Labeled speech dataset
- Annotation guidelines
- Inter-rater reliability report

**Owner:** Data Annotation Specialists  
**Due:** Week 5

---

## 2.3 Week 3-5: Handwriting Data Collection

### Task 2.2.1 — Design Handwriting Collection Methodology

**Description:** Create standardized procedures for handwriting sample collection.

**Specifications:**

| Parameter | Value |
|-----------|-------|
| Resolution | 300 DPI minimum |
| Format | PNG / TIFF |
| Paper Type | Lined, Unlined (both variations) |
| Writing Tasks | Copying, Free writing, Dictation |

**Writing Prompts by Age:**
- Ages 5-7: Simple sentences, letter tracing
- Ages 8-10: Paragraph writing, story prompts
- Ages 11-12: Essay prompts, comprehension responses

**Deliverables:**
- Handwriting collection protocol
- Standardized writing prompts by age group

**Owner:** Data Collection Coordinator  
**Due:** Week 3

---

### Task 2.2.2 — Acquire Handwriting Samples

**Description:** Collect handwriting samples from target demographic.

**Target:**
- 500+ samples
- Include: copying tasks, free writing, dictation responses
- Multiple writing samples per student (3-5)

**Capture Methods:**
- High-resolution scanner (preferred)
- Photography with standardized lighting (fallback)

**Metadata to Capture:**
- Age, grade level
- Writing task type
- Time taken
- Pen/pencil used

**Deliverables:**
- Raw handwriting images (500+)
- Metadata spreadsheet

**Owner:** Data Collection Coordinator  
**Due:** Week 4

---

### Task 2.2.3 — Apply Image Preprocessing

**Description:** Clean and normalize handwriting images.

**Preprocessing Pipeline:**

1. **Grayscale Conversion**
   - Convert RGB to grayscale

2. **Binarization**
   - Otsu thresholding
   - Adaptive thresholding for varied lighting

3. **Skew Correction**
   - Deskew to ±5 degrees
   - Hough transform for line detection

4. **Noise Removal**
   - Morphological operations (erosion/dilation)
   - Gaussian blur for smoothing

**Code Example:**
```python
import cv2
import numpy as np

def preprocess_handwriting(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Binarization - Otsu
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Skew correction
    coords = np.column_stack(np.where(binary > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45: angle = -(90 + angle)
    else: angle = -angle
    (h, w) = binary.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_CUBIC)
    return rotated
```

**Deliverables:**
- Preprocessed handwriting images (500+)
- Preprocessing log

**Owner:** Data Engineer  
**Due:** Week 5

---

### Task 2.2.4 — Handwriting Labeling

**Description:** Label handwriting samples with dyslexia indicators.

**Labeling Criteria:**

| Indicator | Description | Labels |
|-----------|-------------|--------|
| Letter Reversals | b/d, p/q, m/w, n/u confusion | None, Mild, Moderate, Severe |
| Spacing Irregularity | Inconsistent word spacing | None, Mild, Moderate, Severe |
| Character Misplacement | Letters outside expected space | None, Mild, Moderate, Severe |
| Baseline Adherence | Deviation from baseline line | None, Mild, Moderate, Severe |
| Size Consistency | Letter size variation | None, Mild, Moderate, Severe |

**Labeling Process:**
1. Expert annotation with bounding boxes
2. Heatmap generation for error density
3. Inter-rater reliability check

**Deliverables:**
- Labeled handwriting dataset
- Bounding box annotations
- Annotation guidelines

**Owner:** Data Annotation Specialists  
**Due:** Week 5

---

## 2.4 Week 3-5: Text Data Collection

### Task 2.3.1 — Design Text Collection Approach

**Description:** Create standardized procedures for text data collection.

**Specifications:**

| Component | Details |
|-----------|---------|
| Essay Prompts | 10 standardized prompts (age 5-12) |
| Dictation | Standardized passages by reading level |
| Comprehension | Reading response sheets |

**Text Types:**
- Free-form essays
- Dictation responses
- Reading comprehension answers
- Story prompts

**Deliverables:**
- Text collection protocol
- Standardized prompts by age

**Owner:** Data Collection Coordinator  
**Due:** Week 3

---

### Task 2.3.2 — Acquire Text Samples

**Description:** Collect text samples from target demographic.

**Target:**
- 500+ samples
- Diverse prompts representing different cognitive demands
- Balance across age groups

**Collection Methods:**
- Digital submission (form-based)
- Transcription from handwritten responses
- OCR for scanned handwritten text

**Metadata to Capture:**
- Age, grade level
- Prompt ID
- Word count
- Time taken

**Deliverables:**
- Raw text files (500+)
- Metadata spreadsheet

**Owner:** Data Collection Coordinator  
**Due:** Week 4

---

### Task 2.3.3 — Text Normalization

**Description:** Clean and standardize text data.

**Normalization Steps:**

1. **Encoding Standardization**
   - UTF-8 encoding
   - Unicode normalization

2. **Whitespace Handling**
   - Trim leading/trailing whitespace
   - Normalize internal spacing
   - Handle line breaks

3. **Special Characters**
   - Handle accented characters
   - Process emojis (remove or standardize)
   - Handle special symbols

**Code Example:**
```python
import unicodedata
import re

def normalize_text(text):
    # Unicode normalization
    text = unicodedata.normalize('NFKC', text)
    # Whitespace normalization
    text = re.sub(r'\s+', ' ', text).strip()
    # Lowercase (optional based on use case)
    # text = text.lower()
    return text
```

**Deliverables:**
- Normalized text files (500+)
- Normalization log

**Owner:** Data Engineer  
**Due:** Week 5

---

### Task 2.3.4 — Text Labeling

**Description:** Label text samples with dyslexia indicators.

**Labeling Criteria:**

| Indicator | Description | Labels |
|-----------|-------------|--------|
| Spelling Error Rate | Errors per 100 words | Normal, Mild, Moderate, Severe |
| Grammar Inconsistency | Syntactic errors | Normal, Mild, Moderate, Severe |
| Sentence Complexity | Length, clause count | Normal, Mild, Moderate, Severe |
| Reading Ease | Flesch score | Normal, Mild, Moderate, Severe |
| Vocabulary Diversity | Type-token ratio | Normal, Mild, Moderate, Severe |

**Labeling Process:**
1. Automated metrics (spelling, grammar, complexity)
2. Manual review for context
3. Combined score calculation

**Deliverables:**
- Labeled text dataset
- Feature extraction log
- Annotation guidelines

**Owner:** Data Annotation Specialists  
**Due:** Week 5

---

## 2.5 Weeks 5-6: Data Infrastructure

### Task 2.4.1 — Design Database Schema

**Description:** Create PostgreSQL database schema for data storage.

**Core Tables:**

```sql
-- Students table (anonymized)
CREATE TABLE students (
    student_hash VARCHAR(64) PRIMARY KEY,
    age INTEGER,
    grade VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Speech samples
CREATE TABLE speech_samples (
    id SERIAL PRIMARY KEY,
    student_hash VARCHAR(64) REFERENCES students(student_hash),
    audio_path VARCHAR(500),
    duration_seconds FLOAT,
    passage_id VARCHAR(20),
    labels JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Handwriting samples
CREATE TABLE handwriting_samples (
    id SERIAL PRIMARY KEY,
    student_hash VARCHAR(64) REFERENCES students(student_hash),
    image_path VARCHAR(500),
    task_type VARCHAR(50),
    labels JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Text samples
CREATE TABLE text_samples (
    id SERIAL PRIMARY KEY,
    student_hash VARCHAR(64) REFERENCES students(student_hash),
    text_content TEXT,
    prompt_id VARCHAR(20),
    word_count INTEGER,
    labels JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Analysis results
CREATE TABLE analysis_results (
    id SERIAL PRIMARY KEY,
    student_hash VARCHAR(64) REFERENCES students(student_hash),
    modality VARCHAR(20),
    risk_score FLOAT,
    risk_class VARCHAR(20),
    features JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Audit log
CREATE TABLE audit_log (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(64),
    action VARCHAR(100),
    target_hash VARCHAR(64),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Indexes:**
```sql
CREATE INDEX idx_student_hash ON students(student_hash);
CREATE INDEX idx_samples_student ON speech_samples(student_hash);
CREATE INDEX idx_results_student ON analysis_results(student_hash);
CREATE INDEX idx_created_at ON analysis_results(created_at);
```

**Deliverables:**
- Database schema (PostgreSQL)
- Migration scripts
- Schema documentation

**Owner:** Database Admin  
**Due:** Week 6

---

### Task 2.4.2 — Configure Cloud Storage

**Description:** Set up S3/cloud storage for binary files.

**Bucket Structure:**
```
dyslexia-detection-data/
├── raw/
│   ├── speech/
│   ├── handwriting/
│   └── text/
├── processed/
│   ├── speech/
│   ├── handwriting/
│   └── text/
└── models/
```

**Configuration:**
- Versioning enabled
- Lifecycle policies (transition to Glacier after 1 year)
- Server-side encryption (SSE-S3)
- Access controls (least privilege)

**Deliverables:**
- S3 bucket configuration
- Access policies
- Lifecycle rules

**Owner:** DevOps Engineer  
**Due:** Week 6

---

### Task 2.4.3 — Implement Data Versioning

**Description:** Set up data versioning for reproducibility.

**Version Control (DVC):**
```bash
# Initialize DVC
dvc init

# Add data to tracking
dvc add data/speech/
dvc add data/handwriting/
dvc add data/text/

# Configure remote storage
dvc remote add -d myremote s3://dyslexia-detection-data/dvc

# Push to remote
dvc push
```

**Versioning Strategy:**
- Dataset version tags (v1.0, v1.1, etc.)
- Change documentation requirements
- Reproducibility guarantee via config files

**Deliverables:**
- DVC configuration
- Version tracking documentation
- Reproducibility guide

**Owner:** Data Engineer  
**Due:** Week 6

---

### Task 2.4.4 — Create Anonymization Pipeline

**Description:** Implement data anonymization to protect student privacy.

**Anonymization Process:**

1. **Student ID Generation**
   ```python
   import hashlib
   import secrets
   
   def generate_student_hash(real_id, salt=None):
       if salt is None:
           salt = secrets.token_hex(16)
       return hashlib.sha256(f"{real_id}{salt}".encode()).hexdigest()[:16]
   ```

2. **PII Removal**
   - Names removed from text metadata
   - Dates generalized (year only)
   - Location data removed
   - Contact information removed

3. **Verification**
   - Automated PII scanning
   - Manual review for edge cases
   - Re-identification risk assessment

**Deliverables:**
- Anonymization pipeline code
- PII removal verification report
- Re-identification risk assessment

**Owner:** Data Engineer  
**Due:** Week 6

---

## 2.6 Deliverables Summary

| Deliverable | Description | Owner | Due |
|-------------|-------------|-------|-----|
| Speech Dataset | 500+ labeled audio samples | Data Team | Week 5 |
| Handwriting Dataset | 500+ labeled image samples | Data Team | Week 5 |
| Text Dataset | 500+ labeled text samples | Data Team | Week 5 |
| Data Storage Infrastructure | PostgreSQL + S3 operational | DevOps | Week 6 |
| Anonymization Pipeline | Operational de-identification system | Data Engineer | Week 6 |
| Data Quality Report | Dataset statistics and quality metrics | Data Lead | Week 6 |

---

## 2.7 Resources Required

| Resource | Quantity | Role |
|----------|----------|------|
| Data Engineer | 1 FTE | Pipeline development |
| Data Annotation Specialists | 2 FTE | Labeling all modalities |
| Audio Technician | 0.5 FTE | Recording setup and processing |
| Clinical Advisor | 0.25 FTE | Labeling quality oversight |
| DevOps Engineer | 0.5 FTE | Infrastructure setup |
| Database Admin | 0.25 FTE | Schema design |

---

## 2.8 Key Milestones

| Milestone | Description | Success Criteria |
|-----------|-------------|------------------|
| M2.1 | Speech data collection complete | 500+ samples with >90% quality passes |
| M2.2 | Handwriting data collection complete | 500+ samples with >90% quality passes |
| M2.3 | Text data collection complete | 500+ samples with >90% quality passes |
| M2.4 | Storage infrastructure operational | Database and S3 passing stress tests |
| M2.5 | All data anonymized | Zero PII in accessible fields |
| M2.6 | Data quality validated | Inter-rater reliability > 0.8 |

---

## 2.9 Quality Assurance

### Data Quality Checks

| Check | Criteria |
|-------|----------|
| Audio Quality | SNR > 20 dB, no clipping |
| Image Quality | 300+ DPI, no blur, no rotation |
| Text Quality | No encoding errors, complete sentences |
| Label Quality | Cohen's Kappa > 0.8 |
| Completeness | 100% metadata filled |

### Validation Process

1. Automated quality checks (scripts)
2. Manual review (10% sample)
3. Clinical advisor spot check
4. Final quality sign-off

---

## 2.10 Dependencies

- **External Dependencies:**
  - School partnerships for data collection
  - Clinical advisor availability for labeling

- **Internal Dependencies:**
  - Phase 1 complete (requirements, protocols)
  - Ethics approval for data collection

---

## 2.11 Risks and Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Insufficient sample size | High | Plan for 600+ target (buffer) |
| Data quality issues | High | Strict quality checks, re-collection protocol |
| Privacy breach | Critical | Encryption, access controls, audit logging |
| Label inconsistency | Medium | Inter-rater checks, calibration sessions |
| Data loss | Critical | Backup strategy, versioning |

---

## 2.12 Phase 2 Completion Checklist

### Speech Data
- [ ] Collection protocol defined
- [ ] 500+ recordings acquired
- [ ] Preprocessing pipeline operational
- [ ] Labels applied (inter-rater > 0.8)
- [ ] Quality checks passed

### Handwriting Data
- [ ] Collection protocol defined
- [ ] 500+ images acquired
- [ ] Preprocessing pipeline operational
- [ ] Labels applied with bounding boxes
- [ ] Quality checks passed

### Text Data
- [ ] Collection protocol defined
- [ ] 500+ samples acquired
- [ ] Normalization pipeline operational
- [ ] Labels applied (automated + manual)
- [ ] Quality checks passed

### Infrastructure
- [ ] Database schema deployed
- [ ] S3 bucket configured
- [ ] DVC versioning operational
- [ ] Anonymization pipeline tested
- [ ] All data anonymized

---

**Phase 2 Approval:**

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Data Engineering Lead | | | |
| Clinical Advisor | | | |
| DevOps Engineer | | | |
| Project Lead | | | |

---

*Next Phase: Phase 3 — Feature Engineering*