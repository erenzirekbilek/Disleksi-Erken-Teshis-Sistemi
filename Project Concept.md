# 🚀 Dyslexia Early Detection System — Project Phases & Technical Plan

---

## 📋 Project Overview

A multi-modal AI system that analyzes **speech**, **handwriting**, and **text** to detect dyslexia risk in students early. The system outputs a risk score (0–1) and a risk class (Low / Medium / High) with explainable AI-driven feedback.

---

## 🗂️ Phase 1 — Research & Requirements

**Goal:** Establish the scientific and technical foundation before any development begins.

### Key Activities
- Review clinical literature on dyslexia indicators (phonological processing, letter reversal, reading fluency)
- Define target age group and user personas (teachers, parents, school psychologists)
- Identify data requirements for each modality: speech, handwriting, text
- Establish labeling criteria for Low / Medium / High risk classification
- Define privacy and compliance requirements (GDPR, FERPA, COPPA for minors)
- Select evaluation metrics: Accuracy, Precision, Recall, F1, ROC-AUC

### Deliverables
- Requirements specification document
- Data collection protocol
- Ethics & privacy framework
- Annotated literature review

---

## 🗂️ Phase 2 — Data Collection & Preparation

**Goal:** Build a clean, labeled, multi-modal dataset.

### Speech Data
- Collect reading aloud recordings (WAV/MP3 format)
- Apply noise reduction (Spectral Gating / Wiener Filter)
- Perform Voice Activity Detection (VAD) to trim silence
- Normalize audio levels across samples

### Handwriting Data
- Collect scanned or photographed handwriting samples
- Apply grayscale conversion, binarization (Otsu thresholding)
- Correct skew and remove noise using morphological operations
- Label samples for letter reversals (b/d, p/q), irregular spacing, misplacement

### Text Data
- Collect student-written essays, dictation responses, reading comprehension answers
- Normalize and clean raw text
- Label for spelling error rate, grammar inconsistencies, sentence complexity

### Deliverables
- Cleaned and labeled dataset for all three modalities
- Data versioning and storage setup (PostgreSQL + S3/Cloud Storage)
- Data anonymization implementation (hash-based student IDs)

---

## 🗂️ Phase 3 — Feature Engineering

**Goal:** Extract meaningful features from each modality for model input.

### Speech Features
- MFCC (Mel Frequency Cepstral Coefficients)
- Pitch / Fundamental Frequency (F0)
- Energy (RMS)
- Spectral Centroid, Spectral Rolloff
- Zero Crossing Rate
- Sliding window segmentation (25ms frames)

### Handwriting Features
- Character segmentation and stroke analysis
- Spatial consistency measurements
- Letter reversal detection patterns
- Irregular spacing detection

### Text / NLP Features
- Spelling error rate
- Grammar inconsistency score
- Flesch Reading Ease score
- Sentence length variance
- POS tagging and dependency parsing outputs
- BERT embeddings (recommended over Word2Vec/GloVe)

### Deliverables
- Feature extraction pipelines for all three modalities
- Feature importance baseline analysis
- Feature documentation

---

## 🗂️ Phase 4 — Model Development

**Goal:** Train and validate modality-specific models, then build the fusion layer.

### Speech Model
- CNN on spectrogram input
- RNN / LSTM for temporal sequences
- Optional: Transformer-based audio model
- Output: Speech risk score `S_speech`

### Handwriting Model
- CNN architectures: ResNet or EfficientNet
- Optional: Vision Transformer (ViT)
- Tasks: character misplacement, letter reversal, spacing irregularity
- Output: Writing risk score `S_writing`

### NLP Model
- Transformer-based classifier: BERT or DistilBERT
- Input: normalized and tokenized student text
- Output: NLP risk score `S_nlp`

### Fusion Layer (Risk Scoring Engine)
- **Baseline:** Weighted sum — `R = w1·S_speech + w2·S_writing + w3·S_nlp`
- **Advanced (Recommended):** Ensemble meta-model (Gradient Boosting or MLP) taking all three scores as input
- Weights learned via grid search or meta-learning

### Deliverables
- Three trained modality models
- Fusion model
- Model performance reports (Accuracy, F1, ROC-AUC)
- k-fold cross-validation results (k = 5 or 10)

---

## 🗂️ Phase 5 — Explainability (XAI) Layer

**Goal:** Make the system's decisions transparent and understandable to non-technical users.

### Techniques
- **SHAP** — global and local feature importance
- **LIME** — local explanations per individual prediction

### LLM Integration
- Feed model outputs and top features into an LLM prompt
- Generate plain-language explanations for each student report

**Example output:**
> *"Student shows phonological processing difficulty based on low MFCC consistency and high spelling error rate."*

### Deliverables
- SHAP/LIME integration per modality
- LLM explanation generation pipeline
- Explanation quality review by domain experts

---

## 🗂️ Phase 6 — Backend & AI Microservices

**Goal:** Build a scalable, maintainable service architecture.

### Technology Stack
- **API Framework:** FastAPI (async performance)
- **Services:** Separate microservices per modality

| Service | Responsibility |
|---|---|
| audio-service | Speech preprocessing and model inference |
| vision-service | Handwriting preprocessing and model inference |
| nlp-service | Text processing and model inference |
| fusion-service | Score aggregation and final risk classification |

### API Endpoints
- `POST /analyze/speech`
- `POST /analyze/writing`
- `POST /analyze/text`
- `GET /results/{student_id}`

### Communication
- REST (simple setup) or Message Queue via RabbitMQ / Kafka (scalable async)

### Storage
- **PostgreSQL** — structured student data and results
- **S3 / Cloud Storage** — audio files and handwriting images

### Deliverables
- All four microservices deployed and tested
- API documentation
- Database schema

---

## 🗂️ Phase 7 — Frontend Development

**Goal:** Build an intuitive dashboard for educators and school psychologists.

### Technology Stack
- **Framework:** Vue.js 3 (Composition API)
- **State Management:** Pinia
- **Routing:** Vue Router
- **Charts:** Chart.js with vue-chartjs wrapper
- **Styling:** Tailwind CSS
- **HTTP Client:** Axios
- **Build Tool:** Vite

### Vue.js Architecture
- **Composition API** — `<script setup>` syntax for clean, maintainable components
- **Composables** — reusable logic extracted into `useStudentRisk()`, `useAudioCapture()`, `useCharts()` etc.
- **Pinia Stores** — separate stores for `studentStore`, `analysisStore`, `authStore`
- **Vue Router** — protected routes with navigation guards for RBAC (Admin / Teacher / Parent)
- **Component Structure:**
  - `views/` — page-level components (Dashboard, StudentDetail, Reports)
  - `components/` — reusable UI components (RiskGauge, RadarChart, ExplanationCard)
  - `composables/` — shared logic hooks
  - `stores/` — Pinia state modules
  - `services/` — Axios API service layer

### Key UI Components
- **RiskGauge.vue** — animated Low / Medium / High indicator per student
- **ProgressChart.vue** — time-series student progress (vue-chartjs)
- **ErrorHeatmap.vue** — error density across writing samples
- **RadarChart.vue** — Speech vs Writing vs NLP score comparison
- **ExplanationCard.vue** — AI-generated plain-language summary
- **StudentTable.vue** — sortable, filterable student list with risk badges
- **ModalUpload.vue** — drag-and-drop file upload for audio and handwriting samples

### Deliverables
- Fully functional Vue.js 3 dashboard
- Responsive design for desktop and tablet
- Unit tests with Vitest + Vue Test Utils
- User testing with at least 3–5 educators

---

## 🗂️ Phase 8 — Security & Privacy Implementation

**Goal:** Ensure the system meets legal and ethical standards for handling student data.

### Measures
- Data anonymization with hash-based student IDs
- Encryption at rest: AES-256
- Encryption in transit: HTTPS / TLS
- Role-Based Access Control (RBAC): Admin, Teacher, Parent roles
- Audit logging for all data access events
- Data retention and deletion policy

### Deliverables
- Security audit report
- RBAC implementation
- Privacy policy documentation

---

## 🗂️ Phase 9 — Testing & Evaluation

**Goal:** Validate the full system technically and clinically before release.

### Technical Testing
- Unit tests for each microservice
- Integration tests for the full pipeline
- Load testing for scalability under concurrent users

### Model Evaluation
- Accuracy, Precision, Recall, F1-score per modality
- ROC-AUC for risk classification
- k-fold cross-validation (k = 5 or 10)
- Confusion matrix analysis

### Clinical Validation
- Compare system outputs against expert psychologist assessments
- Measure false positive and false negative rates
- Bias audit across age groups, languages, and demographics

### Deliverables
- Full test coverage report
- Model evaluation report
- Clinical validation summary

---

## 🗂️ Phase 10 — Deployment & Scalability

**Goal:** Deploy the system reliably at scale.

### Infrastructure
- **Containerization:** Docker for all services
- **Orchestration:** Kubernetes for horizontal scaling
- **Inference Scaling:** Independent scaling of audio, vision, NLP services based on load
- **CI/CD Pipeline:** Automated testing and deployment on every release

### Deliverables
- Production deployment
- Monitoring and alerting setup
- Scaling playbook

---

## 🚀 Phase 11 — Advanced Upgrades (Elite Level)

These upgrades transform the system from strong to top-tier.

### 1. Multimodal Transformer (Unified Model)
Instead of three separate models fused at the score level, train a single transformer that ingests speech, image, and text simultaneously for deeper cross-modal learning.

### 2. Real-Time Analysis
- Integrate WebRTC for live speech capture
- Deliver instant risk scoring during reading sessions
- Enable real-time feedback for teachers during classroom activities

### 3. Personalized Learning Loop
- System recommends targeted exercises based on identified weaknesses
- Tracks student improvement over time with longitudinal scoring
- Retrains or fine-tunes models on new student data to personalize predictions

---

## 📊 Summary Timeline (Estimated)

| Phase | Duration |
|---|---|
| Phase 1 — Research & Requirements | 2 weeks |
| Phase 2 — Data Collection & Preparation | 4–6 weeks |
| Phase 3 — Feature Engineering | 2–3 weeks |
| Phase 4 — Model Development | 4–6 weeks |
| Phase 5 — Explainability Layer | 1–2 weeks |
| Phase 6 — Backend & Microservices | 3–4 weeks |
| Phase 7 — Frontend Development | 3–4 weeks |
| Phase 8 — Security & Privacy | 1–2 weeks |
| Phase 9 — Testing & Evaluation | 2–3 weeks |
| Phase 10 — Deployment | 1–2 weeks |
| Phase 11 — Advanced Upgrades | Ongoing |

**Total estimated duration: ~6–7 months for core system**

---

## 💡 Key Success Factors

- **Clinical grounding** — involve speech therapists and educational psychologists throughout
- **Data quality** — model performance is only as good as the labeled dataset
- **Explainability** — without clear explanations, educators will not trust or adopt the system
- **Privacy first** — student data requires the highest protection standards
- **Iterative development** — release early, gather teacher feedback, improve continuously
