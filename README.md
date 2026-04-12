# 🧠 Dyslexia Early Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)
![Vue.js](https://img.shields.io/badge/Vue.js-3-green?style=flat&logo=vue.js)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-blue?style=flat&logo=fastapi)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-blue?style=flat&logo=postgresql)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

**A multi-modal AI system that analyzes speech, handwriting, and text to detect dyslexia risk in students early.**

</div>

---

## 📚 Table of Contents

- [About](#about)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Phases](#project-phases)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Data Collection](#data-collection)
- [Security & Compliance](#security--compliance)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## About

Dyslexia Early Detection System is an AI-powered platform designed to identify dyslexia risk factors in students through analysis of:

- **🗣️ Speech**: Phonological processing, fluency, and pronunciation analysis
- **✍️ Handwriting**: Letter reversals, spacing, character placement
- **📝 Text**: Spelling errors, grammar, sentence complexity

The system provides a risk score (0-1) with classifications (Low/Medium/High) and explainable AI-driven feedback for educators and parents.

> **Goal**: Enable early intervention for students with dyslexia by providing accessible, AI-assisted screening tools.

---

## Features

### Core Features
- Multi-modal analysis (speech, handwriting, text)
- Risk classification (Low/Medium/High)
- Explainable AI (SHAP, LIME)
- LLM-generated plain-language explanations
- Longitudinal progress tracking

### Technical Features
- Microservices architecture
- RESTful API (FastAPI)
- Vue.js 3 dashboard
- PostgreSQL + S3 storage
- Docker & Kubernetes ready
- End-to-end encryption

### Compliance
- GDPR compliant
- FERPA compliant
- COPPA compliant (under-13 data)

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Vue.js 3, Pinia, Tailwind CSS, Chart.js |
| **Backend** | FastAPI, Python 3.10+ |
| **Database** | PostgreSQL 15+ |
| **Storage** | AWS S3 (or equivalent) |
| **ML/AI** | PyTorch, TensorFlow, BERT, SHAP |
| **Container** | Docker, Kubernetes |
| **CI/CD** | GitHub Actions |

### ML Models

| Modality | Model |
|----------|-------|
| Speech | CNN/LSTM/Transformer on spectrograms |
| Handwriting | ResNet/EfficientNet/ViT |
| Text | BERT/DistilBERT |
| Fusion | XGBoost/LightGBM or MLP |

---

## Project Phases

| Phase | Duration | Description |
|-------|----------|-------------|
| Phase 1 | 2 weeks | Research & Requirements |
| Phase 2 | 4-6 weeks | Data Collection & Preparation |
| Phase 3 | 2-3 weeks | Feature Engineering |
| Phase 4 | 4-6 weeks | Model Development |
| Phase 5 | 1-2 weeks | Explainability (XAI) Layer |
| Phase 6 | 3-4 weeks | Backend & AI Microservices |
| Phase 7 | 3-4 weeks | Frontend Development |
| Phase 8 | 1-2 weeks | Security & Privacy Implementation |
| Phase 9 | 2-3 weeks | Testing & Evaluation |
| Phase 10 | 1-2 weeks | Deployment & Scalability |

**Total Estimated Duration**: ~6-7 months

---

## Getting Started

### Prerequisites

```bash
Python 3.10+
Node.js 18+
PostgreSQL 15+
Docker (optional)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/erenzirekbilek/Disleksi-Erken-Teshis-Sistemi.git
cd Disleksi-Erken-Teshis-Sistemi
```

2. **Set up Python environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

3. **Set up database**
```bash
psql -f Phase\ 2\ -\ Data\ Collection\ \&\ Preparation/infrastructure/database_schema.sql
```

4. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your settings
```

5. **Run the application**
```bash
# Backend
cd backend
uvicorn main:app --reload

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

### Docker Setup (Alternative)

```bash
docker-compose up -d
```

---

## Project Structure

```
Dyslexia-Early-Detection-System/
├── Project Concept.md                    # Project overview
├── README.md                             # This file
├── .gitignore                           # Git ignore rules
│
├── Phase 1 - Research & Requirements/
│   ├── Research&Requirements.md          # Phase documentation
│   ├── templates/
│   │   ├── consent_form.md              # Parental consent template
│   │   └── user_personas.md              # User personas
│   └── compliance/
│       ├── GDPR_checklist.md            # GDPR compliance
│       ├── FERPA_checklist.md            # FERPA compliance
│       └── COPPA_checklist.md            # COPPA compliance
│
├── Phase 2 - Data Collection & Preparation/
│   ├── DataCollection&Preparation.md    # Phase documentation
│   ├── speech/
│   │   ├── preprocess.py                # Audio preprocessing
│   │   ├── label.py                     # Speech labeling
│   │   └── collection_protocol.md      # Collection guide
│   ├── handwriting/
│   │   ├── preprocess.py                # Image preprocessing
│   │   ├── label.py                     # Handwriting labeling
│   │   └── collection_protocol.md      # Collection guide
│   ├── text/
│   │   ├── preprocess.py                # Text preprocessing
│   │   ├── label.py                     # Text labeling
│   │   └── collection_protocol.md      # Collection guide
│   └── infrastructure/
│       ├── database_schema.sql          # PostgreSQL schema
│       ├── anonymizer.py                # Data anonymization
│       └── s3_config.md                 # S3 storage config
│
└── Phase 3+ (Coming soon)/
    ├── Feature Engineering/
    ├── Model Development/
    ├── Backend Services/
    └── Frontend Dashboard/
```

---

## Data Collection

### Data Requirements

| Modality | Target | Format |
|----------|--------|--------|
| Speech | 500+ samples | WAV/MP3 (48kHz, 16-bit) |
| Handwriting | 500+ images | PNG/TIFF (300 DPI) |
| Text | 500+ samples | Plain text |

### Collection Protocols

Detailed collection protocols are available in:
- `Phase 2 - Data Collection & Preparation/speech/collection_protocol.md`
- `Phase 2 - Data Collection & Preparation/handwriting/collection_protocol.md`
- `Phase 2 - Data Collection & Preparation/text/collection_protocol.md`

---

## Security & Compliance

### Data Protection

- **Encryption at rest**: AES-256
- **Encryption in transit**: TLS 1.3
- **Anonymization**: SHA-256 with salt
- **Access control**: Role-based (RBAC)
- **Audit logging**: All data access tracked

### Compliance Framework

| Regulation | Status |
|------------|--------|
| GDPR | ✅ Complete |
| FERPA | ✅ Complete |
| COPPA | ✅ Complete |

See compliance checklists in `Phase 1 - Research & Requirements/compliance/`

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## Contact

**Project Lead**: Eren Zirek Bilek  
**GitHub**: [@erenzirekbilek](https://github.com/erenzirekbilek)

---

<div align="center">

**Made with ❤️ for educators, parents, and students**

*Helping every child reach their full potential through early detection.*

</div>