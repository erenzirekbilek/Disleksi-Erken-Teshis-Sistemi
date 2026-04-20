# Dyslexia Early Detection System

A multi-modal AI system for early dyslexia risk detection in children using speech, handwriting, and text analysis.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Vue.js](https://img.shields.io/badge/Vue.js-4FC08D?style=flat-square&logo=vuedotjs&logoColor=white)](https://vuejs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](./LICENSE)

[Features](#features) · [Tech Stack](#tech-stack) · [Getting Started](#getting-started) · [Project Structure](#project-structure) · [Contributing](#contributing)

---

## Overview

This system uses machine learning to detect early signs of dyslexia in children through multiple modalities:
- **Speech Analysis** - Audio processing for speech patterns
- **Handwriting Analysis** - Image-based analysis of handwriting samples
- **Text Analysis** - NLP-based text evaluation for writing patterns

---

## Features

### Multi-Modal Analysis
- Speech pattern detection using MFCC features
- Handwriting analysis with CNN/ViT models
- Text analysis with BERT/Transformers

### Deep Learning Frameworks
- **PyTorch** - Primary deep learning framework
- **TensorFlow/Keras** - Alternative model implementations
- **Transformers** - Hugging Face transformers for NLP

### API & Frontend
- FastAPI-based REST API
- Vue.js frontend application
- Real-time analysis endpoints

---

## Tech Stack

| Layer | Technology |
| :--- | :--- |
| Backend | Python, FastAPI, PyTorch, TensorFlow, Keras |
| Frontend | Vue.js 3, TypeScript, Vite |
| NLP | Transformers (BERT, DistilBERT) |
| Computer Vision | OpenCV, PyTorch Vision |
| Audio Processing | Librosa |
| Database | SQLite |

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- pip or conda

### Backend Setup

```bash
# Navigate to MVP
cd MVP

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the API
python -m app.main
```

### Frontend Setup

```bash
cd MVP/frontend
npm install
npm run dev
```

---

## Project Structure

```
Dyslexia Early Detection System/
├── MVP/
│   ├── app/                 # FastAPI application
│   │   ├── main.py         # Main API entry point
│   │   └── text_processor.py
│   └── frontend/           # Vue.js frontend
├── Phase 1 - Data Collection/
├── Phase 2 - Data Preparation/
├── Phase 3 - Feature Engineering/
│   ├── text/
│   ├── handwriting/
│   └── speech/
├── Phase 4 - Model Development/
│   ├── text/               # Text models (PyTorch & Keras)
│   ├── handwriting/        # Handwriting models
│   ├── speech/             # Speech models
│   └── fusion/             # Multi-modal fusion
├── Phase 5 - Explainability/
│   ├── shap_explainer.py
│   ├── lime_explainer.py
│   └── llm_explainer.py
└── tests/
```

---

## Models

### Text Models
- BERT-based classifier (PyTorch)
- DistilBERT classifier (PyTorch)
- CNN/LSTM/GRU (Keras)

### Handwriting Models
- Custom CNN (PyTorch)
- ResNet-based (PyTorch)
- EfficientNet (PyTorch)
- Vision Transformer (ViT)

### Speech Models
- Spectrogram CNN (PyTorch)
- Bidirectional LSTM (PyTorch)
- Transformer encoder (PyTorch)

---

## API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/analyze` | POST | Full multi-modal analysis |
| `/analyze/audio` | POST | Speech analysis |
| `/analyze/handwriting` | POST | Handwriting analysis |
| `/analyze/text` | POST | Text analysis |
| `/explain` | GET | Generate explanation |

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the [MIT License](./LICENSE).

---

## Contact

**Eren Zirekbilek**
- Email: erenzirekbilek@hotmail.com
- GitHub: [@erenkirekbilek](https://github.com/erenkirekbilek)