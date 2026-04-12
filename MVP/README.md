# 🧠 Dyslexia Early Detection System - MVP

A simplified, all-in-one FastAPI application for dyslexia risk detection using multi-modal AI.

## What is This Project?

The **Dyslexia Early Detection System** is an AI-powered platform designed to identify dyslexia risk factors in students through multi-modal analysis:

- **🗣️ Speech Analysis**: Audio feature extraction using MFCC (Mel-Frequency Cepstral Coefficients)
- **✍️ Handwriting Analysis**: Image processing for writing patterns (ink ratio, contour analysis)
- **📝 Text Analysis**: Dyslexia-specific Turkish NLP metrics with phonetic and visual error detection

### Project Goal

Enable early intervention for students with dyslexia by providing accessible, AI-assisted screening tools. The system provides:
- Risk score (0-1) with classifications (Low/Medium/High)
- Detailed metrics for each modality
- LLM-generated plain-language explanations via xAI Grok

---

## Technical Stack

| Layer | Technology |
|-------|------------|
| **Backend Framework** | FastAPI (Python 3.10+) |
| **Frontend** | Vue.js 3, Vite, Axios |
| **Audio Processing** | librosa (MFCC extraction) |
| **Image Processing** | OpenCV, PIL |
| **NLP/Text** | Custom Turkish NLP algorithms |
| **LLM Integration** | xAI Grok (optional) |

---

## Algorithms Used

### 1. Speech Analysis
- **MFCC (Mel-Frequency Cepstral Coefficients)**: 13 coefficients extracted from audio
- **RMS Energy**: Audio energy measurement
- **Zero Crossing Rate**: Voice activity detection

### 2. Handwriting Analysis
- **Ink Ratio**: Proportion of dark pixels in writing sample
- **Contour Detection**: OpenCV contour finding for character analysis
- **Contour Density**: Characters per pixel density calculation

### 3. Text Analysis (Dyslexia-Specific Turkish NLP)

This is the core innovation of the MVP. We use specialized algorithms for Turkish:

| Algorithm | Purpose |
|-----------|---------|
| **Turkish Soundex** | Phonetic encoding adapted for Turkish letters (ç, ş, ğ, ü, ö, ı) |
| **Weighted Levenshtein Distance** | Visual letter confusion detection with higher weights for dyslexia-relevant errors |
| **Visual Similarity Checker** | Detects b↔d, p↔q, m↔n confusions |
| **Turkish Syllable Analyzer** | Heceleme analysis for split/fusion detection |
| **Dictionary Validator** | Turkish word validation with ~500 common words |

#### Text Processing Metrics:
- **Visual Error Rate**: b-d, p-q, m-n letter confusions
- **Phonetic Error Rate**: "ogul"→"okul", "hiyake"→"hikaye" type errors
- **Fusion Candidates**: Words possibly wrongly merged
- **Split Candidates**: Words possibly wrongly split

### 4. Fusion Model
- **Weighted Averaging**: Speech (33%), Handwriting (33%), Text (34%)
- **Risk Thresholds**: Low (<0.33), Medium (0.33-0.66), High (>0.66)

---

## Quick Start

### Backend (FastAPI)

```bash
# Install dependencies
cd MVP
pip install -r requirements.txt

# Set Grok API key (optional)
export GROK_API_KEY="your-api-key"

# Run the app
python -m app.main
```

### Frontend (Vue.js 3)

```bash
# Install dependencies
cd MVP/frontend
npm install

# Run the frontend
npm run dev
```

The frontend will be available at `http://localhost:3000` and proxies API calls to `http://localhost:8000`.

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/analyze` | POST | Full analysis (all modalities) |
| `/analyze/audio` | POST | Audio only |
| `/analyze/handwriting` | POST | Handwriting only |
| `/analyze/text` | POST | Text only |
| `/explain` | GET | Generate explanation |

---

## Example Usage

### Full Analysis with Text
```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "student_id=12345" \
  -F "text=Ben ogul gitmek istiyorum"
```

### With Files
```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "student_id=12345" \
  -F "audio=@audio.wav" \
  -F "image=@writing.png" \
  -F "text=My story..."
```

### Get Explanation
```bash
curl "http://localhost:8000/explain?speech_score=0.2&handwriting_score=0.3&text_score=0.1"
```

---

## Response Format

```json
{
  "student_id": "12345",
  "overall_risk": "medium",
  "overall_score": 0.45,
  "speech_score": 0.30,
  "handwriting_score": 0.25,
  "text_score": 0.80,
  "explanation": "Some areas may benefit from additional support...",
  "top_features": [
    {
      "modality": "text",
      "features": [
        ["visual_error_rate", 0.15],
        ["phonetic_error_rate", 0.08],
        ["fusion_candidates", 2]
      ]
    }
  ]
}
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GROK_API_KEY` | xAI Grok API key (optional, for AI explanations) |

---

## Project Structure

```
MVP/
├── app/
│   ├── main.py              # FastAPI application
│   └── text_processor.py   # Dyslexia-specific Turkish NLP
├── frontend/
│   ├── src/
│   │   ├── views/          # Vue components
│   │   ├── services/       # API client
│   │   └── router/         # Vue Router
│   ├── package.json
│   └── vite.config.js
├── requirements.txt
└── README.md
```

---

## License

MIT
