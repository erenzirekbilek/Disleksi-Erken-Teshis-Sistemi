# 🧠 Dyslexia Early Detection System - MVP

A simplified, all-in-one FastAPI application for dyslexia risk detection using multi-modal AI.

## Features

- **Speech Analysis**: Audio feature extraction using MFCC
- **Handwriting Analysis**: Image processing for writing patterns
- **Text Analysis**: Spelling and grammar evaluation
- **Fusion Model**: Combined risk scoring
- **Grok Integration**: AI-powered plain-language explanations (xAI)

## Quick Start

```bash
# Install dependencies
cd MVP
pip install -r requirements.txt

# Set Grok API key (optional)
export GROK_API_KEY="your-api-key"

# Run the app
python -m app.main
```

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

## Example Usage

```bash
# Full analysis
curl -X POST "http://localhost:8000/analyze" \
  -F "student_id=12345" \
  -F "text=The cat sat on the mat."

# With files
curl -X POST "http://localhost:8000/analyze" \
  -F "student_id=12345" \
  -F "audio=@audio.wav" \
  -F "image=@writing.png" \
  -F "text=My story..."

# Get explanation
curl "http://localhost:8000/explain?speech_score=0.2&handwriting_score=0.3&text_score=0.1"
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GROK_API_KEY` | xAI Grok API key (optional) |

## Tech Stack

- **Framework**: FastAPI
- **Audio**: librosa
- **Image**: OpenCV, PIL
- **LLM**: xAI Grok (optional)

## License

MIT