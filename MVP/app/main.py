# Dyslexia Early Detection System - MVP
# A simplified, all-in-one FastAPI application

import os
import json
import logging
import io
import base64
from typing import Optional, List, Dict, Any
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Dyslexia Early Detection System",
    description="MVP - Multi-modal AI system for dyslexia risk detection",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Pydantic Models
# =============================================================================


class AnalysisRequest(BaseModel):
    student_id: str
    age: int
    grade: str


class AnalysisResponse(BaseModel):
    student_id: str
    overall_risk: str
    overall_score: float
    speech_score: Optional[float] = None
    handwriting_score: Optional[float] = None
    text_score: Optional[float] = None
    explanation: Optional[str] = None
    top_features: Optional[List[Dict]] = None


# =============================================================================
# Core Configuration
# =============================================================================


class Config:
    PROJECT_NAME = "Dyslexia Early Detection System"
    VERSION = "1.0.0"

    # Model settings
    RISK_THRESHOLDS = {"low": 0.33, "medium": 0.66}

    # Feature weights for fusion
    WEIGHTS = {"speech": 0.33, "handwriting": 0.33, "text": 0.34}

    # Grok API settings (xAI)
    GROK_API_KEY = os.environ.get("GROK_API_KEY", "")
    GROK_MODEL = "grok-beta"


config = Config()


# =============================================================================
# Services - Data Processing
# =============================================================================


class AudioProcessor:
    """Simple audio feature extraction"""

    @staticmethod
    def process(audio_data: bytes) -> Dict[str, float]:
        try:
            import librosa

            with io.BytesIO(audio_data) as f:
                y, sr = librosa.load(f, sr=22050, duration=10)

            # Simple features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

            features = {
                "mfcc_mean": float(np.mean(mfcc)),
                "mfcc_std": float(np.std(mfcc)),
                "energy": float(np.mean(librosa.feature.rms(y=y))),
                "zcr": float(np.mean(librosa.feature.zero_crossing_rate(y=y))),
            }

            # Calculate risk score (simplified)
            speech_score = min(max((features["mfcc_mean"] + 50) / 100, 0), 1)

            return {
                "features": features,
                "score": speech_score,
                "risk": "low"
                if speech_score < 0.33
                else "medium"
                if speech_score < 0.66
                else "high",
            }

        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            return {"features": {}, "score": 0.0, "risk": "low"}


class HandwritingProcessor:
    """Simple handwriting feature extraction"""

    @staticmethod
    def process(image_data: bytes) -> Dict[str, float]:
        try:
            import cv2
            from PIL import Image
            import torch

            # Load image
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

            if img is None:
                raise ValueError("Could not decode image")

            # Resize and process
            img = cv2.resize(img, (224, 224))
            _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

            # Simple features
            ink_ratio = np.sum(binary < 127) / binary.size
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            features = {
                "ink_ratio": float(ink_ratio),
                "character_count": len(contours),
                "contour_density": len(contours) / (224 * 224),
            }

            # Calculate risk score
            handwriting_score = min(max(ink_ratio * 2, 0), 1)

            return {
                "features": features,
                "score": handwriting_score,
                "risk": "low"
                if handwriting_score < 0.33
                else "medium"
                if handwriting_score < 0.66
                else "high",
            }

        except Exception as e:
            logger.error(f"Handwriting processing error: {e}")
            return {"features": {}, "score": 0.0, "risk": "low"}


class TextProcessor:
    """Simple text feature extraction"""

    @staticmethod
    def process(text: str) -> Dict[str, float]:
        try:
            words = text.split()
            sentences = text.split(".")

            # Simple features
            word_count = len(words)
            avg_word_length = np.mean([len(w) for w in words]) if words else 0

            # Simple spelling check
            common_errors = ["teh", "recieve", "definately", "occured", "seperate"]
            error_count = sum(1 for w in words if w.lower() in common_errors)
            error_rate = error_count / word_count if word_count > 0 else 0

            features = {
                "word_count": word_count,
                "avg_word_length": avg_word_length,
                "error_rate": error_rate,
                "sentence_count": len([s for s in sentences if s.strip()]),
            }

            # Calculate risk score
            text_score = min(
                max(error_rate * 3 + (1 - min(avg_word_length / 10, 1)) * 0.3, 0), 1
            )

            return {
                "features": features,
                "score": text_score,
                "risk": "low"
                if text_score < 0.33
                else "medium"
                if text_score < 0.66
                else "high",
            }

        except Exception as e:
            logger.error(f"Text processing error: {e}")
            return {"features": {}, "score": 0.0, "risk": "low"}


# =============================================================================
# Services - LLM (Grok)
# =============================================================================


class GrokClient:
    """xAI Grok integration"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.GROK_API_KEY
        self.base_url = "https://api.x.ai/v1"

    def generate(self, prompt: str) -> str:
        if not self.api_key:
            return self._fallback_explanation(prompt)

        try:
            import requests

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful educational assistant.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "model": config.GROK_MODEL,
                "temperature": 0.3,
                "max_tokens": 200,
            }

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return self._fallback_explanation(prompt)

        except Exception as e:
            logger.error(f"Grok API error: {e}")
            return self._fallback_explanation(prompt)

    def _fallback_explanation(self, prompt: str) -> str:
        if "low" in prompt.lower():
            return "Your child shows typical development patterns. Continue with regular reading activities."
        elif "high" in prompt.lower():
            return "We recommend consulting with a reading specialist for a comprehensive evaluation."
        else:
            return "Some areas may benefit from additional support. Consider speaking with your child's teacher."


class ExplanationGenerator:
    """Generate plain-language explanations"""

    def __init__(self):
        self.grok = GrokClient()
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict:
        return {
            "low": """The assessment shows LOW risk for dyslexia. Your child demonstrates typical patterns in speech, handwriting, and text writing. Continue encouraging reading and writing activities at home.

Key observations:
- Speech patterns: Typical development
- Handwriting: Age-appropriate skills
- Text writing: Standard spelling and grammar

Recommendation: Continue regular literacy activities and monitor progress.""",
            "medium": """The assessment shows MEDIUM risk for dyslexia. Some areas may benefit from additional support. Early intervention can be very helpful.

Key observations:
- Some patterns suggest potential difficulty
- Specific areas identified that could be strengthened

Recommendations:
- Consider targeted literacy activities
- Discuss with teacher about classroom accommodations
- Monitor progress closely

Next steps: Schedule a follow-up assessment in 3-6 months.""",
            "high": """The assessment shows HIGH risk for dyslexia indicators. We recommend further professional evaluation.

Key observations:
- Multiple areas show patterns associated with dyslexia
- Early intervention is strongly recommended

Recommendations:
- Consult with a reading specialist or educational psychologist
- Request evaluation through your school
- Consider professional dyslexia assessment

Note: This is a screening tool, not a diagnosis. A comprehensive evaluation is needed for proper diagnosis.""",
        }

    def generate(self, scores: Dict, risk_level: str) -> str:
        # Try Grok first
        if config.GROK_API_KEY:
            prompt = f"""Generate a brief, parent-friendly explanation for dyslexia screening results:
- Speech score: {scores.get("speech", 0):.2f}
- Handwriting score: {scores.get("handwriting", 0):.2f}
- Text score: {scores.get("text", 0):.2f}
- Overall risk: {risk_level}

Keep it 2-4 sentences, encouraging, and include one specific recommendation."""
            return self.grok.generate(prompt)

        # Fallback to templates
        return self.templates.get(risk_level, self.templates["medium"])


# =============================================================================
# Services - Analysis Pipeline
# =============================================================================


class AnalysisPipeline:
    """Main analysis pipeline"""

    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.handwriting_processor = HandwritingProcessor()
        self.text_processor = TextProcessor()
        self.explanation_generator = ExplanationGenerator()

    async def analyze(
        self,
        student_id: str,
        audio_data: Optional[bytes] = None,
        image_data: Optional[bytes] = None,
        text_data: Optional[str] = None,
    ) -> AnalysisResponse:

        scores = {}
        features = {}

        # Process each modality
        if audio_data:
            result = self.audio_processor.process(audio_data)
            scores["speech"] = result["score"]
            features["speech"] = result.get("features", {})

        if image_data:
            result = self.handwriting_processor.process(image_data)
            scores["handwriting"] = result["score"]
            features["handwriting"] = result.get("features", {})

        if text_data:
            result = self.text_processor.process(text_data)
            scores["text"] = result["score"]
            features["text"] = result.get("features", {})

        # Calculate overall score
        if scores:
            weights = config.WEIGHTS
            overall_score = sum(
                scores.get(modality, 0) * weights.get(modality, 0.33)
                for modality in ["speech", "handwriting", "text"]
            )
        else:
            overall_score = 0.0

        # Determine risk level
        if overall_score < config.RISK_THRESHOLDS["low"]:
            overall_risk = "low"
        elif overall_score < config.RISK_THRESHOLDS["medium"]:
            overall_risk = "medium"
        else:
            overall_risk = "high"

        # Generate explanation
        explanation = self.explanation_generator.generate(scores, overall_risk)

        # Get top features
        top_features = []
        for modality, feat in features.items():
            if feat:
                top_features.append(
                    {"modality": modality, "features": list(feat.items())[:5]}
                )

        return AnalysisResponse(
            student_id=student_id,
            overall_risk=overall_risk,
            overall_score=round(overall_score, 3),
            speech_score=round(scores.get("speech", 0), 3),
            handwriting_score=round(scores.get("handwriting", 0), 3),
            text_score=round(scores.get("text", 0), 3),
            explanation=explanation,
            top_features=top_features,
        )


# Initialize pipeline
pipeline = AnalysisPipeline()


# =============================================================================
# API Endpoints
# =============================================================================


@app.get("/")
async def root():
    return {"name": config.PROJECT_NAME, "version": config.VERSION, "status": "running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_student(
    student_id: str = Form(...),
    audio: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
):
    """Analyze a student for dyslexia risk"""

    audio_data = None
    image_data = None

    if audio:
        audio_data = await audio.read()

    if image:
        image_data = await image.read()

    result = await pipeline.analyze(
        student_id=student_id,
        audio_data=audio_data,
        image_data=image_data,
        text_data=text,
    )

    return result


@app.post("/analyze/audio")
async def analyze_audio(student_id: str = Form(...), audio: UploadFile = File(...)):
    """Analyze audio for speech patterns"""

    audio_data = await audio.read()
    result = pipeline.audio_processor.process(audio_data)

    return JSONResponse(
        {
            "student_id": student_id,
            "score": result["score"],
            "risk": result["risk"],
            "features": result.get("features", {}),
        }
    )


@app.post("/analyze/handwriting")
async def analyze_handwriting(
    student_id: str = Form(...), image: UploadFile = File(...)
):
    """Analyze handwriting for writing patterns"""

    image_data = await image.read()
    result = pipeline.handwriting_processor.process(image_data)

    return JSONResponse(
        {
            "student_id": student_id,
            "score": result["score"],
            "risk": result["risk"],
            "features": result.get("features", {}),
        }
    )


@app.post("/analyze/text")
async def analyze_text(student_id: str = Form(...), text: str = Form(...)):
    """Analyze text for writing patterns"""

    result = pipeline.text_processor.process(text)

    return JSONResponse(
        {
            "student_id": student_id,
            "score": result["score"],
            "risk": result["risk"],
            "features": result.get("features", {}),
        }
    )


@app.get("/explain")
async def explain_scores(
    speech_score: float = 0.0, handwriting_score: float = 0.0, text_score: float = 0.0
):
    """Generate explanation for given scores"""

    scores = {
        "speech": speech_score,
        "handwriting": handwriting_score,
        "text": text_score,
    }

    if speech_score + handwriting_score + text_score < config.RISK_THRESHOLDS["low"]:
        risk = "low"
    elif (
        speech_score + handwriting_score + text_score < config.RISK_THRESHOLDS["medium"]
    ):
        risk = "medium"
    else:
        risk = "high"

    explanation = pipeline.explanation_generator.generate(scores, risk)

    return {"scores": scores, "risk": risk, "explanation": explanation}


@app.get("/results/{student_id}")
async def get_results(student_id: str):
    """Get stored results for a student (placeholder)"""

    return {
        "student_id": student_id,
        "message": "Results storage not implemented in MVP",
    }


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║   🧠 Dyslexia Early Detection System - MVP                     ║
║                                                                   ║
║   API: http://localhost:8000                                     ║
║   Docs: http://localhost:8000/docs                              ║
║                                                                   ║
║   To use Grok, set GROK_API_KEY environment variable            ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(app, host="0.0.0.0", port=8000)
