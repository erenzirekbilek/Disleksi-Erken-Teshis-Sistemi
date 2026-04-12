import os
import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    GROK = "grok"


@dataclass
class ExplanationRequest:
    modality: str
    scores: Dict[str, float]
    risk_level: str
    top_features: List[Dict[str, Any]]
    student_context: Optional[Dict] = None


@dataclass
class ExplanationResponse:
    text: str
    provider: str
    model: str
    tokens_used: int
    latency_ms: float


class BaseLLMClient:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str) -> str:
        raise NotImplementedError()


class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        super().__init__(api_key, model)

    def generate(self, prompt: str) -> str:
        try:
            import openai

            openai.api_key = self.api_key

            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful educational assistant that explains dyslexia assessment results to parents and teachers.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=200,
            )

            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class AnthropicClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307"):
        super().__init__(api_key, model)

    def generate(self, prompt: str) -> str:
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.api_key)

            response = client.messages.create(
                model=self.model,
                max_tokens=200,
                system="You are a helpful educational assistant that explains dyslexia assessment results to parents and teachers.",
                messages=[{"role": "user", "content": prompt}],
            )

            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise


class LocalClient(BaseLLMClient):
    def __init__(
        self, model: str = "microsoft/phi-2", endpoint: str = "http://localhost:8000"
    ):
        super().__init__(model=model)
        self.endpoint = endpoint

    def generate(self, prompt: str) -> str:
        try:
            import requests

            response = requests.post(
                self.endpoint, json={"prompt": prompt, "model": self.model}, timeout=30
            )

            if response.status_code == 200:
                return response.json().get("generated_text", "")
            else:
                raise Exception(f"Local API error: {response.status_code}")
        except Exception as e:
            logger.error(f"Local LLM error: {e}")
            return self._fallback_explanation(prompt)


class LLMExplanationGenerator:
    def __init__(
        self,
        provider: LLMProvider = LLMProvider.OPENAI,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        cache_dir: Optional[str] = None,
    ):
        self.provider = provider
        self.cache_dir = cache_dir
        self.cache = {}

        if cache_dir and os.path.exists(cache_dir):
            self._load_cache()

        if provider == LLMProvider.OPENAI:
            self.client = OpenAIClient(
                api_key or os.environ.get("OPENAI_API_KEY"), model
            )
        elif provider == LLMProvider.ANTHROPIC:
            self.client = AnthropicClient(
                api_key or os.environ.get("ANTHROPIC_API_KEY"), model
            )
        elif provider == LLMProvider.HUGGINGFACE:
            self.client = LocalClient(model=model)
        elif provider == LLMProvider.LOCAL:
            self.client = LocalClient(model=model)
        else:
            raise ValueError(f"Unknown provider: {provider}")

        self.templates = self._load_templates()

    def _load_cache(self):
        try:
            cache_file = os.path.join(self.cache_dir, "explanation_cache.json")
            with open(cache_file, "r") as f:
                self.cache = json.load(f)
            logger.info(f"Loaded {len(self.cache)} cached explanations")
        except:
            self.cache = {}

    def _save_cache(self):
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_file = os.path.join(self.cache_dir, "explanation_cache.json")
            with open(cache_file, "w") as f:
                json.dump(self.cache, f, indent=2)

    def _load_templates(self) -> Dict:
        return {
            "low_risk": """Based on the dyslexia screening assessment:
- Overall Risk Level: LOW
- Risk Scores: Speech={speech_score:.2f}, Handwriting={handwriting_score:.2f}, Text={text_score:.2f}
- Top Contributing Factors: {top_features}

Please provide a concise, encouraging explanation (2-3 sentences) for a parent. Focus on strengths and normal development. Avoid medical terminology.""",
            "medium_risk": """Based on the dyslexia screening assessment:
- Overall Risk Level: MEDIUM
- Risk Scores: Speech={speech_score:.2f}, Handwriting={handwriting_score:.2f}, Text={text_score:.2f}
- Top Contributing Factors: {top_features}

Please provide a balanced explanation (3-4 sentences) for a teacher/parent. Acknowledge areas needing support while emphasizing that early intervention is helpful. Include one specific suggestion.""",
            "high_risk": """Based on the dyslexia screening assessment:
- Overall Risk Level: HIGH
- Risk Scores: Speech={speech_score:.2f}, Handwriting={handwriting_score:.2f}, Text={text_score:.2f}
- Top Contributing Factors: {top_features}

Please provide a clear, supportive explanation (3-4 sentences) for a parent. Emphasize that this is a screening not a diagnosis, and recommend next steps for professional evaluation. Be encouraging about intervention options.""",
        }

    def _format_features(self, features: List[Dict]) -> str:
        formatted = []
        for i, f in enumerate(features[:5]):
            feature_name = f.get("feature", f.get("word", f"factor_{i}"))
            importance = f.get("importance", f.get("weight", 0))
            formatted.append(
                f"  {i + 1}. {feature_name} (importance: {importance:.3f})"
            )
        return "\n".join(formatted) if formatted else "  Multiple factors"

    def generate(self, request: ExplanationRequest) -> ExplanationResponse:
        cache_key = f"{request.modality}_{request.risk_level}_{hash(json.dumps(request.scores, sort_keys=True))}"

        if cache_key in self.cache:
            logger.info(f"Using cached explanation for {cache_key}")
            cached = self.cache[cache_key]
            return ExplanationResponse(
                text=cached["text"],
                provider=self.provider.value,
                model=self.model,
                tokens_used=0,
                latency_ms=0,
            )

        template = self.templates.get(
            f"{request.risk_level}_risk", self.templates["medium_risk"]
        )

        prompt = template.format(
            speech_score=request.scores.get("speech", 0),
            handwriting_score=request.scores.get("handwriting", 0),
            text_score=request.scores.get("text", 0),
            top_features=self._format_features(request.top_features),
        )

        if request.student_context:
            prompt += f"\n\nAdditional Context: Age {request.student_context.get('age', 'unknown')}, Grade {request.student_context.get('grade', 'unknown')}"

        start_time = time.time()

        try:
            explanation = self.client.generate(prompt)
            latency = (time.time() - start_time) * 1000

            response = ExplanationResponse(
                text=explanation,
                provider=self.provider.value,
                model=self.client.model,
                tokens_used=len(explanation.split()),
                latency_ms=latency,
            )

            self.cache[cache_key] = {"text": explanation}
            self._save_cache()

            return response

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._fallback_response(request)

    def _fallback_response(self, request: ExplanationRequest) -> ExplanationResponse:
        fallbacks = {
            "low": "Your child shows typical development patterns in the areas assessed. Continue supporting reading and writing activities at home.",
            "medium": "Some areas were identified that may benefit from additional support. Consider speaking with your child's teacher about targeted interventions.",
            "high": "The assessment indicates areas that warrant further evaluation. We recommend consulting with a reading specialist or educational psychologist.",
        }

        return ExplanationResponse(
            text=fallbacks.get(request.risk_level, fallbacks["medium"]),
            provider="fallback",
            model="rules-based",
            tokens_used=0,
            latency_ms=0,
        )

    def generate_batch(
        self, requests: List[ExplanationRequest]
    ) -> List[ExplanationResponse]:
        responses = []
        for req in requests:
            responses.append(self.generate(req))
        return responses


class ExplanationReportGenerator:
    def __init__(self, llm_generator: LLMExplanationGenerator):
        self.llm = llm_generator

    def generate_report(
        self,
        student_id: str,
        speech_result: Optional[Dict] = None,
        handwriting_result: Optional[Dict] = None,
        text_result: Optional[Dict] = None,
        student_context: Optional[Dict] = None,
    ) -> Dict:
        modalities = []

        if speech_result:
            modalities.append(
                {
                    "modality": "speech",
                    "score": speech_result.get("score", 0),
                    "risk": speech_result.get("risk", "low"),
                    "features": speech_result.get("top_features", [])[:5],
                }
            )

        if handwriting_result:
            modalities.append(
                {
                    "modality": "handwriting",
                    "score": handwriting_result.get("score", 0),
                    "risk": handwriting_result.get("risk", "low"),
                    "features": handwriting_result.get("top_features", [])[:5],
                }
            )

        if text_result:
            modalities.append(
                {
                    "modality": "text",
                    "score": text_result.get("score", 0),
                    "risk": text_result.get("risk", "low"),
                    "features": text_result.get("top_features", [])[:5],
                }
            )

        combined_score = (
            sum(m["score"] for m in modalities) / len(modalities) if modalities else 0
        )

        if combined_score < 0.33:
            overall_risk = "low"
        elif combined_score < 0.66:
            overall_risk = "medium"
        else:
            overall_risk = "high"

        all_features = []
        for m in modalities:
            for f in m["features"]:
                all_features.append(
                    {
                        "feature": f"{m['modality']}: {f.get('feature', f.get('word', 'factor'))}",
                        "importance": f.get("importance", f.get("weight", 0)),
                    }
                )

        all_features.sort(key=lambda x: x["importance"], reverse=True)

        request = ExplanationRequest(
            modality="combined",
            scores={m["modality"]: m["score"] for m in modalities},
            risk_level=overall_risk,
            top_features=all_features[:10],
            student_context=student_context,
        )

        llm_response = self.llm.generate(request)

        report = {
            "student_id": student_id,
            "overall_risk": overall_risk,
            "combined_score": combined_score,
            "modalities": modalities,
            "explanation": llm_response.text,
            "explanation_metadata": {
                "provider": llm_response.provider,
                "model": llm_response.model,
                "tokens_used": llm_response.tokens_used,
                "latency_ms": llm_response.latency_ms,
            },
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        return report

    def save_report(self, report: Dict, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved to {output_path}")


def create_llm_generator(
    provider: str = "openai",
    api_key: Optional[str] = None,
    model: str = "gpt-3.5-turbo",
) -> LLMExplanationGenerator:
    provider_enum = LLMProvider(provider.lower())

    return LLMExplanationGenerator(provider=provider_enum, api_key=api_key, model=model)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="LLM Explanation Generation")
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "anthropic", "huggingface", "local"],
    )
    parser.add_argument("--model", default="gpt-3.5-turbo")
    parser.add_argument("--speech-score", type=float)
    parser.add_argument("--handwriting-score", type=float)
    parser.add_argument("--text-score", type=float)
    parser.add_argument(
        "--risk-level", required=True, choices=["low", "medium", "high"]
    )
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    generator = create_llm_generator(args.provider, model=args.model)

    request = ExplanationRequest(
        modality="combined",
        scores={
            "speech": args.speech_score or 0,
            "handwriting": args.handwriting_score or 0,
            "text": args.text_score or 0,
        },
        risk_level=args.risk_level,
        top_features=[],
    )

    response = generator.generate(request)

    print(f"Explanation: {response.text}")

    with open(args.output, "w") as f:
        json.dump(
            {
                "text": response.text,
                "provider": response.provider,
                "model": response.model,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
