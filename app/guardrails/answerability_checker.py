
import os
import json
import re
import logging

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
load_dotenv(".env.example")

logger = logging.getLogger(__name__)


class AnswerabilityChecker:
    """Assess whether retrieved passages address the user's question."""

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is missing from the .env file."
            )

        self.model = os.getenv(
            "GROQ_MODEL",
            "gpt-oss-20b"
        )

        self.client = Groq(api_key=api_key)

    @staticmethod
    def _fallback(reason):
        return {
            "answerable": False,
            "reason": reason
        }

    @staticmethod
    def _parse_response(content):
        if not content or not content.strip():
            raise ValueError("The model returned an empty response.")

        content = re.sub(
            r"<think>[\s\S]*?</think>",
            "",
            content
        ).strip()

        # Remove Markdown JSON fences.
        content = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            content,
            flags=re.IGNORECASE
        ).strip()

        result = json.loads(content)

        if not isinstance(result, dict):
            raise ValueError("The response must be a JSON object.")

        if not isinstance(result.get("answerable"), bool):
            raise ValueError(
                "The answerability result lacks a boolean answerable field."
            )

        reason = result.get("reason")

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(
                "The answerability result lacks a non-empty reason."
            )

        return {
            "answerable": result["answerable"],
            "reason": reason.strip()
        }

    def check(self, question: str, passages: list[dict]) -> dict:

        if not isinstance(question, str) or not question.strip():
            return self._fallback("The question is empty.")

        if not passages:
            return self._fallback(
                "No evidence passages were retrieved."
            )

        evidence = []

        for index, passage in enumerate(passages, start=1):
            title = passage.get("title", "")
            page = passage.get("page", "")

            header = (
                f"Title: {title} | Page: {page}\n"
                if title or page else ""
            )

            evidence.append(
                f"[S{index}]\n"
                f"{header}"
                f"{passage.get('text', '')}"
            )

        prompt = f"""
You are assessing evidence sufficiency for a clinical RAG system.

Question:
{question}

Retrieved evidence:
{chr(10).join(evidence)}

Assessment instructions:
1. Assess only the supplied evidence. Do not use outside medical knowledge.
2. Consider document titles, section headings, and surrounding context.
3. Do not assume that a finding associated with a disease or emergency
   is automatically a symptom of another condition mentioned nearby.
4. A symptom or clinical sign supports an answer only when the evidence
   explicitly connects it to the condition asked about, or establishes
   that relationship clearly in its surrounding context.
5. Distinguish symptoms, signs, complications, risk factors, and treatments.
6. Do not turn a general clinical association into a disease-specific claim.
7. If the evidence supports only part of the requested answer, classify
   it as answerable only when that partial answer is directly supported.
8. If the requested information is not sufficiently supported, classify
   it as insufficient.
9. Always provide a clear, non-empty reason.

Return only a valid JSON object:
{{
    "answerable": true,
    "reason": "Explain exactly what the evidence supports and any limitations."
}}

Use false when evidence is genuinely insufficient.
"""

        for attempt in range(4):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a conservative evidence sufficiency "
                                "assessor. Return valid JSON only."
                            )
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0,
                    max_completion_tokens=1200,
                    reasoning_effort="low",
                    reasoning_format="parsed"
                )

                message = response.choices[0].message
                content = message.content or ""

                return self._parse_response(content)

            except Exception as exc:
                err_str = str(exc).lower()
                if ("rate limit" in err_str or "429" in err_str or "tpm" in err_str) and attempt < 3:
                    import time
                    wait_sec = 16.0 * (attempt + 1)
                    logger.warning("Answerability rate limit hit. Waiting %.1fs...", wait_sec)
                    time.sleep(wait_sec)
                    continue

                logger.exception(
                    "Answerability assessment failed: %s",
                    type(exc).__name__
                )

                return self._fallback(
                    "The evidence assessment failed. "
                    "Answer generation was blocked."
                )
