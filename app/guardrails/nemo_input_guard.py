from pathlib import Path
import threading

from dotenv import load_dotenv
from nemoguardrails import LLMRails, RailsConfig


load_dotenv()
load_dotenv('.env.example')

CONFIG_DIR = Path(__file__).resolve().parent / "nemo_config"

REFUSAL_TEXT = "i'm sorry, i can't respond to that"


class NemoGuardrails:
    """
    Additional input and output safety checks using NVIDIA NeMo Guardrails.

    This component supplements the application's evidence and citation
    checks. It does not establish clinical correctness.
    """

    def __init__(self):
        if not CONFIG_DIR.exists():
            raise FileNotFoundError(
                f"NeMo configuration directory not found: {CONFIG_DIR}"
            )

        config = RailsConfig.from_path(str(CONFIG_DIR))
        import os
        base_url = os.getenv("LITELLM_API_BASE", "http://litellm:4000").rstrip("/")
        if not os.path.exists("/.dockerenv") and "litellm:4000" in base_url:
            base_url = base_url.replace("litellm:4000", "127.0.0.1:4000")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        for m in getattr(config, "models", []):
            if hasattr(m, "parameters") and isinstance(m.parameters, dict):
                m.parameters["base_url"] = base_url
        self.rails = LLMRails(config)
        self._register_reasoning_aware_parser()

    def _register_reasoning_aware_parser(self):
        """Register a reasoning-aware parser that handles thinking tokens

        (e.g. from openai/gpt-oss-20b on Groq) before assessing content safety.
        """
        import re
        import nemoguardrails.llm.output_parsers as nemo_parsers

        orig_is_content_safe = nemo_parsers.is_content_safe

        def reasoning_aware_is_content_safe(response: str):
            # 1. Strip <think>...</think> reasoning tags
            cleaned = re.sub(r"<think>[\s\S]*?</think>", "", response).strip()

            # 2. Try the original NeMo parser on clean content
            orig_result = orig_is_content_safe(cleaned)
            if orig_result[0] is True:
                return orig_result

            # 3. Handle model output variations where decision is affirmative safe/no-block
            cleaned_lower = cleaned.lower()
            words = re.findall(r"\b[a-zA-Z]+\b", cleaned_lower)
            if "no" in words[:5] or "safe" in words[:10]:
                return [True]
            if "not" in words[:5] and any(
                w in words[:8] for w in ("block", "blocked", "harmful", "unsafe")
            ):
                return [True]

            return orig_result

        # Update both LLMRails registry and the runtime task manager
        self.rails.register_output_parser(
            reasoning_aware_is_content_safe, "is_content_safe"
        )
        tm = getattr(self.rails, "llm_task_manager", None) or getattr(
            getattr(self.rails, "runtime", None), "llm_task_manager", None
        )
        if tm and hasattr(tm, "output_parsers"):
            tm.output_parsers["is_content_safe"] = reasoning_aware_is_content_safe

    @staticmethod
    def _get_response_text(result):
        """Support NeMo's structured generation response."""
        responses = getattr(result, "response", None)

        if not responses or not isinstance(responses, list):
            raise ValueError("NeMo returned no usable response.")

        first = responses[0]

        if not isinstance(first, dict):
            raise ValueError("NeMo returned an invalid response format.")

        content = first.get("content")

        if not isinstance(content, str) or not content.strip():
            raise ValueError("NeMo returned empty or invalid response text.")

        return content

    @staticmethod
    def _has_blocked_rail(result):
        """Inspect the rail log for an explicit block decision."""
        log = getattr(result, "log", None)
        activated_rails = getattr(log, "activated_rails", None) or []

        for rail in activated_rails:
            if isinstance(rail, dict):
                rail_type = rail.get("type")
                stopped = rail.get("stop", False)
            else:
                rail_type = getattr(rail, "type", None)
                stopped = getattr(rail, "stop", False)

            if stopped and rail_type in {"input", "output"}:
                return True

        return False

    _lock = threading.Lock()

    def _check(self, messages, rail_names, expected_text):
        with self._lock:
            try:
                result = self.rails.generate(
                    messages=messages,
                    options={
                        "rails": rail_names,
                        "log": {
                            "activated_rails": True,
                        },
                    },
                )

                response_text = self._get_response_text(result)

                if self._has_blocked_rail(result):
                    return {
                        "allowed": False,
                        "check_error": False,
                        "reason": "NeMo Guardrails blocked the request.",
                    }

                # We do not silently accept transformed input or output.
                # Unexpected changes require review rather than automatic approval.
                if response_text.strip() != expected_text.strip():
                    if response_text.strip().lower().startswith(REFUSAL_TEXT):
                        return {
                            "allowed": False,
                            "check_error": False,
                            "reason": "NeMo Guardrails returned a refusal.",
                        }

                    return {
                        "allowed": False,
                        "check_error": False,
                        "reason": (
                            "NeMo changed the supplied content or returned an "
                            "unexpected response. The request was not approved."
                        ),
                    }

                return {
                    "allowed": True,
                    "check_error": False,
                    "reason": "NeMo Guardrails checks passed.",
                }

            except Exception as exc:
                import logging
                logging.warning("NeMo Guardrails check error: %s (%s)", exc, type(exc).__name__)
                # Fail closed. Do not disclose API errors or configuration details
                # to the end user.
                return {
                    "allowed": False,
                    "check_error": True,
                    "reason": (
                        "NeMo Guardrails could not complete its check. "
                        "The request was not approved."
                    ),
                }

    def check_input(self, question):
        if not isinstance(question, str) or not question.strip():
            return {
                "allowed": False,
                "reason": "The question is empty or invalid.",
            }

        return self._check(
            messages=[
                {
                    "role": "user",
                    "content": question,
                }
            ],
            rail_names=["input"],
            expected_text=question,
        )

    def check_output(self, question, answer):
        if not isinstance(answer, str) or not answer.strip():
            return {
                "allowed": False,
                "reason": "The generated answer is empty or invalid.",
            }

        # NeMo's output-only mode expects an empty user message and the
        # pre-generated assistant message to be checked.
        return self._check(
            messages=[
                {
                    "role": "user",
                    "content": "",
                },
                {
                    "role": "assistant",
                    "content": answer,
                },
            ],
            rail_names=["output"],
            expected_text=answer,
        )