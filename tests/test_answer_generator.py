from unittest.mock import MagicMock, patch
import pytest

from app.generation.answer_generator import (
    ClinicalAnswerGenerator,
    ContextLimitExceededError,
    ModelContextLimitMissingError,
    establish_context_limit,
    resolve_gateway_model_and_config,
)
from app.generation.answer_pipeline import AnswerPipeline


@pytest.fixture
def sample_passages():
    return [
        {
            "chunk_id": "chunk-1",
            "filename": "who_type2_diabetes.pdf",
            "page": 12,
            "text": "Common symptoms of type 2 diabetes include excessive thirst, frequent urination, fatigue, and blurred vision.",
        },
        {
            "chunk_id": "chunk-2",
            "filename": "who_type2_diabetes.pdf",
            "page": 13,
            "text": "Long-term clinical findings and complications include retinopathy, chronic kidney disease, and diabetic neuropathy.",
        },
    ]


def test_requests_that_fit(sample_passages):
    """Test 1: Requests within context limit succeed and call LiteLLM safely."""
    generator = ClinicalAnswerGenerator(
        model="gpt-oss-20b",
        api_key="sk-test-key",
        max_context_tokens=10000,
        max_output_tokens=1000,
    )

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content="[S1] Common symptoms include thirst, frequent urination, and fatigue. [S2] Complications include retinopathy and kidney disease."
            ),
            finish_reason="stop",
        )
    ]

    with patch("app.generation.answer_generator.completion", return_value=mock_response) as mock_completion:
        result = generator.generate(
            question="What symptoms and complications are associated with type 2 diabetes?",
            passages=sample_passages,
        )

        # LiteLLM was called once
        assert mock_completion.call_count == 1
        call_kwargs = mock_completion.call_args[1]

        # Verify messages sent to LiteLLM contain complete prompt
        messages = call_kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "who_type2_diabetes.pdf" in messages[1]["content"]
        assert "[S1]" in messages[1]["content"]
        assert "[S2]" in messages[1]["content"]

        # Output reservation was passed
        assert call_kwargs["max_completion_tokens"] == 1000

        # Response payload assertions
        assert "Common symptoms include thirst" in result["answer"]
        assert result["citation_check"]["valid"] is True
        assert result["token_stats"]["total_required_tokens"] <= 10000
        assert result["token_stats"]["prompt_tokens"] > 0
        assert result["token_stats"]["max_output_tokens"] == 1000


def test_requests_that_exceed_limit(sample_passages):
    """Test 2: Requests exceeding context limit are safely rejected before LiteLLM is called."""
    # Set context limit tight enough that prompt tokens + output reservation exceeds limit
    generator = ClinicalAnswerGenerator(
        model="gpt-oss-20b",
        api_key="sk-test-key",
        max_context_tokens=1500,
        max_output_tokens=1400,  # Prompt will be ~250 tokens, so 250 + 1400 = 1650 > 1500
    )

    with patch("app.generation.answer_generator.completion") as mock_completion:
        with pytest.raises(ContextLimitExceededError) as exc_info:
            generator.generate(
                question="What symptoms and complications are associated with type 2 diabetes according to WHO?",
                passages=sample_passages,
            )

        # Crucial: LiteLLM completion must NEVER have been called
        assert mock_completion.call_count == 0

        # Validate exception attributes
        err = exc_info.value
        assert err.prompt_tokens > 0
        assert err.max_output_tokens == 1400
        assert err.max_context_tokens == 1500
        assert err.prompt_tokens + err.max_output_tokens > err.max_context_tokens
        assert "exceeds context limit of 1500 tokens" in str(err)


def test_missing_model_limit_configuration(sample_passages, monkeypatch):
    """Test 3: Missing model limit configuration safely halts request before calling LiteLLM."""
    monkeypatch.delenv("MODEL_CONTEXT_LIMIT", raising=False)
    with patch("app.generation.answer_generator.completion") as mock_completion:
        # Initializing generator with an unmapped model that has no context limit
        with pytest.raises(ModelContextLimitMissingError) as exc_info:
            ClinicalAnswerGenerator(
                model="custom-unmapped-clinical-model",
                api_key="sk-test-key",
                # no explicit limit provided
            )

        assert "Context limit configuration is missing" in str(exc_info.value)
        assert exc_info.value.model == "custom-unmapped-clinical-model"

        # Even with lazy check, generate must safely reject before calling LiteLLM
        generator = ClinicalAnswerGenerator(
            model="custom-unmapped-clinical-model",
            api_key="sk-test-key",
            lazy_limit_check=True,
        )

        with pytest.raises(ModelContextLimitMissingError):
            generator.generate(
                question="What are symptoms of diabetes?",
                passages=sample_passages,
            )

        # LiteLLM was never called
        assert mock_completion.call_count == 0


def test_resolves_gateway_model_alias_and_limit(tmp_path, monkeypatch):
    """Test 4: Gateway alias 'gpt-oss-20b' resolves to underlying model and context limit."""
    cfg_file = tmp_path / "litellm_config.yaml"
    cfg_file.write_text(
        "model_list:\n"
        "  - model_name: gpt-oss-20b\n"
        "    litellm_params:\n"
        "      model: groq/gpt-oss-20b\n"
        "      max_tokens: 131072\n"
    )
    monkeypatch.setenv("LITELLM_CONFIG_PATH", str(cfg_file))

    underlying_model, config = resolve_gateway_model_and_config("gpt-oss-20b")
    assert underlying_model == "groq/gpt-oss-20b"
    assert config.get("model_name") == "gpt-oss-20b"

    limit = establish_context_limit("gpt-oss-20b", underlying_model, config)
    assert limit == 131072

    generator = ClinicalAnswerGenerator(
        model="gpt-oss-20b",
        api_key="sk-test-key",
    )
    assert generator.resolved_model == "groq/gpt-oss-20b"
    assert generator.max_context_tokens == 131072


def test_token_counting_includes_full_request_and_output_reservation(sample_passages):
    """Test 5: Token counting covers system prompt, user prompt, metadata, evidence, and output."""
    generator = ClinicalAnswerGenerator(
        model="gpt-oss-20b",
        api_key="sk-test-key",
        max_context_tokens=100000,
        max_output_tokens=3000,
    )

    context, sources = generator._build_context(sample_passages)
    messages = [
        {"role": "system", "content": "You are a healthcare clinical knowledge assistant."},
        {"role": "user", "content": f"Question: Symptoms?\n\nEvidence:\n{context}"},
    ]

    stats = generator.validate_prompt_size(messages)

    assert stats["prompt_tokens"] > 0
    assert stats["max_output_tokens"] == 3000
    assert stats["total_required_tokens"] == stats["prompt_tokens"] + 3000
    assert stats["max_context_tokens"] == 100000
    assert stats["available_tokens"] == 100000 - stats["total_required_tokens"]


def test_answer_pipeline_safe_handling_of_oversized_requests(sample_passages):
    """Test 6: End-to-end AnswerPipeline handles oversized requests safely."""
    generator = ClinicalAnswerGenerator(
        model="gpt-oss-20b",
        api_key="sk-test-key",
        max_context_tokens=100,  # Deliberately tiny
        max_output_tokens=50,
    )

    mock_retriever = MagicMock()
    mock_retriever.search.return_value = sample_passages

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = sample_passages

    mock_evidence_checker = MagicMock()
    mock_evidence_checker.check.return_value = {"sufficient": True, "reason": "Passed"}

    mock_answerability_checker = MagicMock()
    mock_answerability_checker.check.return_value = {"answerable": True, "reason": "Passed"}

    mock_nemo = MagicMock()
    mock_nemo.check_input.return_value = {"allowed": True, "reason": "Passed"}
    mock_nemo.check_output.return_value = {"allowed": True, "reason": "Passed"}

    pipeline = AnswerPipeline(
        retriever=mock_retriever,
        reranker=mock_reranker,
        evidence_checker=mock_evidence_checker,
        answerability_checker=mock_answerability_checker,
        generator=generator,
        nemo_guardrails=mock_nemo,
    )

    result = pipeline.run("What are symptoms of type 2 diabetes?", verbose=False)

    assert result["status"] == "error"
    assert "exceeded the model context limit" in result["answer"]

