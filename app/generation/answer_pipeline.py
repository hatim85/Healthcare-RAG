import json
import os
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import ClinicalReranker
from app.generation.answer_generator import (
    ClinicalAnswerGenerator,
    ContextLimitExceededError,
    ModelContextLimitMissingError,
    SYSTEM_PROMPT,
)
from app.guardrails.evidence_checker import EvidenceChecker
from app.guardrails.answerability_checker import AnswerabilityChecker
from app.guardrails.claim_verifier import ClaimVerifier
from app.guardrails.nemo_input_guard import NemoGuardrails
from app.guardrails.pii_guard import mask_pii
from app.caching.corpus_version import get_corpus_version
from app.guardrails.output_pii_guard import mask_output_pii
from app.caching.answer_cache import (
    answer_cache_enabled,
    get_cached_answer,
    make_evidence_fingerprint,
    set_cached_answer,
)
from app.retrieval.context_manager import ContextManager
from app.retrieval.quality_filter import EvidenceQualityFilter
from app.generation.context_metrics import ContextMetrics
from app.observability import ModelRouter, PipelineTracer, StructuredEventSink

def sanitize_passages(passages):
    """
    Mask PII in passage text and human readable source metadata.
    
    Preserve structural metadata such as page numbers, chunk IDs, retrieval scores, and other fields used by the RAG pipeline.
    """
    # Note: do not uncomment as we don't want for our application
    # Passages/citations are returned without PII redaction
    return passages

    # Note: do not uncomment as we don't want for our application
    # sanitized = []
    # fields_to_mask = (
    #     "text",
    #     "title",
    #     "filename",
    #     "source",
    #     "file_path",
    #     "document_title",
    # )
    # for passage in passages:
    #     clean_passage = dict(passage)
    #     for field in fields_to_mask:
    #         value = clean_passage.get(field)
    #         if isinstance(value, str) and value:
    #             result = mask_pii(value)
    #             clean_passage[field] = result["text"]
    #     sanitized.append(clean_passage)
    # return sanitized


class AnswerPipeline:
    """End-to-end clinical question-answering pipeline with safety and validation guardrails."""

    def __init__(
        self,
        retriever=None,
        reranker=None,
        evidence_checker=None,
        answerability_checker=None,
        generator=None,
        verifier=None,
        nemo_guardrails=None,
        context_manager=None,
        quality_filter=None,
        model_router=None,
        event_sink=None,
    ):
        self.retriever = retriever or HybridRetriever()
        self.reranker = reranker or ClinicalReranker()
        self.evidence_checker = evidence_checker or EvidenceChecker(
            min_passages=1,
            min_text_chars=80,
            min_rerank_score=None,
        )
        self.answerability_checker = answerability_checker or AnswerabilityChecker()
        self.generator = generator or ClinicalAnswerGenerator()
        self.verifier = verifier or ClaimVerifier()
        self.nemo_guardrails = nemo_guardrails or NemoGuardrails()
        self.context_manager = context_manager or ContextManager(
            max_evidence_tokens=5000
        )
        self.quality_filter = quality_filter or EvidenceQualityFilter(
            min_rerank_score=None,
            min_authority_score=0.0,
        )
        self.model_router = model_router or ModelRouter()
        self.event_sink = event_sink

    def run(
        self,
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
        session_id: str | None = None,
        verbose: bool = True,
        tracer: PipelineTracer | None = None,
        tenant_id: str | None = None,
        user_id: str | None = None,
    ) -> dict:
        """
        Execute the full question answering pipeline with end-to-end observability.

        Args:
            question: The clinical question to answer.
            conversation_history: Optional prior conversation turns [{"role": ..., "content": ...}].
            session_id: Optional session identifier for conversation tracking.
            verbose: Whether to print pipeline progress steps to stdout.
            tracer: Optional existing PipelineTracer instance for distributed correlation.
            tenant_id: Optional tenant identifier for data segregation.
            user_id: Optional user identifier for trace tracking.

        Returns:
            A dictionary containing answer, status, sources, check results, context metrics, and observability trace.
        """
        tracer = tracer or PipelineTracer(
            session_id=session_id,
            question=question,
            tenant_id=tenant_id or "default",
            user_id=user_id or "anonymous",
            event_sink=self.event_sink,
        )

        question = (question or "").strip()
        if not question:
            if verbose:
                print("Please enter a question.")
            tracer.finish(status="error", reason="Empty question provided.")
            return {
                "status": "error",
                "answer": "Please enter a valid question.",
                "question": question,
                "sources": [],
                "observability": tracer.to_dict(),
            }

        # Step 0: Process and prune conversation history (sliding window)
        valid_history: list[dict[str, str]] = []
        history_tokens = 0
        if conversation_history:
            for turn in conversation_history:
                if isinstance(turn, dict) and "role" in turn and "content" in turn:
                    valid_history.append({
                        "role": str(turn["role"]),
                        "content": str(turn["content"]),
                    })

            # Sliding window: keep most recent turns within 2000 tokens
            total_h_tokens = sum(
                self.context_manager.count_tokens(t["content"]) for t in valid_history
            )
            while total_h_tokens > 2000 and len(valid_history) > 2:
                valid_history = valid_history[2:]  # drop oldest turn pair
                total_h_tokens = sum(
                    self.context_manager.count_tokens(t["content"]) for t in valid_history
                )
            history_tokens = total_h_tokens

        # Step 1: Detect and mask personal information in input
        if verbose:
            print("\n[1/8] Detecting and masking personal information...")

        with tracer.stage("input_pii_masking"):
            try:
                pii_result = mask_pii(question)
            except Exception as exc:
                tracer.record_error("input_pii_masking", exc, recoverable=False)
                if verbose:
                    print(
                        "\nREQUEST STOPPED\n"
                        "Privacy checks could not be completed. Please try again."
                    )
                tracer.finish(status="error", reason=str(exc))
                return {
                    "status": "error",
                    "answer": "Privacy checks could not be completed. Please try again.",
                    "question": question,
                    "sources": [],
                    "observability": tracer.to_dict(),
                }

        masked_question = pii_result["text"]
        if verbose:
            if pii_result["pii_detected"]:
                print("Potential PII detected and masked.")
                print("Entity types:", ", ".join(pii_result["entity_types"]))
                print("Entity count:", pii_result["entity_count"])
            else:
                print("No PII detected by the configured recognizers.")

        # Step 2: Input safety guardrails
        if verbose:
            print("\n[2/8] Applying NeMo input safety rails...")

        with tracer.stage("nemo_input_guard"):
            input_result = self.nemo_guardrails.check_input(masked_question)

        if verbose:
            print("Input safety:", input_result["reason"])

        if not input_result["allowed"]:
            if verbose:
                print("\nREQUEST BLOCKED")
                print(
                    "The request could not pass the input safety checks. "
                    "Please review the request and try again."
                )
            tracer.finish(status="blocked", reason=input_result.get("reason"))
            return {
                "status": "blocked",
                "answer": (
                    "The request could not pass the input safety checks. "
                    "Please review the request and try again."
                ),
                "reason": input_result.get("reason"),
                "question": question,
                "sources": [],
                "observability": tracer.to_dict(),
            }

        # Step 3: Retrieval
        if verbose:
            print("\n[3/8] Retrieving candidate passages...")

        allow_retrieval_cache = (not pii_result["pii_detected"] and not valid_history)
        if not allow_retrieval_cache:
            bypass_reason = "pii_detected" if pii_result["pii_detected"] else "multi_turn_history"
            tracer.record_cache_access("retrieval", "bypass", reason=bypass_reason)

        with tracer.stage("retrieval"):
            candidates = self.retriever.search(
                masked_question, 
                top_k=20,
                allow_cache=allow_retrieval_cache,
                tenant_id=tenant_id,
            )

        if not candidates:
            msg = (
                "I cannot answer this question because no relevant "
                "evidence was retrieved from the available documents."
            )
            if verbose:
                print(f"\n{msg}")
            tracer.finish(status="insufficient_evidence", reason=msg)
            return {
                "status": "insufficient_evidence",
                "answer": msg,
                "question": question,
                "sources": [],
                "observability": tracer.to_dict(),
            }

        if verbose:
            print(f"Retrieved {len(candidates)} candidate passages.")

        # Step 4: Reranking
        if verbose:
            print("\n[4/8] Reranking passages...")

        with tracer.stage("reranking"):
            passages = self.reranker.rerank(
                question=masked_question,
                candidates=candidates,
                top_k=3,
            )

        if not passages:
            if verbose:
                print("\nINSUFFICIENT EVIDENCE\nNo passages survived reranking.")
            tracer.finish(status="insufficient_evidence", reason="No passages survived reranking.")
            return {
                "status": "insufficient_evidence",
                "answer": "No passages survived reranking.",
                "question": question,
                "sources": [],
                "observability": tracer.to_dict(),
            }

        if verbose:
            print(f"Selected {len(passages)} passages.")

        # Step 4.5: Document Freshness & Source Quality Filtering
        with tracer.stage("quality_filter"):
            quality_result = self.quality_filter.filter(passages)
            filtered_passages = quality_result["filtered_passages"]

            # Compute and record comprehensive retrieval quality metrics
            tracer.record_retrieval_quality(
                candidates=candidates,
                reranked_passages=passages,
                retained_passages=filtered_passages,
            )
            passages = filtered_passages

        if not passages:
            if verbose:
                print(
                    "\nINSUFFICIENT EVIDENCE\n"
                    "All candidate passages were filtered out due to quality, authority, or freshness criteria."
                )
            tracer.finish(
                status="insufficient_evidence",
                reason="All candidate passages filtered out by quality criteria.",
            )
            return {
                "status": "insufficient_evidence",
                "answer": "All candidate passages were filtered out due to quality, authority, or freshness criteria.",
                "question": question,
                "sources": [],
                "observability": tracer.to_dict(),
            }

        # Step 5: Sanitize retrieved passages for PII
        # Note: do not uncomment as we don't want for our application
        # if verbose:
        #     print("\n[PII] Sanitizing retrieved passages...")
        #
        # with tracer.stage("passage_pii_sanitizing"):
        #     try:
        #         passages = sanitize_passages(passages)
        #     except Exception as exc:
        #         tracer.record_error("passage_pii_sanitizing", exc, recoverable=False)
        #         if verbose:
        #             print(
        #                 "\nREQUEST STOPPED\n"
        #                 "Privacy checks on retrieved evidence could not be completed."
        #                 "The evidence will not be sent to external models."
        #             )
        #         tracer.finish(status="error", reason=str(exc))
        #         return {
        #             "status": "error",
        #             "answer": "Privacy checks on retrieved evidence could not be completed.",
        #             "question": question,
        #             "sources": [],
        #             "observability": tracer.to_dict(),
        #         }
        #
        # if verbose:
        #     print("Retrieved passages passed the PII masking stage.")
            
        # Step 5.5: Dynamic Context Budgeting & Evidence Selection
        if verbose:
            print("\n[CONTEXT] Managing evidence token budget and source diversity...")

        with tracer.stage("context_budgeting"):
            model_limit = getattr(self.generator, "max_context_tokens", 131072) or 131072
            output_res = getattr(self.generator, "max_output_tokens", 3000) or 3000
            safety_margin = 500
            system_tokens = self.context_manager.count_tokens(SYSTEM_PROMPT)
            question_tokens = self.context_manager.count_tokens(masked_question)

            dynamic_evidence_budget = self.context_manager.calculate_dynamic_budget(
                model_context_limit=model_limit,
                system_tokens=system_tokens,
                question_tokens=question_tokens,
                output_reservation=output_res,
                safety_margin=safety_margin,
                history_tokens=history_tokens,
            )

            try:
                context_result = self.context_manager.build_context(
                    question=masked_question,
                    passages=passages,
                    max_evidence_tokens=dynamic_evidence_budget,
                )

                passages = context_result["selected_passages"]

            except Exception as exc:
                tracer.record_error("context_budgeting", exc, recoverable=False)
                if verbose:
                    print(
                        f"\nREQUEST STOPPED\n"
                        f"Context-window management failed: {exc}. "
                        f"Evidence will not be sent to the generator."
                    )
                tracer.finish(status="error", reason=str(exc))
                return {
                    "status": "error",
                    "answer": (
                        "The evidence could not be prepared safely "
                        "for answer generation."
                    ),
                    "question": question,
                    "sources": [],
                    "observability": tracer.to_dict(),
                }

            metrics = ContextMetrics(
                model=str(getattr(self.generator, "model", "Unknown")),
                model_context_limit=model_limit,
                system_tokens=system_tokens,
                question_tokens=question_tokens,
                history_tokens=history_tokens,
                output_reservation=output_res,
                safety_margin=safety_margin,
                allocated_evidence_budget=dynamic_evidence_budget,
                used_evidence_tokens=context_result["total_tokens"],
                raw_evidence_tokens=context_result.get("raw_tokens", context_result["total_tokens"]),
                input_passages_count=context_result["input_passage_count"],
                selected_passages_count=context_result["selected_passage_count"],
                excluded_duplicates=context_result["excluded_duplicates"],
                excluded_budget=context_result["excluded_budget"],
                excluded_diversity=context_result.get("excluded_diversity", 0),
                excluded_redundancy=context_result.get("excluded_redundancy", 0),
                excluded_low_rerank=quality_result.get("excluded_low_rerank", 0),
                excluded_low_authority=quality_result.get("excluded_low_authority", 0),
                excluded_outdated=quality_result.get("excluded_outdated", 0),
                clinical_caveats_protected=context_result.get("clinical_caveats_protected", 0),
                compression_applied=context_result.get("compression_applied", False),
                compression_ratio=context_result.get("compression_ratio", 1.0),
            )
            metrics.compute_ratios()
            tracer.record_tokens(metrics.to_dict())

        if verbose:
            print("\n[CONTEXT METRICS]")
            print(metrics.format_summary())

        # Step 6: Evidence check & answerability check
        if verbose:
            print("\n[5/8] Checking retrieved evidence...")

        with tracer.stage("evidence_check"):
            evidence_result = self.evidence_checker.check(passages)
        if verbose:
            print(f"Evidence check: {evidence_result['reason']}")

        if not evidence_result["sufficient"]:
            if verbose:
                print(
                    "\nINSUFFICIENT EVIDENCE\n"
                    "The available passages did not pass the preliminary checks."
                    "I cannot provide a grounded answer from these passages."
                )
            tracer.finish(status="insufficient_evidence", reason=evidence_result.get("reason"))
            return {
                "status": "insufficient_evidence",
                "answer": (
                    "The available passages did not pass the preliminary checks. "
                    "I cannot provide a grounded answer from these passages."
                ),
                "reason": evidence_result.get("reason"),
                "question": question,
                "sources": passages,
                "context_metrics": metrics.to_dict(),
                "observability": tracer.to_dict(),
            }

        if verbose:
            print("\n[6/8] Assessing whether the evidence answers the question...")

        with tracer.stage("answerability_check"):
            answerability_result = self.answerability_checker.check(
                question=masked_question,
                passages=passages,
            )
        if verbose:
            print("Answerability assessment:", answerability_result["reason"])

        if not answerability_result["answerable"]:
            if verbose:
                print("\nINSUFFICIENT EVIDENCE")
                print("I cannot provide a grounded answer from the available documents.")
            tracer.finish(status="insufficient_evidence", reason=answerability_result.get("reason"))
            return {
                "status": "insufficient_evidence",
                "answer": "I cannot provide a grounded answer from the available documents.",
                "reason": answerability_result.get("reason"),
                "question": question,
                "sources": passages,
                "context_metrics": metrics.to_dict(),
                "observability": tracer.to_dict(),
            }
        
        # Answer cache: consult if single turn, cache enabled, no PII detected
        cache_allowed = (
            answer_cache_enabled()
            and not pii_result["pii_detected"]
            and not valid_history
        )

        model_version = str(
            getattr(self.generator, "model", "")
        )
        prompt_version = os.getenv("ANSWER_PROMPT_VERSION", "")
        retrieval_version = os.getenv("ANSWER_RETRIEVAL_VERSION", "")
        corpus_version = get_corpus_version()
        access_scope = os.getenv("ANSWER_CACHE_ACCESS_SCOPE", "local")

        evidence_fingerprint = make_evidence_fingerprint(passages)

        with tracer.stage("answer_cache"):
            if not cache_allowed:
                reason = (
                    "disabled"
                    if not answer_cache_enabled()
                    else ("pii_detected" if pii_result["pii_detected"] else "multi_turn_history")
                )
                tracer.record_cache_access(
                    "answer", "bypass", reason=reason, key_hash=evidence_fingerprint
                )
                cached_result = None
            else:
                cached_result = get_cached_answer(
                    question=masked_question,
                    model_version=model_version,
                    prompt_version=prompt_version,
                    corpus_version=corpus_version,
                    retrieval_version=retrieval_version,
                    evidence_fingerprint=evidence_fingerprint,
                    access_scope=access_scope,
                    allow_cache=cache_allowed,
                )
                if cached_result is not None:
                    tracer.record_cache_access(
                        "answer", "hit", key_hash=evidence_fingerprint, latency_saved_ms=2500.0
                    )
                else:
                    tracer.record_cache_access(
                        "answer", "miss", key_hash=evidence_fingerprint
                    )

        if cached_result is not None:
            result = cached_result
            if verbose:
                print("Answer cache: matching validated generation found.")
        else:
            result = None
            if verbose:
                print("Answer cache: miss or disabled.")

        # Step 6.8: Model Routing (complexity and evidence based)
        with tracer.stage("model_routing"):
            route_decision = self.model_router.route(
                question=masked_question,
                passages=passages,
                history_turns=len(valid_history),
            )
            tracer.record_model_route(route_decision)

        if verbose:
            print(
                f"\n[MODEL ROUTER] Selected: {route_decision.selected_model} "
                f"(Complexity: {route_decision.complexity_level}, Score: {route_decision.complexity_score})"
            )

        # Step 7: Answer generation
        if verbose:
            print("\n[7/8] Generating an evidence-grounded answer...")

        if result is None:
            with tracer.stage("answer_generation"):
                try:
                    result = self.generator.generate(
                        question=masked_question,
                        passages=passages,
                        conversation_history=valid_history if valid_history else None,
                        model=route_decision.selected_model,
                    )
                except ContextLimitExceededError as exc:
                    # Attempt fallback route if fallback_model differs
                    if (
                        route_decision.fallback_model
                        and route_decision.fallback_model != route_decision.selected_model
                    ):
                        if verbose:
                            print(
                                f"Primary model '{route_decision.selected_model}' context limit reached; "
                                f"attempting fallback to '{route_decision.fallback_model}'..."
                            )
                        route_decision.fallback_occurred = True
                        tracer.record_model_route(route_decision)
                        try:
                            result = self.generator.generate(
                                question=masked_question,
                                passages=passages,
                                conversation_history=valid_history if valid_history else None,
                                model=route_decision.fallback_model,
                            )
                        except Exception as fallback_exc:
                            tracer.record_error("answer_generation_fallback", fallback_exc, recoverable=False)
                            tracer.finish(status="error", reason=str(fallback_exc))
                            return {
                                "status": "error",
                                "answer": "The request exceeded the model context limit.",
                                "error": str(fallback_exc),
                                "question": question,
                                "sources": passages,
                                "context_metrics": metrics.to_dict(),
                                "observability": tracer.to_dict(),
                            }
                    else:
                        tracer.record_error("answer_generation", exc, recoverable=False)
                        if verbose:
                            print(
                                f"\nREQUEST STOPPED\n"
                                f"Prompt size exceeded model context limit: {exc}"
                            )
                        tracer.finish(status="error", reason=str(exc))
                        return {
                            "status": "error",
                            "answer": "The request exceeded the model context limit.",
                            "error": str(exc),
                            "question": question,
                            "sources": passages,
                            "context_metrics": metrics.to_dict(),
                            "observability": tracer.to_dict(),
                        }
                except ModelContextLimitMissingError as exc:
                    tracer.record_error("answer_generation", exc, recoverable=False)
                    if verbose:
                        print(
                            f"\nREQUEST STOPPED\n"
                            f"Model context limit configuration is missing: {exc}"
                        )
                    tracer.finish(status="error", reason=str(exc))
                    return {
                        "status": "error",
                        "answer": "Model context limit configuration is missing.",
                        "error": str(exc),
                        "question": question,
                        "sources": passages,
                        "context_metrics": metrics.to_dict(),
                        "observability": tracer.to_dict(),
                    }

        if result and "answer" in result:
            tracer.record_generation(
                name="answer_generation",
                model=result.get("model", route_decision.selected_model),
                input_prompt=masked_question,
                output_text=result.get("answer", ""),
            )

        # Step 7: Output PII scan
        # Note: Do not uncomment as we don't want for our application
        # if verbose:
        #     print("\n[PII] Scanning generated answer...")
        #
        # with tracer.stage("output_pii_scan"):
        #     try:
        #         answer_pii = mask_output_pii(result["answer"])
        #     except Exception as exc:
        #         tracer.record_error("output_pii_scan", exc, recoverable=False)
        #         if verbose:
        #             print(
        #                 "\nANSWER WITHHELD\n"
        #                 "The final answer could not pass the privacy check."
        #             )
        #         tracer.finish(status="withheld", reason="Output privacy check failed.")
        #         return {
        #             "status": "withheld",
        #             "answer": "The final answer could not pass the privacy check.",
        #             "question": question,
        #             "sources": result.get("sources", []),
        #             "observability": tracer.to_dict(),
        #         }
        #
        # result["answer"] = answer_pii["text"]
        #
        # if verbose:
        #     if answer_pii["pii_detected"]:
        #         print("Potential PII detected and masked in the generated answer.")
        #         print("Entity types:", ", ".join(answer_pii["entity_types"]))
        #         print("Entity count:", answer_pii["entity_count"])
        #     else:
        #         print("No PII detected by the configured recognizers.")

        # Step 8: Output safety rails & claim verification
        if verbose:
            print("\n[8/8] Checking generated answer with NeMo output rails...")

        with tracer.stage("nemo_output_guard"):
            output_result = self.nemo_guardrails.check_output(
                question=masked_question,
                answer=result["answer"],
            )

        if verbose:
            print("Output safety:", output_result["reason"])

        if not output_result["allowed"]:
            if verbose:
                print("\nANSWER WITHHELD")
                print("The generated answer could not pass the output safety checks.")
            tracer.finish(status="withheld", reason=output_result.get("reason"))
            return {
                "status": "withheld",
                "answer": "The generated answer could not pass the output safety checks.",
                "reason": output_result.get("reason"),
                "question": question,
                "sources": result.get("sources", []),
                "observability": tracer.to_dict(),
            }

        citation_check = result.get("citation_check", {})
        tracer.record_score(
            name="citation_validity",
            value=1.0 if citation_check.get("valid", False) else 0.0,
            comment=citation_check.get("reason", "Citation validation check"),
        )
        if not citation_check.get("valid", False):
            if verbose:
                print("\nANSWER WITHHELD")
                print("The generated answer failed citation validation.")
                print(citation_check.get("reason", "Citation validation failed."))
            tracer.finish(status="withheld", reason="Citation validation failed.")
            return {
                "status": "withheld",
                "answer": "The generated answer failed citation validation.",
                "citation_check": citation_check,
                "question": question,
                "sources": result.get("sources", []),
                "observability": tracer.to_dict(),
            }

        with tracer.stage("claim_verification"):
            verification = self.verifier.verify(
                answer=result["answer"],
                sources=result["sources"],
            )

        tracer.record_score(
            name="claim_faithfulness",
            value=1.0 if verification.get("passed", False) else 0.0,
            comment=verification.get("reason", "Claim verification check"),
        )
        tracer.record_score(
            name="retrieval_quality",
            value=tracer.retrieval_quality.retrieval_quality_score,
            comment="Composite retrieval quality score",
        )

        if verbose:
            print("\nClaim verification:", verification["reason"])
            for index, claim in enumerate(verification["claims"], start=1):
                c_status = "SUPPORTED" if claim["supported"] else "NOT VERIFIED"
                print(f"\nClaim {index}: {c_status}")
                print("Text:", claim["claim"])
                print("Sources:", claim["cited_source_ids"])
                print("Explanation:", claim["explanation"])

        if not verification["passed"]:
            if verbose:
                print("\nANSWER WITHHELD")
                print("One or more claims could not be verified, or verification could not be completed.")
            tracer.finish(status="withheld", reason="Claim verification failed.")
            return {
                "status": "withheld",
                "answer": "One or more claims could not be verified, or verification could not be completed.",
                "claim_verification": verification,
                "citation_check": citation_check,
                "question": question,
                "sources": result.get("sources", []),
                "observability": tracer.to_dict(),
            }
        
        # Store only after output safety, citation validation, and
        # claim verification have all passed.
        if cache_allowed:
            set_cached_answer(
                question=masked_question,
                result=result,
                model_version=model_version,
                prompt_version=prompt_version,
                corpus_version=corpus_version,
                retrieval_version=retrieval_version,
                evidence_fingerprint=evidence_fingerprint,
                access_scope=access_scope,
                verified=verification.get("passed") is True,
                allow_cache=cache_allowed,
            )

        updated_history = list(valid_history)
        updated_history.append({"role": "user", "content": question})
        updated_history.append({"role": "assistant", "content": result["answer"]})

        obs_report = tracer.finish(status="success")
        if verbose:
            print("\n" + tracer.format_summary())

        return {
            "status": "success",
            "answer": result["answer"],
            "question": question,
            "sources": result["sources"],
            "citation_check": citation_check,
            "claim_verification": verification,
            "model": result.get("model", "Unknown"),
            "model_route": tracer.model_route,
            "context_metrics": metrics.to_dict(),
            "conversation_history": updated_history,
            "observability": obs_report,
        }

    def run_stream(
        self,
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
        session_id: str | None = None,
        verbose: bool = False,
        tracer: PipelineTracer | None = None,
        tenant_id: str | None = None,
        user_id: str | None = None,
    ):
        """Execute the question answering pipeline and yield Server-Sent Events (SSE).

        Yields:
            str: SSE formatted events:
                - `event: metadata` (sources, model, context metrics)
                - `event: token` (streamed token chunks)
                - `event: done` (final verified result with observability and citations)
                - `event: blocked` (safety filter blocked)
                - `event: insufficient_evidence` (retrieval/evidence check failed)
                - `event: withheld` (safety or verification withheld output)
                - `event: error` (unhandled execution failure)
        """
        tracer = tracer or PipelineTracer(
            session_id=session_id,
            question=question,
            tenant_id=tenant_id or "default",
            user_id=user_id or "anonymous",
            event_sink=self.event_sink,
        )

        question = (question or "").strip()
        if not question:
            tracer.finish(status="error", reason="Empty question provided.")
            yield (
                "event: error\n"
                f"data: {json.dumps({'status': 'error', 'detail': 'Please enter a valid question.', 'question': question})}\n\n"
            )
            return

        # Step 0: Process and prune conversation history
        valid_history: list[dict[str, str]] = []
        history_tokens = 0
        if conversation_history:
            for turn in conversation_history:
                if isinstance(turn, dict) and "role" in turn and "content" in turn:
                    valid_history.append({
                        "role": str(turn["role"]),
                        "content": str(turn["content"]),
                    })

            total_h_tokens = sum(
                self.context_manager.count_tokens(t["content"]) for t in valid_history
            )
            while total_h_tokens > 2000 and len(valid_history) > 2:
                valid_history = valid_history[2:]
                total_h_tokens = sum(
                    self.context_manager.count_tokens(t["content"]) for t in valid_history
                )
            history_tokens = total_h_tokens

        # Step 1: Input PII Masking
        with tracer.stage("input_pii_masking"):
            try:
                pii_result = mask_pii(question)
            except Exception as exc:
                tracer.record_error("input_pii_masking", exc, recoverable=False)
                tracer.finish(status="error", reason=str(exc))
                yield (
                    "event: error\n"
                    f"data: {json.dumps({'status': 'error', 'detail': 'Privacy checks could not be completed.', 'question': question})}\n\n"
                )
                return

        masked_question = pii_result["text"]

        # Step 2: NeMo input safety rails
        with tracer.stage("nemo_input_guard"):
            input_result = self.nemo_guardrails.check_input(masked_question)

        if not input_result["allowed"]:
            tracer.finish(status="blocked", reason=input_result.get("reason"))
            yield (
                "event: blocked\n"
                f"data: {json.dumps({'status': 'blocked', 'answer': 'The request could not pass the input safety checks.', 'reason': input_result.get('reason'), 'question': question, 'sources': []})}\n\n"
            )
            return

        # Step 3: Retrieval
        allow_retrieval_cache = (not pii_result["pii_detected"] and not valid_history)
        if not allow_retrieval_cache:
            bypass_reason = "pii_detected" if pii_result["pii_detected"] else "multi_turn_history"
            tracer.record_cache_access("retrieval", "bypass", reason=bypass_reason)

        with tracer.stage("retrieval"):
            candidates = self.retriever.search(
                masked_question,
                top_k=20,
                allow_cache=allow_retrieval_cache,
                tenant_id=tenant_id,
            )

        if not candidates:
            msg = (
                "I cannot answer this question because no relevant "
                "evidence was retrieved from the available documents."
            )
            tracer.finish(status="insufficient_evidence", reason=msg)
            yield (
                "event: insufficient_evidence\n"
                f"data: {json.dumps({'status': 'insufficient_evidence', 'answer': msg, 'question': question, 'sources': []})}\n\n"
            )
            return

        # Step 4: Reranking
        with tracer.stage("reranking"):
            passages = self.reranker.rerank(
                question=masked_question,
                candidates=candidates,
                top_k=3,
            )

        if not passages:
            tracer.finish(status="insufficient_evidence", reason="No passages survived reranking.")
            yield (
                "event: insufficient_evidence\n"
                f"data: {json.dumps({'status': 'insufficient_evidence', 'answer': 'No passages survived reranking.', 'question': question, 'sources': []})}\n\n"
            )
            return

        # Step 4.5: Quality Filter
        with tracer.stage("quality_filter"):
            quality_result = self.quality_filter.filter(passages)
            filtered_passages = quality_result["filtered_passages"]

            tracer.record_retrieval_quality(
                candidates=candidates,
                reranked_passages=passages,
                retained_passages=filtered_passages,
            )
            passages = filtered_passages

        if not passages:
            tracer.finish(
                status="insufficient_evidence",
                reason="All candidate passages filtered out by quality criteria.",
            )
            yield (
                "event: insufficient_evidence\n"
                f"data: {json.dumps({'status': 'insufficient_evidence', 'answer': 'All candidate passages were filtered out due to quality, authority, or freshness criteria.', 'question': question, 'sources': []})}\n\n"
            )
            return

        # Step 5: Passage PII Sanitizing
        # Note: do not uncomment as we don't want for our application
        # with tracer.stage("passage_pii_sanitizing"):
        #     try:
        #         passages = sanitize_passages(passages)
        #     except Exception as exc:
        #         tracer.record_error("passage_pii_sanitizing", exc, recoverable=False)
        #         tracer.finish(status="error", reason=str(exc))
        #         yield (
        #             "event: error\n"
        #             f"data: {json.dumps({'status': 'error', 'detail': 'Privacy checks on retrieved evidence could not be completed.', 'question': question})}\n\n"
        #         )
        #         return

        # Step 5.5: Context Budgeting
        with tracer.stage("context_budgeting"):
            model_limit = getattr(self.generator, "max_context_tokens", 131072) or 131072
            output_res = getattr(self.generator, "max_output_tokens", 3000) or 3000
            safety_margin = 500
            system_tokens = self.context_manager.count_tokens(SYSTEM_PROMPT)
            question_tokens = self.context_manager.count_tokens(masked_question)

            dynamic_evidence_budget = self.context_manager.calculate_dynamic_budget(
                model_context_limit=model_limit,
                system_tokens=system_tokens,
                question_tokens=question_tokens,
                output_reservation=output_res,
                safety_margin=safety_margin,
                history_tokens=history_tokens,
            )

            try:
                context_result = self.context_manager.build_context(
                    question=masked_question,
                    passages=passages,
                    max_evidence_tokens=dynamic_evidence_budget,
                )
                passages = context_result["selected_passages"]
            except Exception as exc:
                tracer.record_error("context_budgeting", exc, recoverable=False)
                tracer.finish(status="error", reason=str(exc))
                yield (
                    "event: error\n"
                    f"data: {json.dumps({'status': 'error', 'detail': 'Context-window management failed.', 'question': question})}\n\n"
                )
                return

            metrics = ContextMetrics(
                model=str(getattr(self.generator, "model", "Unknown")),
                model_context_limit=model_limit,
                system_tokens=system_tokens,
                question_tokens=question_tokens,
                history_tokens=history_tokens,
                output_reservation=output_res,
                safety_margin=safety_margin,
                allocated_evidence_budget=dynamic_evidence_budget,
                used_evidence_tokens=context_result["total_tokens"],
                raw_evidence_tokens=context_result.get("raw_tokens", context_result["total_tokens"]),
                input_passages_count=context_result["input_passage_count"],
                selected_passages_count=context_result["selected_passage_count"],
                excluded_duplicates=context_result["excluded_duplicates"],
                excluded_budget=context_result["excluded_budget"],
                excluded_diversity=context_result.get("excluded_diversity", 0),
                excluded_redundancy=context_result.get("excluded_redundancy", 0),
                excluded_low_rerank=quality_result.get("excluded_low_rerank", 0),
                excluded_low_authority=quality_result.get("excluded_low_authority", 0),
                excluded_outdated=quality_result.get("excluded_outdated", 0),
                clinical_caveats_protected=context_result.get("clinical_caveats_protected", 0),
                compression_applied=context_result.get("compression_applied", False),
                compression_ratio=context_result.get("compression_ratio", 1.0),
            )
            metrics.compute_ratios()
            tracer.record_tokens(metrics.to_dict())

        # Step 6: Evidence & Answerability Check
        with tracer.stage("evidence_check"):
            evidence_result = self.evidence_checker.check(passages)
        if not evidence_result["sufficient"]:
            tracer.finish(status="insufficient_evidence", reason=evidence_result.get("reason"))
            yield (
                "event: insufficient_evidence\n"
                f"data: {json.dumps({'status': 'insufficient_evidence', 'answer': 'The available passages did not pass preliminary checks.', 'reason': evidence_result.get('reason'), 'sources': passages, 'context_metrics': metrics.to_dict()})}\n\n"
            )
            return

        with tracer.stage("answerability_check"):
            answerability_result = self.answerability_checker.check(
                question=masked_question,
                passages=passages,
            )
        if not answerability_result["answerable"]:
            tracer.finish(status="insufficient_evidence", reason=answerability_result.get("reason"))
            yield (
                "event: insufficient_evidence\n"
                f"data: {json.dumps({'status': 'insufficient_evidence', 'answer': 'I cannot provide a grounded answer from the available documents.', 'reason': answerability_result.get('reason'), 'sources': passages, 'context_metrics': metrics.to_dict()})}\n\n"
            )
            return

        # Answer cache check
        cache_allowed = (
            answer_cache_enabled()
            and not pii_result["pii_detected"]
            and not valid_history
        )
        model_version = str(getattr(self.generator, "model", ""))
        prompt_version = os.getenv("ANSWER_PROMPT_VERSION", "")
        retrieval_version = os.getenv("ANSWER_RETRIEVAL_VERSION", "")
        corpus_version = get_corpus_version()
        access_scope = os.getenv("ANSWER_CACHE_ACCESS_SCOPE", "local")
        evidence_fingerprint = make_evidence_fingerprint(passages)

        with tracer.stage("answer_cache"):
            if cache_allowed:
                cached_result = get_cached_answer(
                    question=masked_question,
                    model_version=model_version,
                    prompt_version=prompt_version,
                    corpus_version=corpus_version,
                    retrieval_version=retrieval_version,
                    evidence_fingerprint=evidence_fingerprint,
                    access_scope=access_scope,
                    allow_cache=cache_allowed,
                )
            else:
                cached_result = None

        if cached_result is not None:
            tracer.record_cache_access("answer", "hit", key_hash=evidence_fingerprint, latency_saved_ms=2500.0)
            yield (
                "event: metadata\n"
                f"data: {json.dumps({'status': 'cache_hit', 'sources': cached_result.get('sources', passages), 'model': cached_result.get('model', model_version)})}\n\n"
            )
            cached_text = cached_result.get("answer", "")
            words = cached_text.split(" ")
            for i, w in enumerate(words):
                sep = " " if i < len(words) - 1 else ""
                yield (
                    "event: token\n"
                    f"data: {json.dumps({'token': w + sep})}\n\n"
                )
            obs_report = tracer.finish(status="success")
            cached_result["observability"] = obs_report
            yield (
                "event: done\n"
                f"data: {json.dumps(cached_result)}\n\n"
            )
            return

        # Step 6.8: Model Routing
        with tracer.stage("model_routing"):
            route_decision = self.model_router.route(
                question=masked_question,
                passages=passages,
                history_turns=len(valid_history),
            )
            tracer.record_model_route(route_decision)

        # Step 7: Answer Generation (Streaming)
        # First send metadata event to client
        yield (
            "event: metadata\n"
            f"data: {json.dumps({'status': 'generating', 'model': route_decision.selected_model, 'sources': passages, 'context_metrics': metrics.to_dict()})}\n\n"
        )

        accumulated_chunks: list[str] = []
        gen_sources: list[dict] = passages

        with tracer.stage("answer_generation"):
            try:
                if hasattr(self.generator, "stream_generate"):
                    for gen_event in self.generator.stream_generate(
                        question=masked_question,
                        passages=passages,
                        conversation_history=valid_history if valid_history else None,
                        model=route_decision.selected_model,
                    ):
                        if gen_event.get("type") == "token":
                            chunk = gen_event.get("content", "")
                            accumulated_chunks.append(chunk)
                            yield (
                                "event: token\n"
                                f"data: {json.dumps({'token': chunk})}\n\n"
                            )
                        elif gen_event.get("type") == "metadata":
                            gen_sources = gen_event.get("sources", passages)
                else:
                    # Fallback for standard generate()
                    gen_res = self.generator.generate(
                        question=masked_question,
                        passages=passages,
                        conversation_history=valid_history if valid_history else None,
                        model=route_decision.selected_model,
                    )
                    ans_text = gen_res.get("answer", "")
                    accumulated_chunks.append(ans_text)
                    yield (
                        "event: token\n"
                        f"data: {json.dumps({'token': ans_text})}\n\n"
                    )
                    gen_sources = gen_res.get("sources", passages)
            except Exception as exc:
                tracer.record_error("answer_generation", exc, recoverable=False)
                tracer.finish(status="error", reason=str(exc))
                yield (
                    "event: error\n"
                    f"data: {json.dumps({'status': 'error', 'detail': str(exc)})}\n\n"
                )
                return

        raw_answer = "".join(accumulated_chunks)
        tracer.record_generation(
            name="answer_generation_stream",
            model=route_decision.selected_model,
            input_prompt=masked_question,
            output_text=raw_answer,
        )

        # Step 8: Post-generation Guardrails
        # Note: Do not uncomment as we don't want for our application
        # with tracer.stage("output_pii_scan"):
        #     try:
        #         answer_pii = mask_output_pii(raw_answer)
        #         clean_answer = answer_pii["text"]
        #     except Exception as exc:
        #         tracer.record_error("output_pii_scan", exc, recoverable=False)
        #         tracer.finish(status="withheld", reason="Output privacy check failed.")
        #         yield (
        #             "event: withheld\n"
        #             f"data: {json.dumps({'status': 'withheld', 'detail': 'Output privacy check failed.'})}\n\n"
        #         )
        #         return
        clean_answer = raw_answer

        with tracer.stage("nemo_output_guard"):
            output_result = self.nemo_guardrails.check_output(
                question=masked_question,
                answer=clean_answer,
            )

        if not output_result["allowed"]:
            tracer.finish(status="withheld", reason=output_result.get("reason"))
            yield (
                "event: withheld\n"
                f"data: {json.dumps({'status': 'withheld', 'reason': output_result.get('reason'), 'sources': gen_sources})}\n\n"
            )
            return

        from app.generation.answer_generator import check_citations
        citation_check = check_citations(clean_answer, gen_sources)
        tracer.record_score(
            name="citation_validity",
            value=1.0 if citation_check.get("valid", False) else 0.0,
            comment=citation_check.get("reason", "Citation validation check"),
        )
        if not citation_check.get("valid", False):
            tracer.finish(status="withheld", reason="Citation validation failed.")
            yield (
                "event: withheld\n"
                f"data: {json.dumps({'status': 'withheld', 'reason': citation_check.get('reason', 'Citation validation failed.'), 'citation_check': citation_check, 'sources': gen_sources})}\n\n"
            )
            return

        with tracer.stage("claim_verification"):
            verification = self.verifier.verify(
                answer=clean_answer,
                sources=gen_sources,
            )

        tracer.record_score(
            name="claim_faithfulness",
            value=1.0 if verification.get("passed", False) else 0.0,
            comment=verification.get("reason", "Claim verification check"),
        )
        tracer.record_score(
            name="retrieval_quality",
            value=tracer.retrieval_quality.retrieval_quality_score,
            comment="Composite retrieval quality score",
        )

        if not verification.get("passed", False):
            tracer.finish(status="withheld", reason="Claim verification failed.")
            yield (
                "event: withheld\n"
                f"data: {json.dumps({'status': 'withheld', 'reason': 'Claim verification failed.', 'claim_verification': verification, 'sources': gen_sources})}\n\n"
            )
            return

        # Cache valid response
        if cache_allowed:
            set_cached_answer(
                question=masked_question,
                result={
                    "answer": clean_answer,
                    "sources": gen_sources,
                    "citation_check": citation_check,
                    "model": route_decision.selected_model,
                },
                model_version=model_version,
                prompt_version=prompt_version,
                corpus_version=corpus_version,
                retrieval_version=retrieval_version,
                evidence_fingerprint=evidence_fingerprint,
                access_scope=access_scope,
                verified=True,
                allow_cache=cache_allowed,
            )

        obs_report = tracer.finish(status="success")

        # Step 9: Yield final done event
        final_payload = {
            "status": "success",
            "answer": clean_answer,
            "question": question,
            "sources": gen_sources,
            "citation_check": citation_check,
            "claim_verification": verification,
            "model": route_decision.selected_model,
            "model_route": tracer.model_route,
            "context_metrics": metrics.to_dict(),
            "observability": obs_report,
        }
        yield (
            "event: done\n"
            f"data: {json.dumps(final_payload)}\n\n"
        )


def main():
    question = input("\nEnter your clinical question: ").strip()

    if not question:
        print("Please enter a question")
        return

    pipeline = AnswerPipeline()
    result = pipeline.run(question, verbose=True)

    if result.get("status") == "success":
        print("\n" + "=" * 72)
        print("CLINICAL KNOWLEDGE ASSISTANT")
        print("=" * 72)

        print("\nQUESTION")
        print(result["question"])

        print("\nANSWER")
        print(result["answer"])

        print("\nSOURCES")
        for source in result["sources"]:
            print(
                f"[{source['source_id']}] "
                f"{source.get('filename', 'Unknown file')} "
                f"- Page {source.get('page', 'Unknown')}"
            )

        print("\nCitation validation:", result.get("citation_check", {}).get("reason"))
        print("Claim verification:", result.get("claim_verification", {}).get("reason"))
        print("Model:", result.get("model", "Unknown"))

        if result.get("context_metrics"):
            cm = result["context_metrics"]
            print(f"Context footprint: {cm.get('total_request_tokens'):,} tokens ({cm.get('context_utilization_ratio', 0)*100:.2f}% of {cm.get('model_context_limit', 0):,} limit)")
            print(f"Evidence tokens: {cm.get('used_evidence_tokens'):,} / {cm.get('allocated_evidence_budget'):,} allocated")


if __name__ == "__main__":
    main()
