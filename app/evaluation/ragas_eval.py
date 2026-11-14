"""Automated RAGAS-style Clinical Evaluation Suite.

Evaluates the Clinical RAG system across 50+ benchmark cases computing:
- Faithfulness / Groundedness (Claim-level entailment against retrieved sources)
- Answer Relevance (Query-response semantic coherence & coverage)
- Context Precision & Recall (Retrieval hit rate against clinical guidelines)
- Security Block Rate (Adversarial prompt injection defense)
- Abstention Accuracy (Clean refusal on out-of-domain medical queries)
"""

import json
import logging
from pathlib import Path
import time
from typing import Any

from app.generation.answer_generator import ClinicalAnswerGenerator
from app.guardrails.answerability_checker import AnswerabilityChecker
from app.guardrails.claim_verifier import ClaimVerifier
from app.guardrails.evidence_checker import EvidenceChecker
from app.guardrails.nemo_input_guard import NemoGuardrails
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import ClinicalReranker

logger = logging.getLogger("ragas_eval")

ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "data" / "evaluation" / "clinical_rag_eval.jsonl"
BENCHMARK_RESULTS_PATH = ROOT / "data" / "evaluation" / "benchmark_results.json"


def compute_answer_relevance(question: str, answer: str, expected_keywords: list[str]) -> float:
    """Compute Answer Relevance score (0.0 - 1.0) based on keyword recall and response structure."""
    if not answer or not answer.strip():
        return 0.0

    answer_lower = answer.lower()
    
    # 1. Keyword coverage
    if expected_keywords:
        matched = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
        keyword_score = matched / len(expected_keywords)
    else:
        keyword_score = 1.0

    # 2. Length and structure appropriateness (penalize overly terse or huge answers)
    length = len(answer.split())
    if 15 <= length <= 300:
        length_penalty = 1.0
    elif length < 15:
        length_penalty = length / 15.0
    else:
        length_penalty = max(0.6, 1.0 - (length - 300) / 500.0)

    # 3. Citation presence
    citation_bonus = 1.0 if ("[S" in answer or "[s" in answer) else 0.7

    return round(0.5 * keyword_score + 0.3 * length_penalty + 0.2 * citation_bonus, 3)


def run_benchmark(
    sample_size: int | None = None,
    stratified_per_domain: int | None = None,
    delay_between_cases: float = 2.0,
) -> dict[str, Any]:
    """Execute automated benchmark evaluation over the clinical test dataset."""
    print("=" * 70)
    print("🏥 Starting Automated Clinical RAG Benchmark Evaluation")
    print("=" * 70)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    if stratified_per_domain:
        from collections import defaultdict
        domain_buckets = defaultdict(list)
        for c in cases:
            domain_buckets[c.get("domain", "clinical")].append(c)
        sampled = []
        for dom, dom_cases in domain_buckets.items():
            sampled.extend(dom_cases[:stratified_per_domain])
        cases = sampled
    elif sample_size and sample_size < len(cases):
        cases = cases[:sample_size]

    print(f"Loaded {len(cases)} benchmark test cases from {DATASET_PATH.name}")

    # Initialize RAG Pipeline components
    guardrails = NemoGuardrails()
    retriever = HybridRetriever()
    reranker = ClinicalReranker()
    evidence_checker = EvidenceChecker()
    answerability_checker = AnswerabilityChecker()
    answer_generator = ClinicalAnswerGenerator()
    claim_verifier = ClaimVerifier()

    case_results = []
    
    # Metrics accumulators
    security_eval = {"passed": 0, "total": 0}
    abstention_eval = {"passed": 0, "total": 0}
    retrieval_hits = {"passed": 0, "total": 0}
    faithfulness_scores: list[float] = []
    relevance_scores: list[float] = []
    grounded_answers = {"passed": 0, "total": 0}

    for idx, case in enumerate(cases, 1):
        case_id = case["id"]
        domain = case.get("domain", "clinical")
        question = case["question"]
        expected_block = case.get("expected_input_block", False)
        expected_answerable = case.get("expected_answerable", True)
        expected_keywords = case.get("expected_keywords", [])

        print(f"\n[{idx}/{len(cases)}] Case: {case_id} (Domain: {domain})")
        print(f"  Q: {question[:80]}...")

        record = {
            "id": case_id,
            "domain": domain,
            "question": question,
            "expected_block": expected_block,
            "expected_answerable": expected_answerable,
            "input_blocked": False,
            "answerable": False,
            "retrieval_hit": None,
            "faithfulness": None,
            "answer_relevance": None,
            "citation_valid": None,
            "status": "PASS",
        }

        # Step 1: Input Guardrail Check
        input_check = guardrails.check_input(question)
        record["input_blocked"] = not input_check["allowed"]

        if expected_block:
            security_eval["total"] += 1
            if record["input_blocked"]:
                security_eval["passed"] += 1
                record["status"] = "PASS"
                print(f"  🛡️ Security Guardrail: BLOCKED (Correct)")
            else:
                record["status"] = "FAIL"
                print(f"  ❌ Security Guardrail: ALLOWED (False Negative)")
            case_results.append(record)
            time.sleep(delay_between_cases)
            continue

        if record["input_blocked"]:
            record["status"] = "FAIL (Unexpected Block)"
            print(f"  ❌ Security Guardrail: UNEXPECTED BLOCK")
            case_results.append(record)
            time.sleep(delay_between_cases)
            continue

        # Step 2: Hybrid Retrieval & Reranking
        candidates = retriever.search(question, top_k=20)
        passages = reranker.rerank(question=question, candidates=candidates, top_k=5) if candidates else []

        expected_pages = set(case.get("expected_relevant_pages", []))
        if expected_pages:
            retrieval_hits["total"] += 1
            retrieved_pages = {p.get("page") for p in passages}
            hit = bool(expected_pages & retrieved_pages)
            record["retrieval_hit"] = hit
            if hit:
                retrieval_hits["passed"] += 1
            print(f"  🔍 Retrieval: {'HIT' if hit else 'MISS'} (Pages: {retrieved_pages})")

        # Step 3: Evidence & Answerability Check
        ev_check = evidence_checker.check(passages)
        if not ev_check["sufficient"]:
            is_answerable = False
        else:
            ans_check = answerability_checker.check(question=question, passages=passages)
            is_answerable = ans_check["answerable"]

        record["answerable"] = is_answerable

        # Handle Abstention cases
        if not expected_answerable:
            abstention_eval["total"] += 1
            if not is_answerable:
                abstention_eval["passed"] += 1
                record["status"] = "PASS"
                print(f"  🛑 Abstention: CORRECT (Refused unsupported question)")
            else:
                record["status"] = "FAIL (Hallucination risk: marked answerable)"
                print(f"  ⚠️ Abstention: FAILED (System attempted to answer unsupported topic)")
            case_results.append(record)
            time.sleep(delay_between_cases)
            continue

        # Step 4: Answer Generation & RAGAS Scoring for Answerable Cases
        if is_answerable:
            try:
                gen_resp = answer_generator.generate(question=question, passages=passages)
                answer = gen_resp.get("answer", "")
                citation_info = gen_resp.get("citation_check", {})
                record["citation_valid"] = citation_info.get("valid", False)

                # RAGAS Metric: Faithfulness (Claim-level entailment)
                if record["citation_valid"]:
                    claim_res = claim_verifier.verify(answer=answer, sources=gen_resp.get("sources", []))
                    faithfulness = 1.0 if claim_res.get("passed", False) else 0.5
                else:
                    faithfulness = 0.0
                
                record["faithfulness"] = faithfulness
                faithfulness_scores.append(faithfulness)

                # RAGAS Metric: Answer Relevance
                relevance = compute_answer_relevance(question, answer, expected_keywords)
                record["answer_relevance"] = relevance
                relevance_scores.append(relevance)

                grounded_pass = bool(record["citation_valid"] and faithfulness >= 0.8)
                grounded_answers["total"] += 1
                if grounded_pass:
                    grounded_answers["passed"] += 1

                record["status"] = "PASS" if grounded_pass else "PARTIAL"
                print(f"  ✅ Grounded Answer: Faithfulness={faithfulness}, Relevance={relevance}, Citations={'VALID' if record['citation_valid'] else 'INVALID'}")
            except Exception as exc:
                logger.warning(f"Generation error on case {case_id}: {exc}")
                record["status"] = f"ERROR ({type(exc).__name__})"
                record["error"] = str(exc)
                print(f"  ⚠️ Generation error: {exc}")
        else:
            record["status"] = "FAIL (False Abstention)"
            print(f"  ❌ False Abstention: Expected answerable but marked unanswerable")

        case_results.append(record)
        time.sleep(delay_between_cases)

    # Compute overall benchmark statistics
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_cases_evaluated": len(cases),
        "security_block_accuracy": {
            "passed": security_eval["passed"],
            "total": security_eval["total"],
            "score": round(security_eval["passed"] / max(1, security_eval["total"]), 3),
        },
        "abstention_accuracy": {
            "passed": abstention_eval["passed"],
            "total": abstention_eval["total"],
            "score": round(abstention_eval["passed"] / max(1, abstention_eval["total"]), 3),
        },
        "retrieval_hit_rate": {
            "passed": retrieval_hits["passed"],
            "total": retrieval_hits["total"],
            "score": round(retrieval_hits["passed"] / max(1, retrieval_hits["total"]), 3) if retrieval_hits["total"] else 1.0,
        },
        "average_faithfulness": round(sum(faithfulness_scores) / max(1, len(faithfulness_scores)), 3) if faithfulness_scores else 0.0,
        "average_answer_relevance": round(sum(relevance_scores) / max(1, len(relevance_scores)), 3) if relevance_scores else 0.0,
        "grounded_answer_pass_rate": {
            "passed": grounded_answers["passed"],
            "total": grounded_answers["total"],
            "score": round(grounded_answers["passed"] / max(1, grounded_answers["total"]), 3) if grounded_answers["total"] else 0.0,
        },
    }

    report_payload = {
        "summary": summary,
        "cases": case_results,
    }

    BENCHMARK_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BENCHMARK_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    print("\n" + "=" * 70)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 70)
    print(json.dumps(summary, indent=2))
    print(f"\nDetailed results saved to {BENCHMARK_RESULTS_PATH}")

    return report_payload


if __name__ == "__main__":
    run_benchmark()

