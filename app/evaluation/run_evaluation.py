import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "data" / "evaluation" / "clinical_rag_eval.jsonl"
RESULTS_PATH = ROOT / "data" / "evaluation" / "results.json"
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import ClinicalReranker
from app.guardrails.evidence_checker import EvidenceChecker
from app.guardrails.answerability_checker import AnswerabilityChecker
from app.guardrails.nemo_input_guard import NemoGuardrails
from app.guardrails.claim_verifier import ClaimVerifier
from app.generation.answer_generator import ClinicalAnswerGenerator




def load_cases():
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main():
    cases = load_cases()

    # Initialize reusable components once per evaluation run.
    guardrails = NemoGuardrails()
    retriever = HybridRetriever()
    reranker = ClinicalReranker()
    evidence_checker = EvidenceChecker(
        min_passages=1,
        min_text_chars=80,
        min_rerank_score=None,
    )
    answerability_checker = AnswerabilityChecker()
    answer_generator = ClinicalAnswerGenerator()
    claim_verifier = ClaimVerifier()

    results = []

    for case in cases:
        case_id = case["id"]
        question = case["question"]

        print(f"\n{'=' * 65}")
        print(f"Test: {case_id}")
        print("Question:", question)

        record = {
            "id": case_id,
            "expected_input_block": case["expected_input_block"],
            "expected_answerable": case["expected_answerable"],
            "input_blocked": None,
            "input_check_error": None,
            "input_block_correct": None,
            "answerable_prediction": None,
            "answerability_correct": None,
            "retrieval_hit": None,
            "citation_valid": None,
            "output_safety_passed": None,
            "claim_verification_passed": None,
            "keyword_coverage_passed": None,
            "grounded_answer_passed": None,
        }

        # 1. Input safety.
        input_result = guardrails.check_input(question)

        input_blocked = not input_result["allowed"]
        check_error = input_result.get("check_error", False)

        record["input_blocked"] = input_blocked
        record["input_check_error"] = check_error

        # A failed guardrail check must not count as a successful
        # security decision in the evaluation.
        record["input_block_correct"] = (
            None
            if check_error
            else input_blocked == case["expected_input_block"]
        )

        print("Input safety:", input_result["reason"])
        
        if check_error:
            record["answerable_prediction"] = None
            record["answerability_correct"] = None
            record["grounded_answer_passed"] = (
                False if case["expected_answerable"] else None
            )
            results.append(record)

            print("Input guardrail ERROR: test outcome is inconclusive.")
            continue

        # If the test expects a block, do not continue processing that input.
        if case["expected_input_block"]:
            record["answerable_prediction"] = None
            results.append(record)
            print(
                "Expected input block:",
                "PASS" if input_blocked else "FAIL",
            )
            continue

        # Unexpected input blocks count as failure for this test.
        if input_blocked:
            record["answerable_prediction"] = None
            record["answerability_correct"] = None
            record["grounded_answer_passed"] = (
                False if case["expected_answerable"] else None
            )
            results.append(record)
            print("Unexpected input block.")
            continue

        # 2. Retrieval.
        candidates = retriever.search(question, top_k=20)

        passages = (
            reranker.rerank(
                question=question,
                candidates=candidates,
                top_k=5,
            )
            if candidates
            else []
        )

        expected_pages = set(case["expected_relevant_pages"])

        if expected_pages:
            retrieved_pages = {passage.get("page") for passage in passages}
            record["retrieval_hit"] = bool(expected_pages & retrieved_pages)

            print("Relevant page retrieved:", record["retrieval_hit"])

        # 3. Evidence sufficiency and answerability.
        evidence_result = evidence_checker.check(passages)

        if not evidence_result["sufficient"]:
            answerable_prediction = False
        else:
            answerability_result = answerability_checker.check(
                question=question,
                passages=passages,
            )
            answerable_prediction = answerability_result["answerable"]

            print(
                "Answerability reason:",
                answerability_result["reason"],
            )

        record["answerable_prediction"] = answerable_prediction
        record["answerability_correct"] = (
            answerable_prediction == case["expected_answerable"]
        )

        print(
            "Answerability:",
            "PASS" if record["answerability_correct"] else "FAIL",
        )

        # 4. Only expected-answerable cases proceed to answer verification.
        if case["expected_answerable"] and answerable_prediction:
            generated = answer_generator.generate(
                question=question,
                passages=passages,
            )
            
            print("Generated answer:", repr(generated["answer"]))
            print("Citation check:", generated.get("citation_check"))

            citation_check = generated.get("citation_check", {})
            record["citation_valid"] = citation_check.get("valid", False)

            output_result = guardrails.check_output(
                question=question,
                answer=generated["answer"],
            )
            record["output_safety_passed"] = output_result["allowed"]

            # Run claim verification only if earlier output checks pass.
            if record["citation_valid"] and record["output_safety_passed"]:
                verification = claim_verifier.verify(
                    answer=generated["answer"],
                    sources=generated["sources"],
                )
                record["claim_verification_passed"] = verification["passed"]

            answer_lower = generated["answer"].lower()
            expected_keywords = case["expected_keywords"]

            record["keyword_coverage_passed"] = all(
                keyword.lower() in answer_lower for keyword in expected_keywords
            )

            record["grounded_answer_passed"] = all(
                [
                    record["citation_valid"],
                    record["output_safety_passed"],
                    record["claim_verification_passed"] is True,
                    record["keyword_coverage_passed"],
                ]
            )

            print("Citation validation:", record["citation_valid"])
            print("Output safety:", record["output_safety_passed"])
            print(
                "Claim verification:",
                record["claim_verification_passed"],
            )
            print("Expected keyword coverage:", record["keyword_coverage_passed"])
            print("Grounded answer:", record["grounded_answer_passed"])

        elif case["expected_answerable"]:
            # An expected-answerable question was incorrectly rejected.
            record["grounded_answer_passed"] = False

        results.append(record)
        import time
        time.sleep(3)

    # 5. Aggregate evaluation metrics.
    def metric_summary(values):
        """Summarize only applicable, non-None evaluation results."""
        evaluated = [value for value in values if value is not None]
        passed = sum(value is True for value in evaluated)
        skipped = len(values) - len(evaluated)

        return {
            "passed": passed,
            "evaluated": len(evaluated),
            "skipped": skipped,
            "accuracy": (round(passed / len(evaluated), 3) if evaluated else None),
        }

    input_block_metrics = metric_summary(
        [row["input_block_correct"] for row in results]
    )

    answerability_metrics = metric_summary(
        [
            row["answerability_correct"]
            for row in results
            if row["expected_input_block"] is False
        ]
    )

    retrieval_metrics = metric_summary([row["retrieval_hit"] for row in results])

    grounded_answer_metrics = metric_summary(
        [
            row["grounded_answer_passed"]
            for row in results
            if row["expected_answerable"] is True
        ]
    )

    report = {
        "dataset": str(DATASET_PATH.relative_to(ROOT)),
        "num_cases": len(results),
        "metrics": {
            "input_block_accuracy": input_block_metrics,
            "answerability_accuracy": answerability_metrics,
            "retrieval_hit_rate": retrieval_metrics,
            "grounded_answer_pass_rate": grounded_answer_metrics,
        },
        "cases": results,
    }

    RESULTS_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\n{'=' * 65}")
    print("EVALUATION SUMMARY")
    print(json.dumps(report["metrics"], indent=2))
    print("\nDetailed results saved to:", RESULTS_PATH)


if __name__ == "__main__":
    main()
