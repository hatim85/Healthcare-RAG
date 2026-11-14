class EvidenceChecker:
    """Apply basic checks before generating an answer."""

    def __init__(
        self,
        min_passages:int=1,
        min_text_chars:int=80,
        min_rerank_score:float=None
    ):
        self.min_passages=min_passages
        self.min_text_chars=min_text_chars
        self.min_rerank_score=min_rerank_score

    def check(self,passages:list[dict])->dict:
        if len(passages) < self.min_passages:
            return {
                "sufficient":False,
                "reason":"Too few passages were retrieved."
            }
        usable_passages=[
            passage
            for passage in passages
            if len(passage.get("text","").strip())
            >=self.min_text_chars
        ]
        if not usable_passages:
            return {
                "sufficient":False,
                "reason":"No passages contain enough text."
            }
        if self.min_rerank_score is not None:
            best_score=max(
                passage.get("rerank_score",float("-inf"))
                for passage in usable_passages
            )
            if best_score < self.min_rerank_score:
                return {
                    "sufficient":False,
                    "reason":(
                        "The best reranker score did not meet the configured preliminary threshold."
                    )
                }
        return {
            "sufficient":True,
            "reason":(
                "Basic passage checks passed. This does not guarantee that the evidence answers the question."
            )
        }