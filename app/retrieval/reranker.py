from sentence_transformers import CrossEncoder

RERANKER_MODEL="cross-encoder/ms-marco-MiniLM-L-6-v2"

class ClinicalReranker:
    """Rerank retrieved passages by their relevance to a question."""

    def __init__(self, model_name:str=RERANKER_MODEL):
        self.model=CrossEncoder(model_name)

    def rerank(self, question:str, candidates:list[dict], top_k:int=5)->list[dict]:
        """
        Rerank candidate chunks.

        Preserves the original RRF score as `rrf_score` and adds the cross-encoder output as `rerank_score`.
        """
        if not candidates or top_k<=0:
            return []
        pairs=[
            (question,candidate["text"])
            for candidate in candidates
        ]
        scores=self.model.predict(pairs)
        reranked=[]

        for candidate,score in zip(candidates,scores):
            result=candidate.copy()
            result["rrf_score"]=result.get("score")
            result["rerank_score"]=float(score)
            reranked.append(result)

        reranked.sort(
            key=lambda item:item["rerank_score"],
            reverse=True
        )

        results=reranked[:top_k]

        for rank,result in enumerate(results,start=1):
            result["rank"]=rank

        return results