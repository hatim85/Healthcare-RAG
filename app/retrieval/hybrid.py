from sentence_transformers import SentenceTransformer
from qdrant_client import models
from app.ingestion.indexer import (COLLECTION_NAME,EMBEDDING_MODEL)
from app.retrieval.qdrant_client import get_qdrant_client
from app.caching.embedding_cache import get_query_embedding
from app.caching.retrieval_cache import (get_cached_retrieval,set_cached_retrieval)

class HybridRetriever:
    """Combine dense semantic search and BM25 full-text search using Qdrant's Reciprocal Rank Fusion."""

    def __init__(self):
        self.client=get_qdrant_client()
        self.embedding_model=SentenceTransformer(EMBEDDING_MODEL)

    def search(
        self,
        question:str, 
        top_k:int=5, 
        candidate_k:int=20,
        allow_cache:bool=True,
        tenant_id:str | None = None,
    )->list[dict]:
        if not isinstance(question,str) or not question.strip():
            raise ValueError("Question cannot be empty.")
        if top_k<1:
            raise ValueError("top_k must be at least 1.")
        if candidate_k<top_k:
            candidate_k = max(20, top_k * 2)

        config_version=f"hybrid-rrf-v1:{EMBEDDING_MODEL}:{tenant_id or 'all'}"
        
        cached_results=get_cached_retrieval(
            question=question,
            top_k=top_k,
            candidate_k=candidate_k,
            config_version=config_version,
            allow_cache=allow_cache
        )
        if cached_results is not None:
            return cached_results
        
        dense_vector=get_query_embedding(
            question=question,
            embedding_model=self.embedding_model,
            model_name=EMBEDDING_MODEL,
            allow_cache=allow_cache
        )
        
        prefetch_filter = None
        if tenant_id and tenant_id != "*":
            prefetch_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="tenant_id",
                        match=models.MatchValue(value=tenant_id),
                    )
                ]
            )

        response=self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=dense_vector.tolist(),
                    using="dense",
                    limit=candidate_k,
                    filter=prefetch_filter,
                ),
                models.Prefetch(
                    query=models.TextQuery(text=question),
                    using="text",
                    limit=candidate_k,
                    filter=prefetch_filter,
                )
            ],
            query=models.FusionQuery(
                fusion=models.Fusion.RRF
            ),
            query_filter=prefetch_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )
        results=[]

        for rank, point in enumerate(response.points,start=1):
            payload=point.payload or {}
            results.append(
                {
                    "rank":rank,
                    "chunk_id":str(point.id),
                    "score":float(point.score),
                    "text":payload.get("text",""),
                    "source":payload.get("source",""),
                    "filename":payload.get("filename",""),
                    "title":payload.get("title",""),
                    "page":payload.get("page"),
                    "chunk_index":payload.get("chunk_index"),
                    "tenant_id":payload.get("tenant_id", "default"),
                }
            )
            
        set_cached_retrieval(
            question=question,
            top_k=top_k,
            candidate_k=candidate_k,
            results=results,
            config_version=config_version,
            allow_cache=allow_cache
        )
        
        return results

    def close(self):
        self.client.close()

if __name__=="__main__":
    test_questions=[
        "What information should clinical documents preserve?",
        "What is the recommended dosage of medication X?"
    ]
    retriever=HybridRetriever()
    try:
        for question in test_questions:
            print(f"\nQuestion: {question}")
            results=retriever.search(
                question=question,
                top_k=3,
                candidate_k=10
            )
            if not results:
                print("no matching passages found.")
                continue
            for result in results:
                print(f"\nRank: {result['rank']}")
                print(f"RRF Score: {result['score']:.4f}")
                print(f"Source: {result['filename']}")
                print(f"Page: {result['page']}")
                print(f"Text: {result['text']}")
    finally:
        retriever.close()