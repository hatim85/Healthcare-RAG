from sentence_transformers import SentenceTransformer
from app.ingestion.indexer import (COLLECTION_NAME, EMBEDDING_MODEL)
from app.retrieval.qdrant_client import get_qdrant_client
from app.caching.embedding_cache import get_query_embedding

class DenseRetriever:
    """Retrieve relevant document chunks from Qdrant Cloud."""

    def __init__(self):
        self.client=get_qdrant_client()
        self.embedding_model=SentenceTransformer(EMBEDDING_MODEL)

    def search(self, question:str, top_k:int=5)->list[dict]:
        """Return the most relevant chunks for a question."""
        if not question.strip():
            raise ValueError("Question cannot be empty.")
        if top_k<1:
            raise ValueError("top_k must be at least 1.")
        query_vector=get_query_embedding(
            question=question,
            normalize_embeddings=self.embedding_model,
            model_name=EMBEDDING_MODEL
        )
        response=self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector.tolist(),
            using="dense",
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )
        results=[]
        for point in response.points:
            payload=point.payload or {}
            results.append(
                {
                    "chunk_id":str(point.id),
                    "score":float(point.score),
                    "text":payload.get("text",""),
                    "source":payload.get("source",""),
                    "filename":payload.get("filename",""),
                    "title":payload.get("title",""),
                    "page":payload.get("page"),
                    "chunk_index":payload.get("chunk_index")
                }
            )
        return results

    def close(self):
        """Close the Qdrant client."""
        self.client.close()

if __name__=="__main__":
    retriever=DenseRetriever()
    try:
        question="What is the recommended dosage of medication X?"
        results=retriever.search(question=question,top_k=3)
        print(f"\nQuestion: {question}")
        print(f"Retrieved {len(results)} results\n")

        for rank, result in enumerate(results,start=1):
            print(f"---Result {rank}---")
            print(f"Score: {result['score']:.4f}")
            print(f"Source: {result['filename']}")
            print(f"Page: {result['page']}")
            print(f"Text: {result['text']}")
            print()
    finally:
        retriever.close()