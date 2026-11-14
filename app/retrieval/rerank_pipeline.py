from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import ClinicalReranker

def main():
    question=input("Enter your clinical question: ").strip()
    if not question:
        print("Please enter a question.")
        return

    retriever=HybridRetriever()
    candidates=retriever.search(question,top_k=20)

    if not candidates:
        print("No matching passages were retrieved.")
        return

    print(f"\nHybrid retrieval returned {len(candidates)} candidates.")

    reranker=ClinicalReranker()
    results=reranker.rerank(
        question=question,
        candidates=candidates,
        top_k=5
    )

    print("\n" + "=" * 70)
    print("RERANKED RESULTS")
    print("=" * 70)

    for result in results:
        print(f"\nRank: {result['rank']}")
        print(f"File: {result.get('filename','Unknown')}")
        print(f"Page: {result.get('page','Unknown')}")
        print(f"RRF score: {result.get('rrf_score')}")
        print(f"Reranker score: {result['rerank_score']:.4f}")
        print(f"Text: {result['text'][:700]}")
        print("-" * 70)

if __name__=="__main__":
    main()