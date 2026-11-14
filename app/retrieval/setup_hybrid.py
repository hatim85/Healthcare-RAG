from qdrant_client import models
from app.ingestion.indexer import COLLECTION_NAME
from app.retrieval.qdrant_client import get_qdrant_client

def setup_hybrid_search():
    client=get_qdrant_client()
    try:
        collection=client.get_collection(COLLECTION_NAME)
        vectors=collection.config.params.vectors
        if not isinstance(vectors,dict) or "dense" not in vectors:
            raise RuntimeError(
                "The collection does not contain the expected"
                "'dense vector."
            )
        # Delete existing text index (if any) so we can recreate
        # it with BM25 scoring enabled — required for TextQuery.
        if "text" in collection.payload_schema:
            client.delete_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="text",
                wait=True
            )
            print("Deleted old text index (recreating with scoring).")

        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="text",
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.WORD,
                min_token_len=2,
                max_token_len=40,
                lowercase=True,
                scoring=True,
            ),
            wait=True
        )
        print("Created full-text index on 'text' with BM25 scoring.")

        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="tenant_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True
        )
        print("Created keyword index on 'tenant_id'.")
        print("Hybrid search setup complete.")
    finally:
        client.close()

if __name__=="__main__":
    setup_hybrid_search()
