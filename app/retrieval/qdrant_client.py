import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()
load_dotenv('.env.example')

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

def get_qdrant_client()->QdrantClient:
    """
    Create an authenticated Qdrant client.
    """
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise RuntimeError(
            "Missing QDRANT_URL or QDRANT_API_KEY in .env"
        )
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60
    )

if __name__=="__main__":
    client=get_qdrant_client()
    try:
        collections=client.get_collections()
        print("Successfully connected to Qdrant Cloud!")
        print("Existing collections:")
        if collections.collections:
            for collection in collections.collections:
                print(f"- {collection.name}")
        else:
            print("No collections found.")
    except Exception as e:
        print("Failed to connect to Qdrant Cloud.")
        print(f"Error: {e}")
    finally:
        client.close()