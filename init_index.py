import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Explicitly creates a standard vector index (records-based)
pc.create_index(
    name="shopeasy-basic-rag",
    dimension=512,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)

print("✅ Standard vector index created successfully!")