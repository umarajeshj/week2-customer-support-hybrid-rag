"""Ingestion service for processing the ShopEasy JSON knowledge base
and pushing vectors into Pinecone using OpenAI embeddings.
"""

from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter


def ingest_documents(file_path: str, index_name: str | None = None) -> dict[str, Any]:
    """Loads the ShopEasy JSON knowledge base, chunks the content,
    generates OpenAI embeddings (512-dim), and stores them in Pinecone.
    """
    # 1. Validate environment keys
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is missing.")
    if not os.environ.get("PINECONE_API_KEY"):
        raise ValueError("PINECONE_API_KEY environment variable is missing.")

    # Determine index name (fallback to default if not provided)
    target_index = index_name or os.environ.get("PINECONE_INDEX_NAME", "shopeasy-basic-rag")

    # 2. Load the JSON knowledge base file
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find JSON knowledge base at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        raise ValueError("The JSON knowledge base must be a list of objects.")

    # 3. Convert JSON items into LangChain Document objects (properly unpacking nested metadata)
    documents = []
    for item in raw_data:
        content = item.get("content") or item.get("text") or json.dumps(item)
        
        # Extract metadata properly from the nested 'metadata' key
        raw_metadata = item.get("metadata", {})
        
        # Fallback if metadata is flat or structured differently
        if not isinstance(raw_metadata, dict):
            raw_metadata = {k: v for k, v in item.items() if k not in ["content", "text"]}

        # Ensure metadata values are valid types for Pinecone (strings, numbers, booleans)
        cleaned_metadata = {
            k: (str(v) if not isinstance(v, (str, int, float, bool)) else v)
            for k, v in raw_metadata.items()
        }

        documents.append(Document(page_content=content, metadata=cleaned_metadata))

    # 4. Chunk the documents for fine-grained retrieval
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    docs = text_splitter.split_documents(documents)

    # 5. Initialize OpenAI Embeddings (explicitly set to 512 dimensions to match your index setup)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        dimensions=512
    )

    # 6. Push chunks into Pinecone Vector Store
    vector_store = PineconeVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        index_name=target_index
    )

    return {
        "status": "success",
        "index_name": target_index,
        "total_documents_loaded": len(documents),
        "total_chunks_created": len(docs),
    }