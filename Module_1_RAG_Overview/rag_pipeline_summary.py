"""
RAG Pipeline Summary — Module 1, Assignment Solution
=====================================================
Standalone, clean extraction of the complete RAG pipeline
built in the Module 1 programming assignment.

This file consolidates all the logic from C1M1_Assignment_Solution.ipynb
into a single runnable script. It is NOT dependent on the Coursera proxy;
set your TOGETHER_API_KEY environment variable to run locally.

Pipeline steps:
    1. Load dataset (news_data_dedup.csv)
    2. Load pre-computed embeddings (embeddings.joblib)
    3. Encode user query with the same embedding model
    4. Retrieve top-k documents via cosine similarity
    5. Format retrieved documents into a context string
    6. Build an augmented prompt (query + context)
    7. Send prompt to LLM and return the response
"""

import json
import os
import numpy as np
import pandas as pd
import joblib
from typing import List, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
LLM_MODEL = "Qwen/Qwen3.5-9B"
DATA_PATH = "news_data_dedup.csv"
EMBEDDINGS_PATH = "embeddings.joblib"

DEFAULT_PROMPT_TEMPLATE = (
    "Answer the user query below. There will be provided additional "
    "information for you to compose your answer. The relevant information "
    "provided is from 2024 and it should be added as your overall knowledge "
    "to answer the query; you should not rely only on this information to "
    "answer the query, but add it to your overall knowledge.\n\n"
    "Query: {query}\n\n"
    "2024 News:\n{documents}"
)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_dataset(path: str = DATA_PATH) -> List[Dict]:
    """Load the news dataset from CSV and return as a list of dicts."""
    df = pd.read_csv(path)
    df["published_at"] = pd.to_datetime(df["published_at"]).dt.strftime("%Y-%m-%d")
    df["updated_at"] = pd.to_datetime(df["updated_at"]).dt.strftime("%Y-%m-%d")
    return df.to_dict(orient="records")


def load_embeddings(path: str = EMBEDDINGS_PATH) -> np.ndarray:
    """Load pre-computed document embeddings."""
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def retrieve(
    query: str,
    embeddings: np.ndarray,
    model: SentenceTransformer,
    top_k: int = 5,
) -> List[int]:
    """
    Encode the query and return indices of the top-k most similar documents.

    Steps:
        1. Encode the query into a dense vector using the same model
           that produced the document embeddings.
        2. Compute cosine similarity between the query vector and every
           document vector.
        3. Sort by descending similarity and return the top-k indices.
    """
    query_embedding = model.encode(query).reshape(1, -1)
    scores = cosine_similarity(query_embedding, embeddings)[0]
    ranked_indices = np.argsort(-scores)
    return ranked_indices[:top_k].tolist()


def query_news(dataset: List[Dict], indices: List[int]) -> List[Dict]:
    """Return the documents at the given indices."""
    return [dataset[i] for i in indices]


def get_relevant_data(
    query: str,
    dataset: List[Dict],
    embeddings: np.ndarray,
    model: SentenceTransformer,
    top_k: int = 5,
) -> List[Dict]:
    """Retrieve and return the top-k relevant documents for a query."""
    indices = retrieve(query, embeddings, model, top_k)
    return query_news(dataset, indices)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def format_relevant_data(documents: List[Dict]) -> str:
    """
    Format a list of retrieved documents into a structured context string.

    Each document is rendered as:
        Title: ..., Description: ..., Published at: ...
        URL: ...
    """
    formatted = []
    for doc in documents:
        entry = (
            f"Title: {doc['title']}, "
            f"Description: {doc['description']}, "
            f"Published at: {doc['published_at']}\n"
            f"URL: {doc['url']}"
        )
        formatted.append(entry)
    return "\n".join(formatted)


# ---------------------------------------------------------------------------
# Prompt Construction
# ---------------------------------------------------------------------------
def build_augmented_prompt(
    query: str,
    dataset: List[Dict],
    embeddings: np.ndarray,
    model: SentenceTransformer,
    top_k: int = 5,
    template: Optional[str] = None,
) -> str:
    """
    Build the full augmented prompt by retrieving relevant documents
    and injecting them into the prompt template.
    """
    if template is None:
        template = DEFAULT_PROMPT_TEMPLATE

    relevant_docs = get_relevant_data(query, dataset, embeddings, model, top_k)
    context = format_relevant_data(relevant_docs)
    return template.format(query=query, documents=context)


# ---------------------------------------------------------------------------
# LLM Call
# ---------------------------------------------------------------------------
def call_llm(
    prompt: str,
    api_key: Optional[str] = None,
    llm_model: str = LLM_MODEL,
    max_tokens: int = 500,
) -> str:
    """
    Send a prompt to the Together.ai API and return the response content.

    Requires either:
        - api_key parameter, or
        - TOGETHER_API_KEY environment variable
    """
    from together import Together

    key = api_key or os.environ.get("TOGETHER_API_KEY")
    if not key:
        raise ValueError(
            "No API key provided. Set TOGETHER_API_KEY or pass api_key."
        )

    client = Together(api_key=key)
    response = client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------
def rag_pipeline(
    query: str,
    dataset: List[Dict],
    embeddings: np.ndarray,
    embedding_model: SentenceTransformer,
    top_k: int = 5,
    use_rag: bool = True,
    api_key: Optional[str] = None,
) -> str:
    """
    End-to-end RAG pipeline.

    Args:
        query:           User question.
        dataset:         List of news article dicts.
        embeddings:      Pre-computed document embeddings.
        embedding_model: SentenceTransformer model for encoding queries.
        top_k:           Number of documents to retrieve.
        use_rag:         If False, sends the raw query without retrieval.
        api_key:         Together.ai API key (optional if env var is set).

    Returns:
        The LLM-generated response string.
    """
    if use_rag:
        prompt = build_augmented_prompt(
            query, dataset, embeddings, embedding_model, top_k
        )
    else:
        prompt = query

    return call_llm(prompt, api_key=api_key)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading dataset...")
    dataset = load_dataset()
    print(f"  {len(dataset)} articles loaded.")

    print("Loading embeddings...")
    embeddings = load_embeddings()
    print(f"  Shape: {embeddings.shape}")

    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # --- Demo: show retrieved documents without calling the LLM ---
    demo_query = "Tell me about the US GDP in the past 3 years."
    print(f"\nQuery: {demo_query}\n")

    docs = get_relevant_data(demo_query, dataset, embeddings, embedding_model, top_k=3)
    print("Retrieved documents:")
    print("-" * 60)
    print(format_relevant_data(docs))
    print("-" * 60)

    prompt = build_augmented_prompt(
        demo_query, dataset, embeddings, embedding_model, top_k=3
    )
    print("\nFull augmented prompt:")
    print("=" * 60)
    print(prompt)
    print("=" * 60)

    # Uncomment below to call the LLM (requires TOGETHER_API_KEY):
    # response = rag_pipeline(
    #     demo_query, dataset, embeddings, embedding_model,
    #     top_k=3, use_rag=True
    # )
    # print("\nLLM Response (with RAG):")
    # print(response)
