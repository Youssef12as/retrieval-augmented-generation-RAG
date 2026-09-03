# Module 2: Information Retrieval and Search Foundations

This module focuses on the **"R" in RAG: Retrieval**. In a corporate or production environment, the LLM is only as good as the context you feed it. If your search engine returns garbage, the LLM will generate garbage (Garbage In, Garbage Out).

This guide provides an enterprise-focused, highly practical overview of different search architectures, with copy-pasteable Python code snippets that you can immediately use in your company's systems.

---

## 1. Retriever Architecture Overview
In an enterprise RAG system, your data usually lives in a Vector Database (like Weaviate, Pinecone, Qdrant, or Milvus) or a Search Engine (like Elasticsearch). 

When a user asks a query, the retriever pipeline typically executes the following steps:
1. **Pre-filtering (Metadata Filtering):** Narrow down the dataset (e.g., "Only search documents from 2024" or "Only search HR policies").
2. **First-Stage Retrieval:** Use a fast algorithm (Keyword, Semantic, or Hybrid) to fetch the top 100 relevant chunks.
3. **Re-ranking (Optional but recommended):** Use a heavier, more accurate model (like a Cross-Encoder) to re-order the top 100 chunks and pick the top 5 for the LLM.

---

## 2. Metadata Filtering
Before running expensive text searches, always filter by metadata if possible. It drastically improves speed and accuracy.

**Business Use Case:** A user asks, *"What is our refund policy?"* You only want to search documents tagged with `department: customer_service` and `status: active`.

```python
# Conceptual example of metadata filtering
documents = [
    {"text": "Refunds are processed in 3 days.", "dept": "support", "year": 2024},
    {"text": "Refunds used to take 10 days.", "dept": "support", "year": 2022},
    {"text": "Employees get 20 vacation days.", "dept": "hr", "year": 2024}
]

def filter_docs(docs, filters):
    return [d for d in docs if all(d.get(k) == v for k, v in filters.items())]

# Only search active support documents from 2024
filtered_docs = filter_docs(documents, {"dept": "support", "year": 2024})
print(filtered_docs) 
# Output: [{'text': 'Refunds are processed in 3 days.', 'dept': 'support', 'year': 2024}]
```
*Note: In production, your Vector DB (e.g., Pinecone/Weaviate) handles this natively at the database level.*

---

## 3. Keyword Search (Sparse Retrieval)
Keyword search relies on exact word matches. It is incredibly fast and works best for specific nouns, IDs, or acronyms (e.g., "Error Code 404", "Employee ID 10932").

### A. TF-IDF (Term Frequency - Inverse Document Frequency)
TF-IDF measures how important a word is to a document. 
- **TF:** How often does the word appear in this doc?
- **IDF:** How rare is this word across *all* docs? (Rare words carry more weight).

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

corpus = [
    "How to reset your corporate password.",
    "The corporate policy for remote work.",
    "How to reset the office printer."
]
query = ["reset password"]

# 1. Initialize and fit the vectorizer
vectorizer = TfidfVectorizer()
X_corpus = vectorizer.fit_transform(corpus)

# 2. Transform the query
X_query = vectorizer.transform(query)

# 3. Calculate similarity
scores = cosine_similarity(X_query, X_corpus)[0]

# Print top match
best_idx = scores.argmax()
print(f"Top match: {corpus[best_idx]} (Score: {scores[best_idx]:.2f})")
```

### B. BM25 (Best Match 25)
BM25 is the industry standard for keyword search (used by Elasticsearch). It improves upon TF-IDF by adding a **saturation limit** (if a word appears 100 times, it isn't 100x more important than if it appears 10 times) and **document length normalization** (penalizes extremely long documents).

```python
# pip install rank_bm25
from rank_bm25 import BM25Okapi

corpus = [
    "How to reset your corporate password.",
    "The corporate policy for remote work.",
    "How to reset the office printer."
]
# Tokenize the corpus (split by spaces)
tokenized_corpus = [doc.lower().split(" ") for doc in corpus]

# Initialize BM25
bm25 = BM25Okapi(tokenized_corpus)

query = "reset password"
tokenized_query = query.lower().split(" ")

# Get scores
doc_scores = bm25.get_scores(tokenized_query)
best_doc = bm25.get_top_n(tokenized_query, corpus, n=1)

print(f"BM25 Top Match: {best_doc[0]}")
```

---

## 4. Semantic Search (Dense Retrieval)
Keyword search fails at synonyms. If a user searches *"I can't log in"*, BM25 won't match *"Password reset instructions"* because the exact words don't overlap. 
**Semantic search** converts text into dense vectors (embeddings) that capture meaning.

**Embedding Model Deepdive:** The industry standard for enterprise text embeddings is currently models from OpenAI (e.g., `text-embedding-3-small`) or open-source models like `BAAI/bge-large-en-v1.5` or `nomic-embed-text`.

```python
# pip install sentence-transformers
from sentence_transformers import SentenceTransformer, util

# Load a fast, open-source embedding model
model = SentenceTransformer('BAAI/bge-small-en-v1.5')

corpus = [
    "Password reset instructions.",
    "The corporate policy for remote work.",
    "How to fix the office printer."
]

# 1. Pre-compute embeddings for the database
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

# 2. User searches with a conceptually similar (but word-different) query
query = "I can't log in to my account"
query_embedding = model.encode(query, convert_to_tensor=True)

# 3. Compute Cosine Similarity
hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=1)[0]

top_hit_id = hits[0]['corpus_id']
print(f"Semantic Top Match: {corpus[top_hit_id]} (Score: {hits[0]['score']:.2f})")
```

---

## 5. Hybrid Search
**The Enterprise Gold Standard.** 
Keyword search is great for specific terms ("Q3 Financial Report 2023"). Semantic search is great for concepts ("How did we do financially last year?"). **Hybrid search** combines both using a scoring algorithm called **RRF (Reciprocal Rank Fusion)**.

RRF formula: `RRF_Score = 1 / (k + BM25_Rank) + 1 / (k + Semantic_Rank)` (where k is usually 60).

```python
def reciprocal_rank_fusion(bm25_ranks, semantic_ranks, k=60):
    """
    bm25_ranks: dict of {doc_id: rank_position}
    semantic_ranks: dict of {doc_id: rank_position}
    """
    rrf_scores = {}
    all_docs = set(bm25_ranks.keys()).union(set(semantic_ranks.keys()))
    
    for doc in all_docs:
        # If doc is missing from a list, give it a penalty rank (e.g., 1000)
        bm25_rank = bm25_ranks.get(doc, 1000)
        sem_rank = semantic_ranks.get(doc, 1000)
        
        rrf_scores[doc] = (1.0 / (k + bm25_rank)) + (1.0 / (k + sem_rank))
        
    # Sort docs by RRF score descending
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

# Example output from two systems
bm25_ranking = {"docA": 1, "docB": 2, "docC": 3}
semantic_ranking = {"docC": 1, "docA": 2, "docB": 3}

final_ranking = reciprocal_rank_fusion(bm25_ranking, semantic_ranking)
print(f"Hybrid RRF Final Ranking: {final_ranking}")
# Notice how docA and docC get boosted because they performed well in at least one system.
```

---

## 6. Evaluating Retrieval & Metrics
In a production system, you must measure if your retriever is actually pulling the right documents. You create a "Golden Dataset" of queries and their known correct document IDs.

### Key Metrics:
1. **Precision@K:** Out of the top K results returned, what percentage were actually relevant?
2. **Recall@K:** Out of all the relevant documents that exist, what percentage did we manage to find in the top K?
3. **MRR (Mean Reciprocal Rank):** How far down the list is the *first* correct answer? If the correct answer is at position 1, score = 1. If at position 2, score = 0.5. If at position 3, score = 0.33. 
4. **NDCG (Normalized Discounted Cumulative Gain):** Evaluates the entire ranking order. It rewards systems that put the highly relevant documents at the very top, and penalizes systems that push them down.

```python
def calculate_mrr(predictions, ground_truth_id):
    """
    predictions: list of retrieved doc_ids ordered by rank
    ground_truth_id: the ID of the correct document
    """
    try:
        rank = predictions.index(ground_truth_id) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0 # Document was not retrieved

preds = ["doc4", "doc1", "doc7"]
truth = "doc1"
print(f"MRR Score: {calculate_mrr(preds, truth):.2f}") 
# Output: 0.50 (Because it was the 2nd result)
```


