# Module 3: Information Retrieval with Vector Databases

This module bridges the gap between theoretical search algorithms and **production-grade enterprise infrastructure**. When building a RAG system for a company with millions of documents, you cannot calculate cosine similarity in memory across the entire dataset. You need a dedicated **Vector Database**, advanced **Chunking** strategies, and a **Reranking** step to maximize accuracy.

This guide contains practical, copy-pasteable Python implementations for these enterprise patterns.

---

## 1. Vector Databases & ANN (Approximate Nearest Neighbors)
When you compute standard Cosine Similarity (K-Nearest Neighbors or KNN), the system compares your query against *every single document*. This is perfectly fine for 1,000 documents, but impossibly slow for 10,000,000 documents.

**The Solution: ANN (Approximate Nearest Neighbors).**
Vector Databases (like Weaviate, Milvus, Pinecone, or Qdrant) use algorithms like **HNSW (Hierarchical Navigable Small World)** to organize vectors into a graph. Instead of checking every document, the database navigates the graph to find the *approximate* closest matches in milliseconds.

### Weaviate API Quickstart
Weaviate is a popular open-source Vector Database. Here is how you connect, insert data, and query it in Python.

```python
# pip install weaviate-client
import weaviate
import weaviate.classes as wvc

# 1. Connect to Weaviate (Local Docker instance)
client = weaviate.connect_to_local()

try:
    # 2. Create a Collection (Schema/Table)
    questions = client.collections.create(
        name="Question",
        vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai()
    )

    # 3. Insert Data
    questions.data.insert_many([
        {"answer": "DNA", "question": "What is the building block of life?", "category": "biology"},
        {"answer": "42", "question": "What is the meaning of life?", "category": "philosophy"}
    ])

    # 4. Perform a Vector Search (Semantic Search)
    response = questions.query.near_text(
        query="biology foundations",
        limit=2
    )

    for obj in response.objects:
        print(obj.properties)

finally:
    client.close()
```

---

## 2. Document Chunking
You cannot feed a 500-page PDF into an embedding model or an LLM. You must split it into smaller "chunks". 

### A. Basic Chunking (Fixed-Size)
Splitting text strictly by character count. *Warning: This often cuts words or sentences in half, destroying context.*

### B. Advanced Chunking (Recursive Character Splitting)
This is the industry standard. It tries to split by double newlines (`\n\n`), then single newlines (`\n`), then spaces (` `), ensuring that paragraphs and sentences stay together as much as possible.

```python
# pip install langchain-text-splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """Our company refund policy states that you can return items within 30 days.

However, electronic devices must be returned within 14 days and in their original packaging. 

For inquiries, contact support@company.com."""

# Set chunk size to 100 characters with an overlap of 20 characters
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

chunks = text_splitter.split_text(text)
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---\n{chunk}")
```
*Note the `chunk_overlap`. Overlap ensures that if a concept spans across the border of two chunks, the meaning isn't lost.*

---

## 3. Query Parsing
Users rarely type perfect semantic queries. They type things like: *"Show me the financial reports from Q3 2023 about marketing expenses."*

A naive vector search might struggle with the exact date constraint. **Query Parsing** uses a lightweight LLM step *before* retrieval to extract metadata filters and rewrite the query.

```python
# Conceptual Query Parsing
user_query = "What were our marketing expenses in Q3 2023?"

# 1. LLM extracts the intent and metadata
parsed_query = {
    "search_intent": "marketing expenses costs",
    "filters": {
        "department": "marketing",
        "year": 2023,
        "quarter": "Q3"
    }
}

# 2. You pass this parsed object to your Vector Database (e.g., Weaviate)
# to perform a Metadata-Filtered Vector Search.
```

---

## 4. Reranking (Cross-Encoders) & ColBERT
In Module 2, we learned that Vector Databases use "Bi-Encoders" (they embed the query and document separately and compute cosine similarity). This is fast but misses deep linguistic nuance.

**Reranking Pattern:**
1. **First Stage (Fast):** Use a Vector Database to fetch the top 100 documents.
2. **Second Stage (Accurate):** Use a **Cross-Encoder** to score the relationship between the query and those 100 documents simultaneously. Cross-Encoders are too slow to run on millions of documents, but perfect for reranking 100 documents.

### Implementing a Reranker with `sentence-transformers`

```python
from sentence_transformers import CrossEncoder
import numpy as np

# Load a pre-trained Cross-Encoder model
# This model outputs a relevance score between 0 and 1
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

query = "How do I reset my password?"
# Assume these 3 documents were returned by our fast Vector Database
retrieved_docs = [
    "To reset your printer, hold the power button.",
    "Click 'Forgot Password' on the login screen to receive a reset link.",
    "Our password policy requires 12 characters and a symbol."
]

# We format the input as pairs: (query, document)
pairs = [[query, doc] for doc in retrieved_docs]

# The cross-encoder evaluates the pairs together
scores = reranker.predict(pairs)

# Sort the documents by their new Cross-Encoder score
ranked_indices = np.argsort(-scores)

print("--- Reranked Results ---")
for idx in ranked_indices:
    print(f"Score: {scores[idx]:.2f} | Doc: {retrieved_docs[idx]}")
```
*Notice how the printer reset document might have had high keyword similarity, but the Cross-Encoder pushes the actual login password document to the top because it understands the semantic relationship better.*

### What is ColBERT?
ColBERT (Contextualized Late Interaction over BERT) is a specialized retrieval architecture. Instead of squashing a whole document into a single vector, ColBERT creates a vector for *every single token/word* in the document. When querying, it compares the query's token vectors against the document's token vectors (Late Interaction). It is highly accurate and frequently used in advanced RAG pipelines.

