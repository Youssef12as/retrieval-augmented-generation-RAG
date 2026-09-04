# Module 4: LLMs and Text Generation

While the previous modules focused on finding the right data (Retrieval), this module focuses on the **"G" in RAG: Generation**. Once you have the context, how do you extract maximum value from the Large Language Model (LLM)?

This guide provides practical techniques for Prompt Engineering, controlling LLM behavior in production, handling hallucinations, and deciding when to use RAG vs. Fine-Tuning.

---

## 1. LLM Sampling Strategies (Controlling the Output)
In an enterprise setting, you need the LLM's output to be predictable. LLMs generate text one token (word piece) at a time, calculating probabilities for the next token. You can control this process using two main parameters: **Temperature** and **Top-p**.

- **Temperature (0.0 to 1.0+):** 
  - `0.0`: The model always picks the most probable next word. *Use this for RAG, coding, math, or exact data extraction.*
  - `0.7+`: The model explores less probable words, making it creative. *Use this for brainstorming or marketing copy.*
- **Top-p (Nucleus Sampling):** Only considers the top tokens whose cumulative probability reaches *p*. If Top-p is 0.9, it ignores the bottom 10% of improbable words.

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("TOGETHER_API_KEY"),
    base_url="https://api.together.xyz/v1"
)

def generate_strict_answer(prompt):
    """
    Highly deterministic setup for RAG and data extraction.
    We want facts, not creativity.
    """
    response = client.chat.completions.create(
        model="Qwen/Qwen3.5-9B",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0, # Zero creativity, maximum determinism
        top_p=0.1        # Extremely strict token selection
    )
    return response.choices[0].message.content
```

---

## 2. Advanced Prompt Engineering
A RAG prompt isn't just `Context + Query`. To get reliable results from smaller or cheaper models, you must use advanced prompting techniques.

### A. Few-Shot Prompting
Instead of just giving instructions, give the model **examples** of the exact input and expected output format. This practically guarantees the model will return valid JSON or specific structures.

```python
few_shot_prompt = """
You are a helpful assistant that classifies customer complaints.

Example 1:
User: "My laptop screen is cracked."
Category: Hardware

Example 2:
User: "I can't reset my password."
Category: IT Support

User: "The printer on the 3rd floor is out of ink."
Category: """
# The LLM will confidently complete this with "Hardware" or "IT Support"
```

### B. Chain of Thought (CoT)
If a task requires reasoning, force the LLM to explain its logic *before* giving the final answer. This prevents hallucinated jumps in logic.

```python
cot_rag_prompt = """
Based on the following documents, answer the user's question.

Documents:
Doc 1: "Refunds for electronics take 14 days."
Doc 2: "Refunds for clothing take 30 days."

Question: "I bought a TV and a shirt. How long will it take to refund both?"

Instructions:
1. Think step-by-step about the TV.
2. Think step-by-step about the shirt.
3. Provide your final answer.

Reasoning:
"""
```

---

## 3. Handling Hallucinations
Even with RAG, an LLM might hallucinate (make up facts) if the retrieved context doesn't actually contain the answer. 

**The Fix:** Explicitly instruct the LLM to admit ignorance.

```python
anti_hallucination_prompt = f"""
You are a strict, factual enterprise assistant.
Answer the question using ONLY the provided context. 

Context: {retrieved_context}
Question: {user_query}

CRITICAL RULES:
- If the context does not contain the answer, you MUST reply exactly with: "I don't have enough information to answer this."
- Do not use outside knowledge.
"""
```

---

## 4. Evaluating LLM Performance (LLM-as-a-Judge)
How do you know if your RAG system is getting better or worse when you change the prompt? Standard metrics (like BLEU or ROUGE) fail because they only check for exact word matches.

**The Industry Standard:** Use a large, powerful LLM (like GPT-4) to grade the output of your production LLM.

```python
def evaluate_rag_response(question, context, generated_answer):
    """
    Uses an LLM as an automated judge to check for Groundedness.
    """
    judge_prompt = f"""
    You are an expert evaluator. 
    Question: {question}
    Context: {context}
    Generated Answer: {generated_answer}
    
    Does the Generated Answer rely entirely on the Context, without making up facts?
    Reply with only "YES" or "NO".
    """
    
    evaluation = client.chat.completions.create(
        model="meta-llama/Llama-3-70b-chat-hf", # Use a very smart model to judge
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0.0
    ).choices[0].message.content.strip()
    
    return evaluation == "YES"
```
*(Frameworks like **Ragas** and **TruLens** automate this process for enterprise pipelines).*

---

## 5. RAG vs. Fine-Tuning
A common enterprise question: *"Should we use RAG, or should we Fine-Tune a model on our data?"*

- **Use RAG when:**
  - You need to query highly dynamic, frequently changing data (e.g., daily news, stock prices).
  - You need source citations (knowing exactly which document the answer came from).
  - You have strict access controls (Document A is only visible to HR).
- **Use Fine-Tuning when:**
  - You want the model to learn a specific *tone* or *format* (e.g., writing code in your company's proprietary programming language).
  - RAG is too slow or you run out of context window space.
- **The Ideal State:** Use Fine-Tuning to teach the model *how* to talk, and RAG to give it *what* to talk about.

---

## 6. Introduction to Agentic RAG
Standard RAG is a straight line: `Retrieve -> Generate -> Output`.
**Agentic RAG** allows the LLM to think, loop, and use tools.

Instead of retrieving immediately, an Agent might:
1. Receive the query: *"Compare our 2022 revenue to our 2023 revenue."*
2. Realize it needs two searches.
3. Call a `search_tool` for 2022.
4. Call a `search_tool` for 2023.
5. Combine the data and generate the final answer.
---
