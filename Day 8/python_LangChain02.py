"""
In LangChain, chains are one of the core abstractions used to connect multiple steps into a single workflow for working with LLMs.

Think of a chain as:

> **Input → one or more processing steps → output**

A chain lets you combine:

* prompts
* language models
* tools
* memory
* retrieval systems
* custom Python functions

into a reusable pipeline.


# Why Chains Matter

Without chains, you'd manually write logic like:

```python
formatted_prompt = ...
response = llm(...)
parsed_output = ...
```

Chains give:

* modularity
* reusability
* composability
* cleaner code

# Types of Chains

## 1. Sequential Chains

Steps happen one after another.

Example:

1. Summarize article
2. Translate summary
3. Generate bullet points

```text
Article
   ↓
Summary
   ↓
Translation
   ↓
Bullet Points
```

---

## 2. Retrieval Chains (RAG)

Very common in AI apps.

Workflow:

1. User asks question
2. Retrieve documents from vector DB
3. Add docs to prompt
4. LLM answers using retrieved context

```text
Question
   ↓
Retriever
   ↓
Relevant Docs
   ↓
LLM
   ↓
Answer
```

This powers:

* chat with PDFs
* company knowledge bots
* AI search systems

"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple terms."
)

model = ChatOpenAI()

chain = prompt | model

response = chain.invoke({"topic": "quantum computing"})
print(response)