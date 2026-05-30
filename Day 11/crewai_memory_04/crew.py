"""
Project Scope :

An enterprise customer-support AI system must:

Remember recent conversation context
Store historical customer interactions
Retrieve domain knowledge
Compress old conversations

Distinguish:
episodic memories
semantic memories
symbolic records
vector embeddings

About the memory :

1. Short-Term Memory
short_term_memory = {}

Used for:
•	Current active conversation 
•	Temporary reasoning 
•	Session context

2. Long-Term Memory
ChromaDB + SQLite
Used for:
•	Persistent enterprise knowledge 
•	Historical interactions 
•	Retrieval across sessions

3.Vector Memory Example
Semantic Search
•	retrieve_vector_memory(
    "refund problem"
)
Finds semantically similar conversations even if wording differs.

4. Symbolic Memory Example
Structured Rule Retrieval
retrieve_symbolic_memory("C101")
Returns structured facts:
•	billing issue 
•	refund history 
•	previous support actions

Memory Architecture :

| Memory Type       | Description                           | Storage Type             |
| ----------------- | ------------------------------------- | ------------------------ |
| Short-Term Memory | Temporary active conversation context | In-memory dictionary     |
| Long-Term Memory  | Persistent historical memory          | ChromaDB + SQLite        |
| Vector Memory     | Embedding-based semantic retrieval    | ChromaDB                 |
| Symbolic Memory   | Structured facts and rules            | SQLite                   |
| Episodic Memory   | Event-based experiences               | JSON + vector embeddings |
| Semantic Memory   | General enterprise knowledge          | JSON + vector embeddings |


"""

import pandas as pd

from crewai import Crew, Task

from agents.support_agent import (
    support_agent,
    process_customer
)

from agents.summarizer_agent import (
    summarizer_agent,
    summarize_conversation
)

from agents.memory_manager import MemoryManager

memory = MemoryManager()

df = pd.read_csv("data/customer_interactions.csv")

for idx, row in df.iterrows():

    text = row["conversation"]

    memory.store_vector_memory(
        doc_id=f"interaction_{idx}",
        text=text
    )

    memory.store_symbolic_memory(
        customer_id=row["customer_id"],
        issue_type=row["issue_type"],
        notes=row["agent_notes"]
    )

support_task = Task(
    description="""
    Handle enterprise customer support using:
    - vector memory
    - symbolic memory
    - episodic memory
    - semantic memory
    """,
    expected_output="Comprehensive support response",
    agent=support_agent
)

compression_task = Task(
    description="""
    Compress customer conversations while
    retaining critical enterprise context
    """,
    expected_output="Compressed memory summary",
    agent=summarizer_agent
)

crew = Crew(
    agents=[
        support_agent,
        summarizer_agent
    ],
    tasks=[
        support_task,
        compression_task
    ],
    verbose=True
)

if __name__ == "__main__":

    query = """
    Customer says:
    I was billed twice again and
    my refund is still pending.
    """

    result = process_customer("C101", query)

    print(result)

    summary = summarize_conversation(result)

    print(summary)

    crew.kickoff()