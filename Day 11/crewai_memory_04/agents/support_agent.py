#Short-Term Memory Agent

from crewai import Agent
from agents.memory_manager import MemoryManager

memory = MemoryManager()

short_term_memory = {}

support_agent = Agent(
    role="Enterprise Support Agent",
    goal="Provide personalized customer support using multiple memory systems",
    backstory="""
    Expert AI support specialist capable of using:
    - short-term memory
    - long-term memory
    - episodic memory
    - semantic memory
    """,
    verbose=True
)

def process_customer(customer_id, query):

    short_term_memory[customer_id] = query

    vector_results = memory.retrieve_vector_memory(query)

    symbolic_results = memory.retrieve_symbolic_memory(customer_id)

    episodic_memory = memory.load_episodic_memory()

    semantic_memory = memory.load_semantic_memory()

    response = f"""
    CUSTOMER QUERY:
    {query}

    SHORT TERM MEMORY:
    {short_term_memory.get(customer_id)}

    VECTOR MEMORY:
    {vector_results}

    SYMBOLIC MEMORY:
    {symbolic_results}

    EPISODIC MEMORY:
    {episodic_memory}

    SEMANTIC MEMORY:
    {semantic_memory}
    """

    return response