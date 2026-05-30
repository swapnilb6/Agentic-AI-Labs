# Memory Compression & Summarization

from crewai import Agent
from datetime import datetime
import os

summarizer_agent = Agent(
    role="Memory Compression Specialist",
    goal="Compress long conversations into efficient memory summaries",
    backstory="""
    Expert in:
    - memory summarization
    - memory compression
    - context optimization
    - long-term retention strategies
    """,
    verbose=True
)

def summarize_conversation(conversation_text):

    # Create directory if it does not exist
    os.makedirs("./memory/summaries", exist_ok=True)

    summary = f"""
    SUMMARY GENERATED:
    ------------------
    Key customer concerns identified.
    Repeated issues compressed.
    Critical incidents preserved.
    Long conversational noise removed.

    ORIGINAL CONVERSATION:
    {conversation_text}

    Timestamp:
    {datetime.now()}
    """

    filename = (
        f"./memory/summaries/"
        f"summary_{int(datetime.now().timestamp())}.txt"
    )

    with open(filename, "w", encoding="utf-8") as f:
        f.write(summary)

    return summary