#pip install langchain-classic

"""
LANGCHAIN + OPENAI QUICK QUIZ BOT
=================================

This example teaches the MAIN LangChain concepts:

1. LLM / Chat Model
2. Prompt Templates
3. Chains
4. Memory
5. Message History
6. Structured Conversation Flow

Compatible with:
- Python 3.11+
- LangChain 1.x
- OpenAI SDK 1.x

INSTALL
-------

pip install \
  langchain \
  langchain-openai

SET ENVIRONMENT VARIABLE
------------------------

Linux/macOS:
export OPENAI_API_KEY="your_key"

Windows PowerShell:
$env:OPENAI_API_KEY="your_key"

RUN
---

python bot_langchain01.py

Note: 

ConversationChain is considered older architecture now.

Modern LangChain prefers:

LCEL (LangChain Expression Language)
LangGraph
Runnable pipelines

"""


from langchain_openai import ChatOpenAI

# Modern imports
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationChain


# ==========================================
# OPENAI MODEL
# ==========================================

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.3
)

# ==========================================
# MEMORY
# ==========================================

memory = ConversationBufferMemory()

"""
ConversationBufferMemory in LangChain is a foundational memory module that stores the entire, 
raw history of a conversation in a simple buffer. It ensures LLMs maintain context over multiple interactions, 
making it ideal for chat applications needing to reference past messages, 
though it can become inefficient with very long dialogues due to token limits
"""


# ==========================================
# CHAIN
# ==========================================

"""
The ConversationChain is a core LangChain utility designed for building stateful chat applications.

A standard ConversationChain consists of three primary elements:

a. LLM/Chat Model: The underlying engine (e.g., OpenAI, Anthropic) that generates responses.

b. Memory: A module that stores and retrieves chat history. By default, it uses ConversationBufferMemory, 
which maintains a full transcript of the dialogue.

c. Prompt Template: A fixed template that formats the history and user input into a single prompt for the LLM, 
often defining the AI's persona as "talkative and helpful"
"""
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

print("LangChain Quiz Bot Started!")
print("Type 'exit' to quit.\n")

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    response = conversation.predict(input=user_input)

    print(f"\nBot: {response}\n")