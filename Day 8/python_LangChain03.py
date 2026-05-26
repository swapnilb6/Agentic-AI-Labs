# pip install -U langchain langchain-text-splitters
 
"""
How Chunking works :
 
RecursiveCharacterTextSplitter tries to preserve meaning by splitting text in this order:
 
Paragraphs (\n\n)
Lines (\n)
Spaces ( )
Characters ("")
 
 
"""
 
from langchain_text_splitters import RecursiveCharacterTextSplitter
 
# Example document
text = """
Artificial Intelligence (AI) is transforming industries worldwide.
Machine learning is a subset of AI that enables systems to learn from data.
Deep learning uses neural networks with multiple layers.
LangChain helps developers build LLM-powered applications.
Chunking is important for Retrieval-Augmented Generation (RAG).
"""
 
# Create the splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,      # max characters per chunk
    chunk_overlap=20,    # overlap between chunks
)
 
# Split the text
chunks = splitter.split_text(text)
 
# Print chunks
for i, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {i} ---")
    print(chunk)