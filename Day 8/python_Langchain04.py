from langchain_text_splitters import RecursiveCharacterTextSplitter

with open("sample_doc_01.txt", "r", encoding="utf-8") as f:
    document = f.read()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)

docs = splitter.create_documents([document])

print(f"Total chunks: {len(docs)}")

print("\nFirst chunk:")
print(docs[0].page_content)