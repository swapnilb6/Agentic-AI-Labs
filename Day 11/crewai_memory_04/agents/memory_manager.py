# Memory Manager

import chromadb
import sqlite3
import json
from sentence_transformers import SentenceTransformer

class MemoryManager:

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        self.client = chromadb.PersistentClient(path="./memory/vector_store")

        self.collection = self.client.get_or_create_collection(
            name="enterprise_memory"
        )

        self.conn = sqlite3.connect("./memory/symbolic_memory.db")
        self.cursor = self.conn.cursor()

        self.create_symbolic_tables()

    def create_symbolic_tables(self):
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS customer_memory (
            customer_id TEXT,
            issue_type TEXT,
            notes TEXT
        )
        """)
        self.conn.commit()

    def store_vector_memory(self, doc_id, text):
        embedding = self.model.encode(text).tolist()

        self.collection.add(
            ids=[doc_id],
            documents=[text],
            embeddings=[embedding]
        )

    def retrieve_vector_memory(self, query, top_k=3):
        query_embedding = self.model.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        return results["documents"]

    def store_symbolic_memory(self, customer_id, issue_type, notes):
        self.cursor.execute("""
        INSERT INTO customer_memory
        VALUES (?, ?, ?)
        """, (customer_id, issue_type, notes))

        self.conn.commit()

    def retrieve_symbolic_memory(self, customer_id):
        self.cursor.execute("""
        SELECT * FROM customer_memory
        WHERE customer_id=?
        """, (customer_id,))

        return self.cursor.fetchall()

    def load_semantic_memory(self):
        with open("data/semantic_knowledge.json") as f:
            return json.load(f)

    def load_episodic_memory(self):
        with open("data/episodic_events.json") as f:
            return json.load(f)