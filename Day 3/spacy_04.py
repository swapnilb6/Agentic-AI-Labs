"""
Program - Work with Vectorization


Packages required: 
pip install scikit-learn gensim sentence-transformers

IMPORTANT NOTE : 

1. TF-IDF
Large sparse vectors
No semantic understanding
“loan” and “credit” are unrelated

2. Word2Vec
Dense vectors
Learns word relationships
Needs averaging for sentence meaning

3. Embeddings
Best semantic understanding
Sentence-level meaning
Works well for similarity, clustering, search

===> Fintech Scenario 
For real applications:

TF-IDF → rule-based systems, keyword search
Word2Vec → basic semantic similarity
Embeddings → fraud detection, intent classification, chatbots

"""



documents = [
    "Customer applied for a loan from Wells Fargo Bank",
    "Citi Bank approved the credit card request",
    "Fraud detected in transaction from unknown account",
    "User transferred money to another account",
    "Loan repayment was delayed due to insufficient balance"
]


#TF-IDF (Sparse, Count-Based)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
tfidf_vectors = vectorizer.fit_transform(documents)

print("TF-IDF Shape:", tfidf_vectors.shape)
print("Feature Names:", vectorizer.get_feature_names_out())

# Convert first document to array
print("TF-IDF Vector (Doc 1):\n", tfidf_vectors[0].toarray())


#Word2Vec (Static Embeddings)

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# Tokenize
tokenized_docs = [simple_preprocess(doc) for doc in documents]

# Train model
w2v_model = Word2Vec(sentences=tokenized_docs, vector_size=50, window=3, min_count=1)

# Get vector for a word
print("Word2Vec vector for 'loan':\n", w2v_model.wv['loan'])

# Document vector = average of word vectors
import numpy as np

def document_vector(doc):
    words = simple_preprocess(doc)
    return np.mean([w2v_model.wv[word] for word in words], axis=0)

print("Word2Vec Doc Vector:\n", document_vector(documents[0]))

# Embeddings (Contextual, Modern)

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(documents)

print("Embedding Shape:", embeddings.shape)
print("Embedding (Doc 1):\n", embeddings[0])