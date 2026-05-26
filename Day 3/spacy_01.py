# NLP Example with Stemming, Lemmatization, Stop Words, NER with Python lib Spacy

"""
Important Notes
A. spaCy does NOT include stemming (by design—it prefers linguistically correct lemmatization)
B. If you strictly need stemming, you’d normally bring back NLTK or another stemmer
C. For fintech applications, lemmatization is usually better than stemming

Why use this approach :

- Single library → simpler, fewer dependency issues
- Faster and production-ready
- Built-in pipeline (tokenization + NER + POS + lemma)

About Pseudo-stemming : 
Pseudo-stemming = Lemmatization + Lowercasing used as a substitute for stemming when using spaCy.

Why pseudo_stemming : 

- Spacy doesn't support Stemming
- Accuracy in domains like Fintech

"""


'''
Packages required : 
pip install spacy
python -m spacy download en_core_web_sm
'''

import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Sample fintech sentence
text = "Wells Fargo Bank approved a loan of USD 5 million to John Berry on 12th March 2025 in NYC."

# Process text
doc = nlp(text)

# ---------------- TOKENIZATION ----------------
tokens = [token.text for token in doc]
print("Tokens:", tokens)

# ---------------- STOP WORD REMOVAL ----------------
filtered_tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
print("After Stop Word Removal:", filtered_tokens)

# ---------------- LEMMATIZATION ----------------
lemmatized = [token.lemma_ for token in doc if not token.is_punct]
print("Lemmatized Words:", lemmatized)

# ---------------- "STEMMING" (Approximation) ----------------
# spaCy does NOT support true stemming.
# But we can approximate using lemma + lowercasing.
pseudo_stemmed = [token.lemma_.lower() for token in doc if not token.is_punct]
print("Pseudo-Stemmed:", pseudo_stemmed)

# ---------------- NAMED ENTITY RECOGNITION (NER) ----------------
print("\nNamed Entities:")
for ent in doc.ents:
    print(f"{ent.text} -> {ent.label_}")