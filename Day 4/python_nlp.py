def generate_ngrams(text, n):
    tokens = text.split()
    """
    I, AM, AN, INVESTMENT, Banker,...
    """
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return ngrams
 
 
text = "I am an investment banker with focus on APAC market"
 
unigrams = generate_ngrams(text, 1)
bigrams = generate_ngrams(text, 2)
trigrams = generate_ngrams(text, 3)
 
print("Unigrams:", unigrams)
print("Bigrams:", bigrams)
print("Trigrams:", trigrams)