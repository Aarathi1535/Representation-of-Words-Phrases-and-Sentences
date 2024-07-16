import nltk
nltk.download('wordnet')
nltk.download('brown')
from nltk.corpus import wordnet as wn
from nltk.corpus import brown
from collections import defaultdict
import numpy as np
from scipy.spatial.distance import cosine

# 1. Corpus Processing
corpus = brown.words()[:1000000]  # Take the first 1 million tokens(ELectronic collection of text samples in American English)
vocabulary = set(corpus)
cooccurrence_matrix = defaultdict(lambda: defaultdict(int))

for i in range(len(corpus) - 1):
    word1 = corpus[i]
    word2 = corpus[i + 1]
    if word1 in vocabulary and word2 in vocabulary:
        cooccurrence_matrix[word1][word2] += 1

# 2. WordNet Integration
def wordnet_similarity(word1, word2):
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)
    max_similarity = 0
    for synset1 in synsets1:
        for synset2 in synsets2:
            similarity = synset1.wup_similarity(synset2)
            if similarity is not None and similarity > max_similarity:
                max_similarity = similarity
    return max_similarity

# 3. Similarity Calculation
def calculate_similarity(word1, word2):
    cooccurrence_vector1 = [cooccurrence_matrix[word1].get(word, 0) for word in vocabulary]
    cooccurrence_vector2 = [cooccurrence_matrix[word2].get(word, 0) for word in vocabulary]
    cosine_similarity = 1 - cosine(cooccurrence_vector1, cooccurrence_vector2)  # Cosine similarity
    wordnet_sim = wordnet_similarity(word1, word2)
    # Combine scores (you can adjust weights)
    combined_similarity = 0.7 * cosine_similarity + 0.3 * wordnet_sim
    return combined_similarity

# Example Usage
word1 = "cat"
word2 = "dog"
similarity_score = calculate_similarity(word1, word2)
print(f"Similarity between '{word1}' and '{word2}': {similarity_score}") #Similarity between 'cat' and 'dog': 0.6502088895052344
