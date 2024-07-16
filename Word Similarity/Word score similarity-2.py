from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine
import scipy
print(scipy.__version__)

# 1. Load Pre-trained Embeddings
embeddings_path = r"C:\Users\hp\Downloads\GoogleNews-vectors-negative300.bin.gz"  # Replace with your embedding file
model = KeyedVectors.load_word2vec_format(embeddings_path, binary=True)

# 2. Similarity Calculation
def calculate_similarity(word1, word2):
    vector1 = model[word1]
    vector2 = model[word2]
    similarity = 1 - cosine(vector1, vector2)  # Cosine similarity (1 - distance)
    return similarity

# Example Usage
word1 = "cat"
word2 = "dog"
similarity_score = calculate_similarity(word1, word2)
print(f"Similarity between '{word1}' and '{word2}': {similarity_score}")
import scipy.linalg

# Test triu function
print(scipy.linalg.triu([[1, 2], [3, 4]]))
