# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
sample_text = [
    "The cat sat on the mat.",
    "The dog sat on the log.",
    "Cats and dogs are great pets.",
    "The mat and log are both places to sit."
]

from sklearn.preprocessing import OneHotEncoder
import numpy as np
from gensim.models import Word2Vec
import nltk
nltk.download('punkt_tab')
from sklearn.feature_extraction.text import TfidfVectorizer


# Preprocessing: Tokenize words
words = [word.lower() for sentence in sample_text for word in sentence.split()]

# Reshape words for OneHotEncoder
words_reshaped = np.array(words).reshape(-1, 1)

# Initialize OneHotEncoder
onehot_encoder = OneHotEncoder(sparse_output=False)

# Fit and transform words to one-hot encode
onehot_encoded = onehot_encoder.fit_transform(words_reshaped)

# Display the vocabulary and one-hot encoded vectors
print("Vocabulary:")
print(onehot_encoder.categories_)

print("\nOne-Hot Encoded Vectors:")
print(onehot_encoded)




# Preprocessing: Tokenize each sentence
tokenized_sentences = [nltk.word_tokenize(sentence.lower()) for sentence in sample_text]

# Train Word2Vec model
word2vec_model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)

# Get vector representation of words
word_vector_cat = word2vec_model.wv['cat']
print("Word2Vec vector for 'cat':")
print(word_vector_cat)

# Print vocabulary
print("\nWord2Vec Vocabulary:")
print(list(word2vec_model.wv.index_to_key))


from gensim.models import KeyedVectors

# Load the pre-trained GloVe embeddings
glove_file = "D:/glove.6B/glove.6B.100d.txt"
glove_model = KeyedVectors.load_word2vec_format(glove_file, binary=False, no_header=True)

# Access GloVe vector for a word
glove_vector_cat = glove_model['cat']
print("\nGloVe vector for 'cat':")
print(glove_vector_cat)

# Print vocabulary
print("\nGloVe Vocabulary:")
print(list(glove_model.key_to_index)[:10])  # Print first 10 words




# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the sample text
tfidf_matrix = tfidf_vectorizer.fit_transform(sample_text)

# Get feature names (words)
feature_names = tfidf_vectorizer.get_feature_names_out()

# Convert TF-IDF matrix to dense format
dense_tfidf = tfidf_matrix.todense()

# Display TF-IDF matrix
print("\nTF-IDF Matrix:")
for i, sentence in enumerate(dense_tfidf):
    print(f"Sentence {i+1}:")
    print(dict(zip(feature_names, sentence.tolist()[0])))
