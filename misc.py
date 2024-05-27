import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from pymongo import MongoClient
import tensorflow as tf
#from tensorflow.keras.preprocessing.text import Tokenizer
Tokenizer = tf.keras.preprocessing.text.Tokenizer
#from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
import numpy as np

# Step 1: Connect to MongoDB
client = MongoClient('localhost', 27017)  # Adjust the host and port if necessary
db = client['articles']  # Database name
collection = db['articles']  # Collection name

# Fetch all articles from the database
articles = list(collection.find())
article_contents = [article['content'] for article in articles]

if not article_contents:
    raise ValueError("No articles found in the database.")

# Step 2: Tokenize and create sequences
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(article_contents)
sequences = tokenizer.texts_to_sequences(article_contents)
padded_sequences = pad_sequences(sequences, padding='post')

if padded_sequences.size == 0:
    raise ValueError("Padded sequences array is empty.")

# Step 3: Create an embedding model
embedding_dim = 16
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim=embedding_dim, input_length=padded_sequences.shape[1]),
    tf.keras.layers.GlobalAveragePooling1D()
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 4: Get embeddings for the articles
article_embeddings = model.predict(padded_sequences)

# Function to get embedding for a given text
def get_embedding(text):
    seq = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(seq, padding='post', maxlen=padded_sequences.shape[1])
    
    if padded_seq.size == 0:
        raise ValueError("Input sequence resulted in an empty array after padding")
    
    return model.predict(padded_seq)

# Function to find the most similar article
def find_most_similar_article(keyword):
    keyword_embedding = get_embedding(keyword)[0]
    
    # Compute cosine similarity
    similarities = np.dot(article_embeddings, keyword_embedding) / (np.linalg.norm(article_embeddings, axis=1) * np.linalg.norm(keyword_embedding))
    
    if similarities.size == 0:
        raise ValueError("Similarities array is empty.")
    
    most_similar_index = np.argmax(similarities)
    
    most_similar_article = articles[most_similar_index]
    print(f"Most similar article to '{keyword}':")
    print(f"Title: {most_similar_article['title']}")
    print(f"Content: {most_similar_article['content']}\n")

# Get keyword input from the user
keyword = input("Enter a keyword to search for: ")

# Find and print the most similar article
find_most_similar_article(keyword)
