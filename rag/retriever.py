import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)


import pickle
import re
import string

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Text preprocessing
# -----------------------------

stop_words = set(stopwords.words("english"))

# Keep legal words because they carry important meaning.
legal_keep = {"shall", "not", "may", "must"}
stop_words = stop_words - legal_keep

lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = str(text).lower()

    # Same punctuation removal used in the original notebook
    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator)

    # Same tokenization used in the original notebook
    tokens = word_tokenize(text)

    # Same stopword filtering
    tokens = [word for word in tokens if word not in stop_words]

    # Same lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)


# -----------------------------
# Load legal clause dataset
# -----------------------------

print("Loading legal clause dataset...")

df = pd.read_csv("Data/legal_contract_clauses.csv")

texts = df["clause_text"].astype(str).tolist()
labels = df["risk_level"].astype(str).tolist()
types = df["clause_type"].astype(str).tolist()


# -----------------------------
# Create TF-IDF index
# -----------------------------

print("Preprocessing legal clauses...")

processed_texts = [preprocess_text(text) for text in texts]

print("Building TF-IDF index...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
)

document_matrix = vectorizer.fit_transform(processed_texts)

print(
    f"TF-IDF index ready: "
    f"{document_matrix.shape[0]} documents, "
    f"{document_matrix.shape[1]} features"
)


# -----------------------------
# Retrieval
# -----------------------------

def retrieve(query, k=3):
    processed_query = preprocess_text(query)

    query_vector = vectorizer.transform([processed_query])

    similarities = cosine_similarity(
        query_vector,
        document_matrix,
    ).flatten()

    top_indices = np.argsort(similarities)[-k:][::-1]

    results = []

    for idx in top_indices:
        results.append(
            {
                "clause": texts[idx],
                "risk": labels[idx],
                "type": types[idx],
            }
        )

    return results