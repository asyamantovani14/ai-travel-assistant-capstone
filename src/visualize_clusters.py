import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import streamlit as st

# Load and preprocess NLTK
nltk.download("punkt")
nltk.download("stopwords")
stop = set(stopwords.words("english"))

def preprocess(text):
    return " ".join(w for w in word_tokenize(text.lower())
                    if w.isalpha() and w not in stop)

# Streamlit UI
st.title("📊 Article Cluster Visualization")

# Load your cleaned articles
with open("data/blogs/clean_articles.json","r",encoding="utf-8") as f:
    arts = json.load(f)

texts  = [a["clean_text"] for a in arts]
titles = [a["title"]       for a in arts]
proc   = [preprocess(t)     for t in texts]

# TF-IDF + KMeans
X          = TfidfVectorizer(max_features=500).fit_transform(proc)
n_clusters = min(3, len(proc))
labels     = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(X)

# PCA → 2D
coords = PCA(n_components=2, random_state=42).fit_transform(X.toarray())

# Plot
fig, ax = plt.subplots(figsize=(6,4))
for c in range(n_clusters):
    pts = coords[labels==c]
    ax.scatter(pts[:,0], pts[:,1], label=f"Cluster {c}", alpha=0.7)
ax.set_title("Article Clusters (PCA 2D)")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.legend()
st.pyplot(fig)
