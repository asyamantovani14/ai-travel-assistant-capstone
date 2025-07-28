import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter, defaultdict
import os

# Download NLTK data
nltk.download("punkt")
nltk.download("stopwords")

# === CONFIG ===
INPUT_FILE = "data/blogs/clean_articles.json"
OUTPUT_FILE = "data/knowledge_base/family_travel_knowledge.json"
N_CLUSTERS = 3  # verrà ridotto se necessario

# === FUNZIONI ===

def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text.lower())
    return " ".join([w for w in words if w.isalpha() and w not in stop_words])

def extract_practical_tips(text):
    lines = text.split(".")
    keywords = ["tip", "recommend", "suggest", "advice", "important", "note"]
    return [line.strip() for line in lines if any(k in line.lower() for k in keywords)]

def extract_tags(text, top_n=5):
    words = word_tokenize(text.lower())
    words = [w for w in words if w.isalpha()]
    stop_words = set(stopwords.words("english"))
    filtered = [w for w in words if w not in stop_words]
    return [word for word, _ in Counter(filtered).most_common(top_n)]

def describe_clusters(texts, labels, top_n=5):
    """Genera descrizione testuale per ogni cluster basata sui termini più frequenti"""
    cluster_terms = defaultdict(list)
    for text, label in zip(texts, labels):
        cluster_terms[label].extend(text.split())

    descriptions = {}
    stop_words = set(stopwords.words("english"))
    for label, terms in cluster_terms.items():
        filtered = [t for t in terms if t not in stop_words]
        most_common = [w for w, _ in Counter(filtered).most_common(top_n)]
        descriptions[int(label)] = f"Topics: {', '.join(most_common)}"
    return descriptions

# === CARICAMENTO ARTICOLI ===

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"❌ File not found: {INPUT_FILE}")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    articles = json.load(f)

valid_articles = [a for a in articles if "clean_text" in a and a["clean_text"].strip()]
print(f"✅ Loaded {len(valid_articles)} valid articles.")

if len(valid_articles) == 0:
    raise ValueError("❌ No valid articles with 'clean_text'. Check your clean_articles.json file.")

# === PREPROCESSING ===

processed_texts = [preprocess_text(a["clean_text"]) for a in valid_articles]

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(processed_texts)

# === KMEANS CLUSTERING ===

N_CLUSTERS = min(N_CLUSTERS, len(valid_articles))
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
labels = kmeans.fit_predict(X)

# === DESCRIZIONE CLUSTER ===

cluster_descriptions = describe_clusters(processed_texts, labels)

# === COSTRUZIONE KNOWLEDGE BASE ===

knowledge_base = []
for i, article in enumerate(valid_articles):
    tips = extract_practical_tips(article["clean_text"])
    tags = extract_tags(article["clean_text"])
    knowledge_base.append({
        "title": article["title"],
        "url": article["url"],
        "cluster": int(labels[i]),
        "cluster_description": cluster_descriptions[int(labels[i])],
        "tips": tips,
        "tags": tags,
    })

# === SALVATAGGIO ===

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(knowledge_base, f, indent=2, ensure_ascii=False)

print(f"✅ Knowledge base with cluster descriptions saved to {OUTPUT_FILE}")
