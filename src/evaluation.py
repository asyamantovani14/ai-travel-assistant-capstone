print("👋 Starting evaluation script...")

import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import nltk

# Ensure NLTK data is available
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# === CONFIGURATION ===
CLEAN_FILE = 'data/blogs/clean_articles.json'
ENRICHED_FILE = 'data/knowledge_base/family_travel_knowledge.json'
FEEDBACK_CSV = 'logs/feedback.csv'

# === 1. Evaluate Clustering Quality ===
def evaluate_clustering(clean_file, enriched_file):
    print("\n>> Evaluating Clustering Quality")

    # Load cleaned articles
    if not os.path.exists(clean_file):
        print(f"❌ Clean file not found: {clean_file}")
        return None
    with open(clean_file, 'r', encoding='utf-8') as f:
        clean_articles = json.load(f)
    print(f"🔍 {len(clean_articles)} clean articles loaded")

    # Load enriched KB entries
    if not os.path.exists(enriched_file):
        print(f"❌ Enriched file not found: {enriched_file}")
        return None
    with open(enriched_file, 'r', encoding='utf-8') as f:
        enriched = json.load(f)
    print(f"🔍 {len(enriched)} enriched KB entries loaded")

    # Map titles to clean_text
    clean_map = {a['title']: a['clean_text'] for a in clean_articles if 'clean_text' in a}
    texts, labels = [], []
    for entry in enriched:
        title = entry.get('title')
        cluster = entry.get('cluster')
        text = clean_map.get(title)
        if text and cluster is not None:
            texts.append(text)
            labels.append(cluster)
    print(f"🔍 {len(texts)} articles matched for clustering evaluation with labels")

    n_samples = len(texts)
    n_clusters = len(set(labels))
    if n_samples < 2 or n_clusters < 2:
        print("⚠️ Not enough data or clusters to compute metrics.")
        return None

    # Preprocess texts
    stop = set(stopwords.words('english'))
    def prep(t):
        return ' '.join(w for w in word_tokenize(t.lower()) if w.isalpha() and w not in stop)
    proc = [prep(t) for t in texts]

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(proc)

    # Compute metrics with guards
    sil = None
    if n_samples >= 3:
        sil = silhouette_score(X, labels)
    ch = None
    if n_samples > n_clusters:
        try:
            ch = calinski_harabasz_score(X.toarray(), labels)
        except ValueError:
            ch = None
    db = None
    if n_samples > n_clusters:
        try:
            db = davies_bouldin_score(X.toarray(), labels)
        except ValueError:
            db = None

    print(f"🎯 Silhouette Score: {sil:.3f}" if sil is not None else "🎯 Silhouette: N/A (need ≥3 samples)")
    print(f"🏆 Calinski–Harabasz Index: {ch:.3f}" if ch is not None else "🏆 Calinski–Harabasz: N/A (need n_samples > n_clusters)")
    print(f"🔄 Davies–Bouldin Index: {db:.3f}" if db is not None else "🔄 Davies–Bouldin: N/A (need n_samples > n_clusters)")

    return {
        'silhouette': sil,
        'calinski_harabasz': ch,
        'davies_bouldin': db
    }

# === 2. Summarize User Feedback with Sentiment ===
def summarize_feedback(feedback_csv):
    print("\n>> Summarizing User Feedback")
    if not os.path.exists(feedback_csv):
        print("ℹ️ No feedback log found.")
        return None
    df = pd.read_csv(feedback_csv)

    # Determine feedback column
    feedback_col = 'choice' if 'choice' in df.columns else 'preferred' if 'preferred' in df.columns else None
    if not feedback_col:
        print(f"⚠️ No 'choice'/'preferred' column. Columns: {df.columns.tolist()}")
        return None

    counts = df[feedback_col].value_counts(normalize=True).mul(100).round(1)
    print("\n=== User Feedback Summary ===")
    for label, pct in counts.items():
        print(f"- {label}: {pct}%")

    # Sentiment analysis on notes
    if 'notes' in df.columns:
        sia = SentimentIntensityAnalyzer()
        notes = df['notes'].dropna().astype(str)
        if not notes.empty:
            scores = notes.apply(lambda txt: sia.polarity_scores(txt)['compound'])
            avg = scores.mean()
            print(f"😊 Average feedback sentiment (compound): {avg:.3f}")

    return counts.to_dict()

# === MAIN ===
if __name__ == '__main__':
    metrics = evaluate_clustering(CLEAN_FILE, ENRICHED_FILE)
    feedback = summarize_feedback(FEEDBACK_CSV)

    # Save metrics report
    report = {'metrics': metrics, 'feedback': feedback}
    os.makedirs('data', exist_ok=True)
    with open('data/metrics_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print("\n✅ Metrics report saved to data/metrics_report.json")
