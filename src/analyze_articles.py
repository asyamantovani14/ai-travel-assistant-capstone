import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import string

# File paths
INPUT_FILE = "data/blogs/clean_articles.json"

# Load stopwords
stop_words = set(stopwords.words("english"))

# Load articles
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    articles = json.load(f)

# Combine all text into one
all_text = " ".join([article["clean_text"] for article in articles])

# Tokenize and clean
tokens = word_tokenize(all_text.lower())
tokens = [t for t in tokens if t.isalpha() and t not in stop_words and t not in string.punctuation]

# Count word frequency
word_freq = Counter(tokens)

# Top 20 words
print("🔍 Top 20 most common words:")
for word, freq in word_freq.most_common(20):
    print(f"{word}: {freq}")
