import json
import re
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import download

# Make sure NLTK resources are downloaded
download("punkt")
download("stopwords")

INPUT_FILE = "data/blogs/scraped_articles.json"
OUTPUT_FILE = "data/blogs/clean_articles.json"

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # Remove punctuation and digits
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    filtered = [word for word in tokens if word not in stopwords.words("english")]
    return " ".join(filtered)

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"File not found: {INPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        articles = json.load(f)

    cleaned_articles = []
    for article in articles:
        cleaned_text = clean_text(article.get("text", ""))
        cleaned_articles.append({
            "title": article.get("title", ""),
            "url": article.get("url", ""),
            "clean_text": cleaned_text
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned_articles, f, indent=2)

    print(f"✅ Cleaned {len(cleaned_articles)} articles and saved to '{OUTPUT_FILE}'")
