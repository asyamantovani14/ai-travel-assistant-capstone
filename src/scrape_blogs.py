import json
import os
from newspaper import Article
from tqdm import tqdm
from datetime import datetime

# File paths
VALID_URLS_FILE = "data/blogs/valid_urls.json"
OUTPUT_FILE = "data/blogs/scraped_articles.json"
FAILED_LOG = "data/blogs/failed_articles.json"

# Create the output directory if it doesn't exist
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# Load valid URLs
with open(VALID_URLS_FILE, "r", encoding="utf-8") as f:
    urls = json.load(f)

# Lists to store results
articles_data = []
failed_urls = []

print(f"📄 Starting scraping of {len(urls)} articles...")

for url in tqdm(urls):
    try:
        article = Article(url)
        article.download()
        article.parse()

        if not article.text.strip():
            raise ValueError("Empty article text.")

        articles_data.append({
            "url": url,
            "title": article.title,
            "text": article.text,
            "authors": article.authors,
            "publish_date": article.publish_date.isoformat() if article.publish_date else None,
            "scraped_at": datetime.utcnow().isoformat()
        })

    except Exception as e:
        print(f"⚠️  Failed to parse {url}: {e}")
        failed_urls.append({"url": url, "error": str(e)})

# Save the successfully scraped articles
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(articles_data, f, ensure_ascii=False, indent=2)

# Save the failed URLs for review
with open(FAILED_LOG, "w", encoding="utf-8") as f:
    json.dump(failed_urls, f, ensure_ascii=False, indent=2)

print(f"✅ Scraping finished: {len(articles_data)} articles saved.")
print(f"❌ Failed to scrape: {len(failed_urls)} (see {FAILED_LOG})")
