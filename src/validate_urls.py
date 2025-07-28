import os
import json
import requests

ALL_URLS_FILE = "data/blogs/all_urls.json"
VALID_URLS_FILE = "data/blogs/valid_urls.json"

def is_url_valid(url):
    try:
        response = requests.head(url, timeout=5, allow_redirects=True)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error for {url}: {e}")
        return False

def filter_valid_urls(url_list):
    return [url for url in url_list if is_url_valid(url)]

if __name__ == "__main__":
    if not os.path.exists(ALL_URLS_FILE):
        print(f"❌ File not found: {ALL_URLS_FILE}")
        exit(1)

    with open(ALL_URLS_FILE, "r", encoding="utf-8") as f:
        all_urls = json.load(f)

    print(f"🔎 Checking {len(all_urls)} URLs...")
    valid_urls = filter_valid_urls(all_urls)

    print(f"✅ Found {len(valid_urls)} valid URLs.")

    os.makedirs(os.path.dirname(VALID_URLS_FILE), exist_ok=True)
    with open(VALID_URLS_FILE, "w", encoding="utf-8") as f:
        json.dump(valid_urls, f, indent=2, ensure_ascii=False)
