# src/data_loader/multi_loader.py

import pandas as pd
import json
import os

def load_kaggle_trips(path):
    df = pd.read_csv(path)
    return [f"{row['location']} - {row['description']}" for _, row in df.iterrows() if 'location' in row and 'description' in row]

def load_arukikata(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [entry["text"] for entry in data if "text" in entry]

def load_opentravel(path):
    df = pd.read_csv(path)
    return [f"{row['city']} - {row['description']}" for _, row in df.iterrows() if 'city' in row and 'description' in row]

def load_commoncrawl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def load_all_sources(base_dir="data/sources"):
    sources = []

    for file in os.listdir(base_dir):
        full_path = os.path.join(base_dir, file)

        if file.endswith("kaggle_trips.csv"):
            sources.extend(load_kaggle_trips(full_path))
        elif file.endswith("arukikata.json"):
            sources.extend(load_arukikata(full_path))
        elif file.endswith("open_travel_data.csv"):
            sources.extend(load_opentravel(full_path))
        elif file.endswith(".txt"):
            sources.extend(load_commoncrawl(full_path))

    return sources

