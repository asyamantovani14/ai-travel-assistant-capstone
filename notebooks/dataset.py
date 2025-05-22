from datasets import load_dataset

dataset = load_dataset("osunlp/TravelPlanner")
dataset["train"][0]

dataset["train"].features
dataset["train"].column_names
dataset["train"].shuffle(seed=42).select(range(3))
documents = [entry["trip_description"] for entry in dataset["train"] if entry["trip_description"]]
