import pickle

# Load the extracted entities
with open("data/index/entities.pkl", "rb") as f:
    entities_by_doc = pickle.load(f)

num_docs_to_show = 5

print(f"\n Showing named entities from {num_docs_to_show} documents:\n")

for i, entry in enumerate(entities_by_doc[:num_docs_to_show]):
    print(f"--- Document {i+1} ---")
    print(f"Text: {entry['text']}\n")
    print("Entities:")
    for ent in entry["entities"]:
        print(f" - {ent[0]} ({ent[1]})")  # ent is a tuple (text, label)
    print("\n")
