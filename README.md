Perfetto, ti preparo una versione corretta e coerente del file **README.md**: ho tolto i dataset che in realtà non hai usato (HuggingFace, Kaggle, ecc.) e ho lasciato solo quelli veri (blog curati). Inoltre ho reso uniforme lo stile, corretto la ripetizione della sezione *Data Sources* e migliorato la leggibilità.

---

```markdown
# ai-travel-assistant-capstone
AI-based travel assistant for personalized itinerary planning, focused on pet-friendly and road trip travellers.  
Built with a Retrieval-Augmented Generation (RAG) architecture combining NLP and curated travel blog data.  
Capstone project supervised by Prof. Abhinay Pandya, OPIT University.

# AI-Powered Travel Assistant – Capstone Project

This project is part of my Master’s capstone thesis at OPIT University, supervised by Prof. Abhinay Pandya.

It aims to develop an intelligent travel assistant focused on:
- **Pet-friendly travellers** looking for suitable places to stay and visit  
- **Road trip travellers** interested in flexible, on-the-go itineraries  

## Project Summary

The assistant leverages a **Retrieval-Augmented Generation (RAG)** architecture that combines lightweight language models (LLMs) with a retrieval pipeline.  
It provides personalized travel itineraries based on free-text user queries, retrieving relevant content from curated travel blogs.

## Technologies Used

- Python  
- Streamlit  
- FAISS  
- LangChain  
- spaCy / NLTK  
- sentence-transformers  
- Git  

## Project Structure

```

ai-travel-assistant-capstone/
├── data/          # Raw and processed data
├── notebooks/     # Exploratory notebooks and prototyping
├── src/           # Core modules (RAG pipeline, NLP, utils)
├── interface/     # Streamlit app or CLI interface
├── docs/          # Diagrams, report material
├── tests/         # Unit and integration tests
├── requirements.txt
└── README.md

````

## Data Sources

The knowledge base was developed using more than 500 curated travel blog articles from:  
- [Along Dusty Roads](https://www.alongdustyroads.com) – curated travel guides and stories  
- [The Culture Trip](https://theculturetrip.com) – cultural insights and travel tips  
- [Earth Trekkers](https://www.earthtrekkers.com) – family travel itineraries and recommendations  

These blogs were scraped, cleaned, and enriched (NER, KeyBERT, clustering) to form the Retrieval-Augmented Generation (RAG) knowledge base.

## How to Run

```bash
# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate   # On Windows use venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the assistant
streamlit run interface/app.py
````

## 🧪 Running Tests

You can run the automated tests using `pytest`:

```bash
pytest
```

Other useful commands:

```bash
# Run retrieval-augmented interface (with FAISS)
python src/debug/terminal_interface.py

# Run GPT-only response (without document context)
python tests/test_vs_gpt_simple.py

# Compare GPT responses against golden references
pytest tests/test_golden_comparison.py --accept-new-golden
```

## 🛠 Mock Tool Agents

To simulate external information, mock agents are included in
`src/agents/tool_wrappers.py`:

* 🗺️ `mock_google_maps_route()` – travel duration simulation
* 🍽️ `mock_restaurant_recommendation()` – cuisine-based suggestions
* 🏨 `mock_hotel_suggestions()` – pet-friendly hotels

These are injected dynamically into prompts to simulate grounded assistant behavior.

## Timeline

* **May–June**: RAG pipeline, NLP components, document indexing
* **July**: Interface design, itinerary generation, testing
* **August**: Documentation, final report, and presentation


```
