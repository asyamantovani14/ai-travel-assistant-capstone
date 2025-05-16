# ai-travel-assistant-capstone
AI-based travel assistant for personalized itinerary planning, focused on pet-friendly and road trip travellers. Built with a RAG architecture combining NLP and travel data sources. Capstone project supervised by Prof. Abhinay Pandya, OPIT University.

# AI-Powered Travel Assistant – Capstone Project

This project is part of my Bachelor's capstone thesis at OPIT University, supervised by Prof. Abhinay Pandya.

It aims to develop an intelligent travel assistant focused on:
- **Pet-friendly travellers** looking for suitable places to stay and visit  
- **Road trip travellers** interested in flexible, on-the-go itineraries  

## Project Summary

The assistant leverages a **Retrieval-Augmented Generation (RAG)** architecture that combines lightweight language models (LLMs) with a retrieval pipeline. It provides personalized travel itineraries based on free-text user queries, retrieving relevant content from blogs, datasets, or APIs.

## Technologies Used

- Python  
- Streamlit  
- FAISS  
- LangChain  
- spaCy / NLTK  
- sentence-transformers  
- Git  

## Project Structure

ai-travel-assistant-capstone/
├── data/ # Raw and processed data
├── notebooks/ # Exploratory notebooks and prototyping
├── src/ # Core modules (rag_pipeline, NLP, utils)
├── interface/ # Streamlit app or CLI interface
├── docs/ # Diagrams, report material
├── requirements.txt # Dependencies
└── README.md


## Data Sources

- TravelPlanner Dataset – HuggingFace  
- TripCraft Dataset – arXiv  
- Traveler Trip Dataset – Kaggle  
- Arukikata Travelogue Dataset – travel blog narratives  
- Open Travel Data (OPTD) – GitHub  
- Common Crawl (filtered blog content)  
- Public APIs: TripIt, Skyscanner, Amadeus (free tier access)

## How to Run

```bash
# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate   # On Windows use venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the assistant (example)
streamlit run interface/app.py

## Timeline
May–June: RAG pipeline, NLP components, document indexing

July: Interface design, itinerary generation, testing

August: Documentation, final report and presentation

