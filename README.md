# Atlas Travel Assistant

Atlas is a conversational travel planner built as an OPIT University capstone
project. It combines a curated travel knowledge base, semantic retrieval, live
provider data, and an LLM to produce itineraries that can be refined over a
multi-turn conversation.

## Product Features

- Persistent travel-planning conversations in the browser
- Retrieval-augmented answers backed by local travel documents
- Country, activity, duration, and budget preferences
- Live weather through Open-Meteo
- Optional Google Maps, Yelp, and Geoapify integrations
- Inspectable source relevance and extracted trip details
- Responsive desktop and mobile interface
- Markdown itinerary export
- Automated knowledge-base refresh pipeline

## Technology

- FastAPI and Uvicorn
- HTML, CSS, and JavaScript frontend with no build step
- OpenAI API
- FAISS and sentence-transformers
- spaCy and NLTK
- Pytest and Playwright

## Run Locally

Create and activate a virtual environment:

```bash
python -m venv venv
```

On Windows:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn web_app:app --app-dir src --reload --port 8000
```

On macOS or Linux:

```bash
source venv/bin/activate
pip install -r requirements.txt
python -m uvicorn web_app:app --app-dir src --reload --port 8000
```

Open `http://127.0.0.1:8000`.

## Configuration

Create a local `.env` file. Only `OPENAI_API_KEY` is required for generated
answers. The remaining providers are optional.

```dotenv
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4.1-mini
GOOGLE_MAPS_API_KEY=
YELP_API_KEY=
GEOAPIFY_API_KEY=
```

When a live provider is unavailable, Atlas reports that explicitly instead of
inventing current routes, businesses, or prices.

## Tests

Install development dependencies and run the suite:

```bash
pip install -r requirements-dev.txt
pytest
```

Capture and validate desktop/mobile layouts while the server is running:

```bash
python scripts/capture_ui.py
```

Live OpenAI golden comparisons are opt-in:

```powershell
$env:RUN_LIVE_TESTS = "1"
pytest tests/test_golden_comparison.py
```

## Knowledge Refresh

Preview the refresh pipeline:

```bash
python src/run_daily_pipeline.py --dry-run
```

Run the complete refresh:

```bash
python src/run_daily_pipeline.py
```

Downloaded blog mirrors and user interaction logs remain local and are ignored
by Git. See `data/README.md` for the data policy and cleanup workflow.

## Legacy Interface

The previous Streamlit interface remains available during the migration:

```bash
streamlit run interface/app.py
```
