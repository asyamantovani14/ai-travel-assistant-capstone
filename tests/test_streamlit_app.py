from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "interface" / "app.py"


def test_streamlit_app_loads_the_chat_experience():
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()

    assert not app.exception
    assert app.title[0].value == "Travel Assistant"
    assert app.chat_input[0].placeholder == "Plan a trip or refine your itinerary"
    assert app.sidebar.header[0].value == "Trip preferences"
