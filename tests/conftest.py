# tests/conftest.py
def pytest_addoption(parser):
    parser.addoption(
        "--accept-new-golden", action="store_true", default=False,
        help="Automatically accept new golden responses and overwrite existing ones."
    )

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "golden: mark test as comparing with golden output"
    )
