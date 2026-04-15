import pytest

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "live: marks tests that require a built knowledge base"
    )