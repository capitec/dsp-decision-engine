import pytest


# auto-mode so every async def test_* is treated as asyncio without decoration
def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: mark test as async")
