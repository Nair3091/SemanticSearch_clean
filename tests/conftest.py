import pytest

from src.track1_query_understanding import decomposer
from src.track2_vision_retrieval import retriever


@pytest.fixture(autouse=True)
def fresh_caches():
    """Searches cache frame embeddings and parsed queries; start every test without them."""
    retriever.clear_caches()
    decomposer.clear_cache()
    yield
    retriever.clear_caches()
    decomposer.clear_cache()
