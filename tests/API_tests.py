from config import (
    COHERE_API_KEY,
    COLLECTION_NAME,
    COHERE_MODEL,
    QDRANT_API_KEY,
    QDRANT_URL,
    TEMP_DIR,
)


def test_configuration_defaults_are_available():
    assert TEMP_DIR == "temp_assets"
    assert COLLECTION_NAME == "video_knowledge"
    assert COHERE_MODEL == "command-a-03-2025"


def test_environment_values_are_strings_or_none():
    assert QDRANT_URL is None or isinstance(QDRANT_URL, str)
    assert QDRANT_API_KEY is None or isinstance(QDRANT_API_KEY, str)
    assert COHERE_API_KEY is None or isinstance(COHERE_API_KEY, str)
