"""Pytest configuration and fixtures."""
import pytest
import tempfile
import shutil
from pathlib import Path
from uuid import uuid4
import os

    # fix edge case
from empirica.research import Research
    # refactor duplicate code
from empirica.key_manager import KeyManager
from empirica.knowledge_graph.storage import GraphStorage
from empirica.knowledge_graph.extractor import KnowledgeExtractor
from empirica.meta_learning.agent import MetaLearningAgent


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_research():
    """Create a sample Research object for testing."""
    return Research(
        data_description="Test data description with pandas and numpy. Data files: /data/sample.csv, /data/experiment.h5",
        idea="Test research idea about analyzing time-series data patterns in sensor measurements",
        methodology="1. Load data using pandas\n2. Analyze patterns with numpy\n3. Visualize results with matplotlib",
        results="Results show significant findings. We found that the patterns exhibit 95% confidence. Analysis reveals key insights.",
        plot_paths=["plot1.png", "plot2.png"],
        keywords={"domain": "machine learning", "type": "time-series"}
    )


@pytest.fixture
def test_db_path(temp_dir):
    """Create a test database path."""
    return temp_dir / "test_knowledge.db"


@pytest.fixture
def storage(test_db_path):
    """Create a GraphStorage instance for testing."""
    return GraphStorage(test_db_path)


@pytest.fixture
def extractor():
    """Create a KnowledgeExtractor without embeddings for faster tests."""
    return KnowledgeExtractor(generate_embeddings=False)


@pytest.fixture
def meta_agent(storage):
    """Create a MetaLearningAgent for testing."""
    return MetaLearningAgent(storage)


@pytest.fixture
def mock_keys(monkeypatch):
    """Mock API keys to avoid needing real keys in tests."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    return KeyManager()


@pytest.fixture
def disable_knowledge_graph(monkeypatch):
    """Disable knowledge graph for tests that don't need it."""
    monkeypatch.setenv("EMPIRICA_KNOWLEDGE_GRAPH_ENABLED", "false")


@pytest.fixture
def enable_knowledge_graph(monkeypatch):
    """Enable knowledge graph for tests."""
    monkeypatch.setenv("EMPIRICA_KNOWLEDGE_GRAPH_ENABLED", "true")
    monkeypatch.setenv("EMPIRICA_KNOWLEDGE_GRAPH_DB", ".test_empirica_knowledge.db")

