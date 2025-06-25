# Empirica Test Suite

This directory contains comprehensive tests for the Empirica codebase, including unit tests, integration tests, and end-to-end tests.

## Test Structure

```
tests/
├── conftest.py                    # Pytest fixtures and configuration
├── fixtures/                      # Test data fixtures
│   └── sample_research.json      # Sample Research object data
├── test_unit/                    # Unit tests
│   ├── test_models.py           # Knowledge graph models
│   ├── test_storage.py          # Storage backend
│   ├── test_extractor.py        # Knowledge extraction
│   ├── test_query.py            # Graph queries
│   ├── test_analyzer.py          # Pattern analysis
│   ├── test_meta_models.py      # Meta-learning models
│   ├── test_versioning.py      # Version control
│   ├── test_checkpoints.py      # Checkpoint manager
│   ├── test_provenance.py      # Provenance tracking
│   └── test_suggestions.py      # Suggestion engine
├── test_integration/             # Integration tests
│   ├── test_knowledge_graph_integration.py
│   ├── test_suggestions_integration.py
│   └── test_reproducibility_integration.py
└── test_e2e/                     # End-to-end tests
    ├── test_full_workflow.py
    └── test_knowledge_graph_workflow.py
```

## Running Tests

### Install Test Dependencies

```bash
pip install -e ".[test]"
```

### Run All Tests

```bash
pytest tests/
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/test_unit/

# Integration tests only
pytest tests/test_integration/ -m integration

# End-to-end tests only
pytest tests/test_e2e/ -m e2e
```

### Run with Coverage

```bash
pytest tests/ --cov=empirica --cov-report=html
```

### Run Specific Test File

```bash
pytest tests/test_unit/test_models.py
```

### Run with Verbose Output

```bash
pytest tests/ -v
```

### Run Tests Matching Pattern

```bash
pytest tests/ -k "test_storage"
```

## Test Markers

Tests are marked with pytest markers:

- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.slow` - Slow-running tests (if any)

## Test Fixtures

Common fixtures available in `conftest.py`:

- `temp_dir` - Temporary directory for test files
- `sample_research` - Sample Research object
- `storage` - GraphStorage instance
- `extractor` - KnowledgeExtractor instance
- `meta_agent` - MetaLearningAgent instance
- `mock_keys` - Mocked API keys
- `enable_knowledge_graph` - Enable KG for tests
- `disable_knowledge_graph` - Disable KG for tests

## Notes

- Tests use temporary directories that are cleaned up automatically
- Test databases are created in temporary locations
- Integration tests may require knowledge graph to be enabled
- End-to-end tests may require API keys (can be skipped if not available)

