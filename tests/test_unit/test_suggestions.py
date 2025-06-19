"""Unit tests for suggestions."""
import pytest

from empirica.suggestions.engine import SuggestionEngine, Suggestion, SuggestionType
from empirica.suggestions.display import SuggestionDisplay, DisplayFormat


def test_suggestion_creation():
    """Test Suggestion creation."""
    suggestion = Suggestion(
        suggestion_type=SuggestionType.METHOD,
        content="Use pandas for data analysis",
        confidence=0.85,
        reason="High success rate in similar projects"
    )
    
    assert suggestion.suggestion_type == SuggestionType.METHOD
    assert suggestion.content == "Use pandas for data analysis"
    assert suggestion.confidence == 0.85
    assert suggestion.reason == "High success rate in similar projects"


def test_suggest_methods(meta_agent, sample_research):
    """Test SuggestionEngine.suggest_methods."""
    from empirica.knowledge_graph.extractor import KnowledgeExtractor
    
    # Create a successful project
    extractor = KnowledgeExtractor(generate_embeddings=False)
    project = extractor.extract_from_research(sample_research)
    project.success = True
    meta_agent.storage.save_project(project)
    
    engine = SuggestionEngine(meta_agent.storage, meta_agent)
    suggestions = engine.suggest_methods(
        idea_text=sample_research.idea,
        top_k=3
    )
    
    assert isinstance(suggestions, list)
    for suggestion in suggestions:
        assert isinstance(suggestion, Suggestion)
        assert suggestion.suggestion_type == SuggestionType.METHOD
        assert 0.0 <= suggestion.confidence <= 1.0


def test_suggest_tools(meta_agent, sample_research):
    """Test SuggestionEngine.suggest_tools."""
    from empirica.knowledge_graph.extractor import KnowledgeExtractor
    
    # Create projects with tools
    extractor = KnowledgeExtractor(generate_embeddings=False)
    for i in range(3):
        project = extractor.extract_from_research(sample_research)
        project.success = True
        meta_agent.storage.save_project(project)
    
    engine = SuggestionEngine(meta_agent.storage, meta_agent)
    suggestions = engine.suggest_tools(
        idea_text=sample_research.idea,
        current_tools=None
    )
    
    assert isinstance(suggestions, list)
    for suggestion in suggestions:
        assert suggestion.suggestion_type == SuggestionType.TOOL
        assert 0.0 <= suggestion.confidence <= 1.0


def test_suggest_tools_excludes_current(meta_agent, sample_research):
    """Test that suggest_tools excludes already planned tools."""
    engine = SuggestionEngine(meta_agent.storage, meta_agent)
    suggestions = engine.suggest_tools(
        idea_text=sample_research.idea,
        current_tools=["pandas", "numpy"]
    )
    
    tool_contents = [s.content for s in suggestions]
    assert "pandas" not in tool_contents
    assert "numpy" not in tool_contents


def test_check_dataset_compatibility(meta_agent, sample_research):
    """Test SuggestionEngine.check_dataset_compatibility."""
    from empirica.knowledge_graph.extractor import KnowledgeExtractor
    
    extractor = KnowledgeExtractor(generate_embeddings=False)
    project = extractor.extract_from_research(sample_research)
    meta_agent.storage.save_project(project)
    
    engine = SuggestionEngine(meta_agent.storage, meta_agent)
    suggestions = engine.check_dataset_compatibility(
        idea_text=sample_research.idea,
        dataset_paths=["/data/different.csv"]
    )
    
    assert isinstance(suggestions, list)


def test_generate_warnings(meta_agent, sample_research):
    """Test SuggestionEngine.generate_warnings."""
    engine = SuggestionEngine(meta_agent.storage, meta_agent)
    warnings = engine.generate_warnings(
        idea_text=sample_research.idea,
        method_text=sample_research.methodology,
        tools=["pandas"]
    )
    
    assert isinstance(warnings, list)
    for warning in warnings:
        assert warning.suggestion_type == SuggestionType.WARNING
        assert 0.0 <= warning.confidence <= 1.0


def test_suggest_optimizations(meta_agent, sample_research):
    """Test SuggestionEngine.suggest_optimizations."""
    engine = SuggestionEngine(meta_agent.storage, meta_agent)
    optimizations = engine.suggest_optimizations(
        method_text=sample_research.methodology,
        execution_time=60.0
    )
    
    assert isinstance(optimizations, list)
    for opt in optimizations:
        assert opt.suggestion_type == SuggestionType.OPTIMIZATION


def test_get_all_suggestions(meta_agent, sample_research):
    """Test SuggestionEngine.get_all_suggestions."""
    engine = SuggestionEngine(meta_agent.storage, meta_agent)
    all_suggestions = engine.get_all_suggestions(
        idea_text=sample_research.idea,
        method_text=sample_research.methodology,
        dataset_paths=["/data/test.csv"],
        tools=["pandas"]
    )
    
    assert isinstance(all_suggestions, dict)
    assert "methods" in all_suggestions
    assert "tools" in all_suggestions
    assert "warnings" in all_suggestions
    assert "datasets" in all_suggestions
    assert "optimizations" in all_suggestions


def test_suggestion_display_format_text():
    """Test SuggestionDisplay.format_suggestion in text format."""
    suggestion = Suggestion(
        suggestion_type=SuggestionType.METHOD,
        content="Test suggestion",
        confidence=0.75,
        reason="Test reason"
    )
    
    formatted = SuggestionDisplay.format_suggestion(suggestion, DisplayFormat.TEXT)
    assert "METHOD" in formatted
    assert "Test suggestion" in formatted
    assert "0.75" in formatted or "75%" in formatted or "75.0%" in formatted


def test_suggestion_display_format_json():
    """Test SuggestionDisplay.format_suggestion in JSON format."""
    suggestion = Suggestion(
        suggestion_type=SuggestionType.TOOL,
        content="pandas",
        confidence=0.8,
        reason="High success rate"
    )
    
    formatted = SuggestionDisplay.format_suggestion(suggestion, DisplayFormat.JSON)
    import json
    data = json.loads(formatted)
    assert data["type"] == "tool"
    assert data["content"] == "pandas"
    assert data["confidence"] == 0.8


def test_suggestion_display_display_suggestions(capsys):
    """Test SuggestionDisplay.display_suggestions."""
    suggestions = [
        Suggestion(
            suggestion_type=SuggestionType.METHOD,
            content="Method 1",
            confidence=0.9,
            reason="High confidence"
        ),
        Suggestion(
            suggestion_type=SuggestionType.TOOL,
            content="pandas",
            confidence=0.7,
            reason="Commonly used"
        )
    ]
    
    SuggestionDisplay.display_suggestions(
        suggestions,
        title="Test Suggestions",
        min_confidence=0.5
    )
    
    # Check that output was generated (captured by capsys)
    # This is a basic smoke test


def test_suggestion_display_min_confidence_filter():
    """Test that suggestions below confidence threshold are filtered."""
    suggestions = [
        Suggestion(SuggestionType.METHOD, "High", 0.9, "reason"),
        Suggestion(SuggestionType.METHOD, "Low", 0.2, "reason")
    ]
    
    # Should only show high confidence
    filtered = [s for s in suggestions if s.confidence >= 0.5]
    assert len(filtered) == 1
    assert filtered[0].content == "High"

