"""Integration tests for suggestions."""
import pytest
import os
from pathlib import Path

try:
    from empirica.empirica import Empirica
    EMPIRICA_AVAILABLE = True
except ImportError:
    EMPIRICA_AVAILABLE = False

from empirica.research import Research


@pytest.mark.integration
@pytest.mark.skipif(not EMPIRICA_AVAILABLE, reason="cmbagent not available")
def test_suggestions_with_empty_knowledge_graph(temp_dir):
    """Test suggestion integration with empty knowledge graph."""
    os.environ["EMPIRICA_KNOWLEDGE_GRAPH_ENABLED"] = "true"
    os.environ["EMPIRICA_KNOWLEDGE_GRAPH_DB"] = str(temp_dir / "test.db")
    
    emp = Empirica(project_dir=str(temp_dir / "project"))
    emp.research.idea = "Test research idea"
    
    # Should not crash with empty graph
    suggestions = emp.get_suggestions(stage="idea")
    assert isinstance(suggestions, dict)


@pytest.mark.integration
@pytest.mark.skipif(not EMPIRICA_AVAILABLE, reason="cmbagent not available")
def test_suggestions_confidence_thresholds(temp_dir, sample_research):
    """Test suggestion confidence thresholds."""
    os.environ["EMPIRICA_KNOWLEDGE_GRAPH_ENABLED"] = "true"
    os.environ["EMPIRICA_KNOWLEDGE_GRAPH_DB"] = str(temp_dir / "test.db")
    
    emp = Empirica(project_dir=str(temp_dir / "project"))
    emp.research = sample_research
    
    # Store a project first
    emp.store_to_knowledge_graph(success=True)
    
    # Get suggestions
    suggestions = emp.get_suggestions(stage="idea")
    assert isinstance(suggestions, dict)
    
    # If suggestions exist, check confidence scores
    if "similar_projects" in suggestions:
        for project in suggestions["similar_projects"]:
            assert "similarity" in project
            assert 0.0 <= project["similarity"] <= 1.0

