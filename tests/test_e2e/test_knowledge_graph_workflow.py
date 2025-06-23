"""End-to-end tests for knowledge graph workflow."""
import pytest
import os
from pathlib import Path

try:
    from empirica.empirica import Empirica
    EMPIRICA_AVAILABLE = True
except ImportError:
    EMPIRICA_AVAILABLE = False

from empirica.research import Research


@pytest.mark.e2e
@pytest.mark.skipif(not EMPIRICA_AVAILABLE, reason="cmbagent not available")
def test_full_workflow_storing_to_knowledge_graph(temp_dir):
    """Test full workflow storing to knowledge graph."""
    os.environ["EMPIRICA_KNOWLEDGE_GRAPH_ENABLED"] = "true"
    os.environ["EMPIRICA_KNOWLEDGE_GRAPH_DB"] = str(temp_dir / "test.db")
    
    emp = Empirica(project_dir=str(temp_dir / "project"))
    
    # Simulate workflow steps
    emp.set_data_description("Test data with pandas and numpy")
    emp.research.idea = "Test research idea"
    emp.research.methodology = "1. Load data\n2. Analyze"
    emp.research.results = "Results show significant findings"
    
    # Store to knowledge graph
    emp.store_to_knowledge_graph(success=True, quality_score=0.75)
    
    # Verify
    project_ids = emp.storage.list_projects()
    assert len(project_ids) > 0
    
    project = emp.storage.load_project(project_ids[0])
    assert project.success is True
    assert project.quality_score == 0.75


@pytest.mark.e2e
@pytest.mark.skipif(not EMPIRICA_AVAILABLE, reason="cmbagent not available")
def test_finding_similar_projects_after_multiple_runs(temp_dir):
    """Test finding similar projects after multiple runs."""
    os.environ["EMPIRICA_KNOWLEDGE_GRAPH_ENABLED"] = "true"
    os.environ["EMPIRICA_KNOWLEDGE_GRAPH_DB"] = str(temp_dir / "test.db")
    
    # Create multiple projects
    for i in range(3):
        emp = Empirica(project_dir=str(temp_dir / f"project{i}"))
        emp.set_data_description("Analyze time-series data")
        emp.research.idea = f"Analyze time-series patterns {i}"
        emp.store_to_knowledge_graph(success=True)
    
    # Find similar projects
    emp2 = Empirica(project_dir=str(temp_dir / "project_search"))
    emp2.research.idea = "Analyze time-series patterns 0"
    
    similar = emp2.find_similar_projects(top_k=5)
    assert len(similar) > 0

