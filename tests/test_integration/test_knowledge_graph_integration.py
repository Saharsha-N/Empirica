"""Integration tests for knowledge graph with Empirica."""
import pytest
import tempfile
from pathlib import Path
import os

# Try to import Empirica, skip tests if cmbagent not available
try:
    from empirica.empirica import Empirica
    EMPIRICA_AVAILABLE = True
except ImportError:
    EMPIRICA_AVAILABLE = False

from empirica.research import Research


@pytest.mark.integration
@pytest.mark.skipif(not EMPIRICA_AVAILABLE, reason="cmbagent not available")
def test_empirica_knowledge_graph_initialization(temp_dir):
    """Test that Empirica initializes knowledge graph correctly."""
    os.environ["EMPIRICA_KNOWLEDGE_GRAPH_ENABLED"] = "true"
    os.environ["EMPIRICA_KNOWLEDGE_GRAPH_DB"] = str(temp_dir / "test.db")
    
    emp = Empirica(project_dir=str(temp_dir / "project"))
    
    # Knowledge graph should be enabled
    assert emp.knowledge_graph_enabled is True
    assert emp.storage is not None
    assert emp.extractor is not None
    assert emp.query is not None
    assert emp.meta_agent is not None


@pytest.mark.integration
@pytest.mark.skipif(not EMPIRICA_AVAILABLE, reason="cmbagent not available")
def test_empirica_stores_project_to_knowledge_graph(temp_dir, sample_research):
    """Test that Empirica stores projects to knowledge graph."""
    os.environ["EMPIRICA_KNOWLEDGE_GRAPH_ENABLED"] = "true"
    os.environ["EMPIRICA_KNOWLEDGE_GRAPH_DB"] = str(temp_dir / "test.db")
    
    emp = Empirica(project_dir=str(temp_dir / "project"))
    emp.research = sample_research
    
    # Store to knowledge graph
    emp.store_to_knowledge_graph(success=True, quality_score=0.8)
    
    # Verify project was stored
    project_ids = emp.storage.list_projects()
    assert len(project_ids) > 0
    
    # Load and verify
    project = emp.storage.load_project(project_ids[0])
    assert project is not None
    assert project.success is True
    assert project.quality_score == 0.8


@pytest.mark.integration
@pytest.mark.skipif(not EMPIRICA_AVAILABLE, reason="cmbagent not available")
def test_empirica_find_similar_projects(temp_dir):
    """Test finding similar projects."""
    os.environ["EMPIRICA_KNOWLEDGE_GRAPH_ENABLED"] = "true"
    os.environ["EMPIRICA_KNOWLEDGE_GRAPH_DB"] = str(temp_dir / "test.db")
    
    emp = Empirica(project_dir=str(temp_dir / "project1"))
    
    # Create first project
    emp.set_data_description("Analyze time-series data with pandas")
    emp.research.idea = "Analyze time-series patterns in sensor data"
    emp.store_to_knowledge_graph(success=True)
    
    # Create second project with similar idea
    emp2 = Empirica(project_dir=str(temp_dir / "project2"))
    emp2.research.idea = "Analyze time-series patterns in sensor data"
    
    # Find similar projects
    similar = emp2.find_similar_projects(top_k=5)
    assert len(similar) > 0
    assert all("project_id" in s for s in similar)
    assert all("similarity" in s for s in similar)


@pytest.mark.integration
@pytest.mark.skipif(not EMPIRICA_AVAILABLE, reason="cmbagent not available")
def test_empirica_get_suggestions(temp_dir, sample_research):
    """Test Empirica.get_suggestions at different stages."""
    os.environ["EMPIRICA_KNOWLEDGE_GRAPH_ENABLED"] = "true"
    os.environ["EMPIRICA_KNOWLEDGE_GRAPH_DB"] = str(temp_dir / "test.db")
    
    emp = Empirica(project_dir=str(temp_dir / "project"))
    emp.research = sample_research
    
    # Get suggestions for idea stage
    suggestions = emp.get_suggestions(stage="idea")
    assert isinstance(suggestions, dict)
    
    # Get suggestions for method stage
    suggestions = emp.get_suggestions(stage="method")
    assert isinstance(suggestions, dict)


@pytest.mark.integration
@pytest.mark.skipif(not EMPIRICA_AVAILABLE, reason="cmbagent not available")
def test_version_control_integration(temp_dir, sample_research):
    """Test version control integration in workflow."""
    os.environ["EMPIRICA_KNOWLEDGE_GRAPH_ENABLED"] = "true"
    os.environ["EMPIRICA_KNOWLEDGE_GRAPH_DB"] = str(temp_dir / "test.db")
    
    emp = Empirica(project_dir=str(temp_dir / "project"))
    
    if emp.version_control:
        # Manually set idea to trigger version control
        emp.research.idea = "Test idea"
        emp.version_control.create_version("idea", "Test idea")
        
        versions = emp.version_control.list_versions("idea")
        assert len(versions) > 0


@pytest.mark.integration
@pytest.mark.skipif(not EMPIRICA_AVAILABLE, reason="cmbagent not available")
def test_checkpoint_integration(temp_dir, sample_research):
    """Test checkpoint integration in workflow."""
    os.environ["EMPIRICA_KNOWLEDGE_GRAPH_ENABLED"] = "true"
    os.environ["EMPIRICA_KNOWLEDGE_GRAPH_DB"] = str(temp_dir / "test.db")
    
    emp = Empirica(project_dir=str(temp_dir / "project"))
    emp.research = sample_research
    
    if emp.checkpoint_manager:
        checkpoint_id = emp.checkpoint_manager.auto_checkpoint("idea_generated", emp.research)
        assert checkpoint_id is not None
        
        checkpoint = emp.checkpoint_manager.get_checkpoint(checkpoint_id)
        assert checkpoint is not None
        assert checkpoint.stage == "idea_generated"

