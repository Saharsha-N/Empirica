"""End-to-end tests for full workflow."""
import pytest
import os
from pathlib import Path

try:
    from empirica.empirica import Empirica
    EMPIRICA_AVAILABLE = True
except ImportError:
    EMPIRICA_AVAILABLE = False


@pytest.mark.e2e
@pytest.mark.skipif(not EMPIRICA_AVAILABLE, reason="cmbagent not available")
def test_workflow_without_knowledge_graph(temp_dir):
    """Test complete workflow without knowledge graph."""
    os.environ["EMPIRICA_KNOWLEDGE_GRAPH_ENABLED"] = "false"
    
    emp = Empirica(project_dir=str(temp_dir / "project"))
    
    # Should work without knowledge graph
    assert emp.knowledge_graph_enabled is False
    emp.set_data_description("Test data description")
    
    # Workflow should proceed normally
    assert emp.research.data_description == "Test data description"


@pytest.mark.e2e
@pytest.mark.skipif(not EMPIRICA_AVAILABLE, reason="cmbagent not available")
def test_workflow_with_knowledge_graph(temp_dir):
    """Test complete workflow with knowledge graph enabled."""
    os.environ["EMPIRICA_KNOWLEDGE_GRAPH_ENABLED"] = "true"
    os.environ["EMPIRICA_KNOWLEDGE_GRAPH_DB"] = str(temp_dir / "test.db")
    
    emp = Empirica(project_dir=str(temp_dir / "project"))
    
    assert emp.knowledge_graph_enabled is True
    emp.set_data_description("Test data")
    emp.research.idea = "Test idea"
    emp.research.methodology = "Test method"
    emp.research.results = "Test results"
    
    # Store to knowledge graph
    emp.store_to_knowledge_graph(success=True)
    
    # Verify storage
    project_ids = emp.storage.list_projects()
    assert len(project_ids) > 0

