"""Integration tests for reproducibility."""
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
def test_reproduce_project_creates_valid_research(temp_dir, sample_research):
    """Test reproduce_project creates valid Research object."""
    os.environ["EMPIRICA_KNOWLEDGE_GRAPH_ENABLED"] = "true"
    os.environ["EMPIRICA_KNOWLEDGE_GRAPH_DB"] = str(temp_dir / "test.db")
    
    emp = Empirica(project_dir=str(temp_dir / "project"))
    emp.research = sample_research
    emp.store_to_knowledge_graph(success=True)
    
    # Get project ID
    project_ids = emp.storage.list_projects()
    assert len(project_ids) > 0
    project_id = project_ids[0]
    
    # Reproduce
    reproduced = emp.reproduce_project(str(project_id))
    
    assert isinstance(reproduced, Research)
    assert reproduced.idea == sample_research.idea
    assert reproduced.methodology == sample_research.methodology
    assert reproduced.results == sample_research.results


@pytest.mark.integration
@pytest.mark.skipif(not EMPIRICA_AVAILABLE, reason="cmbagent not available")
def test_validate_reproducibility(temp_dir, sample_research):
    """Test validate_reproducibility."""
    os.environ["EMPIRICA_KNOWLEDGE_GRAPH_ENABLED"] = "true"
    os.environ["EMPIRICA_KNOWLEDGE_GRAPH_DB"] = str(temp_dir / "test.db")
    
    emp = Empirica(project_dir=str(temp_dir / "project"))
    emp.research = sample_research
    emp.store_to_knowledge_graph(success=True)
    
    project_ids = emp.storage.list_projects()
    project_id = project_ids[0]
    
    if emp.reproducibility_engine:
        reproduced = emp.reproduce_project(str(project_id))
        validation = emp.reproducibility_engine.validate_reproducibility(
            project_id,
            reproduced
        )
        
        assert "validated" in validation
        assert "checks" in validation


@pytest.mark.integration
@pytest.mark.skipif(not EMPIRICA_AVAILABLE, reason="cmbagent not available")
def test_generate_reproducibility_report(temp_dir, sample_research):
    """Test generate_reproducibility_report."""
    os.environ["EMPIRICA_KNOWLEDGE_GRAPH_ENABLED"] = "true"
    os.environ["EMPIRICA_KNOWLEDGE_GRAPH_DB"] = str(temp_dir / "test.db")
    
    emp = Empirica(project_dir=str(temp_dir / "project"))
    emp.research = sample_research
    emp.store_to_knowledge_graph(success=True)
    
    project_ids = emp.storage.list_projects()
    project_id = project_ids[0]
    
    if emp.reproducibility_engine:
        report = emp.reproducibility_engine.generate_reproducibility_report(project_id)
        
        assert "project_id" in report
        assert "components" in report
        assert "reproducibility_status" in report

