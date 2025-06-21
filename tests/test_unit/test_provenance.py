"""Unit tests for provenance tracking."""
import pytest
from pathlib import Path
from datetime import datetime

from empirica.reproducibility.provenance import ProvenanceTracker


def test_capture_provenance(temp_dir):
    """Test capture_provenance."""
    pt = ProvenanceTracker(temp_dir)
    
    agent_config = {
        "idea_maker_model": "gpt-4o",
        "method_model": "gpt-4.1"
    }
    
    provenance = pt.capture_provenance(
        research_idea="Test idea",
        methodology="Test method",
        data_description="Test data",
        agent_config=agent_config
    )
    
    assert "timestamp" in provenance
    assert "environment" in provenance
    assert "inputs" in provenance
    assert "agent_config" in provenance
    assert provenance["agent_config"] == agent_config


def test_capture_environment(temp_dir):
    """Test _capture_environment."""
    pt = ProvenanceTracker(temp_dir)
    env = pt._capture_environment()
    
    assert "python_version" in env
    assert "platform" in env
    assert "cwd" in env
    assert "packages" in env
    assert isinstance(env["packages"], dict)


def test_capture_inputs_with_checksums(temp_dir):
    """Test _capture_inputs with checksums."""
    pt = ProvenanceTracker(temp_dir)
    
    inputs = pt._capture_inputs(
        research_idea="Test idea",
        methodology="Test method",
        data_description="Test data"
    )
    
    assert "research_idea" in inputs
    assert "methodology" in inputs
    assert "data_description" in inputs
    
    for component in inputs.values():
        assert "content" in component
        assert "checksum" in component
        assert "length" in component
        assert len(component["checksum"]) > 0  # SHA256 hex string


def test_capture_data_checksums(temp_dir):
    """Test _capture_data_checksums."""
    pt = ProvenanceTracker(temp_dir)
    
    # Create a test data file
    test_file = temp_dir / "test_data.csv"
    test_file.write_text("test,data\n1,2\n3,4", encoding='utf-8')
    
    checksums = pt._capture_data_checksums()
    
    # Should find the test file
    assert len(checksums) > 0
    assert "test_data.csv" in checksums or any("test_data.csv" in path for path in checksums.keys())


def test_capture_code_versions(temp_dir):
    """Test _capture_code_versions."""
    pt = ProvenanceTracker(temp_dir)
    code_info = pt._capture_code_versions()
    
    assert isinstance(code_info, dict)
    # Should have empirica version
    assert "empirica_version" in code_info


def test_update_execution_end(temp_dir):
    """Test update_execution_end."""
    pt = ProvenanceTracker(temp_dir)
    
    # First capture provenance
    pt.capture_provenance(
        research_idea="Test",
        methodology="Test",
        data_description="Test",
        agent_config={}
    )
    
    # Update with execution end
    pt.update_execution_end(
        execution_time=120.5,
        success=True,
        results_checksum="abc123"
    )
    
    provenance = pt.load_provenance()
    assert provenance is not None
    assert provenance["execution_time_seconds"] == 120.5
    assert provenance["success"] is True
    assert provenance["results_checksum"] == "abc123"
    assert "execution_end" in provenance


def test_load_provenance(temp_dir):
    """Test load_provenance."""
    pt = ProvenanceTracker(temp_dir)
    
    # Capture and save
    pt.capture_provenance(
        research_idea="Test",
        methodology="Test",
        data_description="Test",
        agent_config={}
    )
    
    # Load
    loaded = pt.load_provenance()
    assert loaded is not None
    assert "timestamp" in loaded
    assert "inputs" in loaded


def test_load_nonexistent_provenance(temp_dir):
    """Test loading non-existent provenance returns None."""
    pt = ProvenanceTracker(temp_dir)
    loaded = pt.load_provenance()
    assert loaded is None


def test_provenance_file_creation(temp_dir):
    """Test that provenance file is created."""
    pt = ProvenanceTracker(temp_dir)
    
    pt.capture_provenance(
        research_idea="Test",
        methodology="Test",
        data_description="Test",
        agent_config={}
    )
    
    assert pt.provenance_file.exists()


def test_provenance_checksum_consistency(temp_dir):
    """Test that checksums are consistent."""
    pt = ProvenanceTracker(temp_dir)
    
    inputs1 = pt._capture_inputs("Same content", "Same method", "Same data")
    inputs2 = pt._capture_inputs("Same content", "Same method", "Same data")
    
    # Same content should produce same checksum
    assert inputs1["research_idea"]["checksum"] == inputs2["research_idea"]["checksum"]
    
    # Different content should produce different checksum
    inputs3 = pt._capture_inputs("Different content", "Same method", "Same data")
    assert inputs1["research_idea"]["checksum"] != inputs3["research_idea"]["checksum"]


def test_provenance_execution_start_time(temp_dir):
    """Test execution_start timestamp."""
    pt = ProvenanceTracker(temp_dir)
    
    start_time = datetime.now()
    provenance = pt.capture_provenance(
        research_idea="Test",
        methodology="Test",
        data_description="Test",
        agent_config={},
        execution_start=start_time
    )
    
    assert "execution_start" in provenance
    assert provenance["execution_start"] == start_time.isoformat()

