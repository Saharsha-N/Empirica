"""Unit tests for version control."""
import pytest
from pathlib import Path

from empirica.reproducibility.versioning import VersionControl


def test_create_version(temp_dir):
    """Test create_version."""
    vc = VersionControl(temp_dir)
    
    version_id = vc.create_version(
        component_type="idea",
        content="Test idea content",
        metadata={"test": True}
    )
    
    assert version_id is not None
    assert isinstance(version_id, str)


def test_get_version(temp_dir):
    """Test get_version and get_latest_version."""
    vc = VersionControl(temp_dir)
    
    version_id = vc.create_version("idea", "Test idea")
    version = vc.get_version("idea", version_id)
    
    assert version is not None
    assert version.version_id == version_id
    assert version.content == "Test idea"
    
    latest = vc.get_latest_version("idea")
    assert latest is not None
    assert latest.version_id == version_id


def test_list_versions(temp_dir):
    """Test list_versions."""
    vc = VersionControl(temp_dir)
    
    # Create multiple versions
    vc.create_version("idea", "Idea v1")
    vc.create_version("idea", "Idea v2")
    vc.create_version("idea", "Idea v3")
    
    versions = vc.list_versions("idea")
    assert len(versions) == 3
    # Should be sorted newest first
    assert versions[0].content == "Idea v3"
    assert versions[-1].content == "Idea v1"


def test_diff_versions(temp_dir):
    """Test diff_versions."""
    vc = VersionControl(temp_dir)
    
    v1_id = vc.create_version("idea", "Original idea\nLine 2\nLine 3")
    v2_id = vc.create_version("idea", "Updated idea\nLine 2\nNew line")
    
    diff = vc.diff_versions("idea", v1_id, v2_id)
    
    assert "version1" in diff
    assert "version2" in diff
    assert "added_lines" in diff
    assert "removed_lines" in diff
    assert diff["length_diff"] != 0


def test_rollback_to_version(temp_dir):
    """Test rollback_to_version."""
    vc = VersionControl(temp_dir)
    
    original_id = vc.create_version("idea", "Original content")
    vc.create_version("idea", "Changed content")
    
    # Rollback
    rolled_back_content = vc.rollback_to_version("idea", original_id)
    
    assert rolled_back_content == "Original content"
    
    # Should create a new version from rollback
    latest = vc.get_latest_version("idea")
    assert latest.metadata.get("rollback") is True


def test_create_branch(temp_dir):
    """Test create_branch."""
    vc = VersionControl(temp_dir)
    
    base_id = vc.create_version("idea", "Base idea")
    branch_id = vc.create_branch(
        component_type="idea",
        base_version_id=base_id,
        branch_name="experiment-1",
        content="Branched idea"
    )
    
    assert branch_id is not None
    branch = vc.get_version("idea", branch_id)
    assert branch.parent_version == base_id
    assert branch.metadata.get("branch") == "experiment-1"
    assert branch.metadata.get("is_branch") is True


def test_get_version_history(temp_dir):
    """Test get_version_history."""
    vc = VersionControl(temp_dir)
    
    v1_id = vc.create_version("idea", "Idea 1")
    v2_id = vc.create_version("idea", "Idea 2", parent_version=v1_id)
    
    history = vc.get_version_history("idea")
    
    assert len(history) >= 2
    # Check that relationships are preserved
    v2_entry = next((v for v in history if v["version_id"] == v2_id), None)
    assert v2_entry is not None
    assert v2_entry["parent_version"] == v1_id


def test_version_metadata_preservation(temp_dir):
    """Test that version metadata is preserved."""
    vc = VersionControl(temp_dir)
    
    metadata = {"author": "test", "timestamp": "2024-01-01"}
    version_id = vc.create_version(
        component_type="idea",
        content="Test",
        metadata=metadata
    )
    
    version = vc.get_version("idea", version_id)
    assert version.metadata == metadata


def test_multiple_component_types(temp_dir):
    """Test versioning for different component types."""
    vc = VersionControl(temp_dir)
    
    idea_id = vc.create_version("idea", "Test idea")
    method_id = vc.create_version("method", "Test method")
    result_id = vc.create_version("result", "Test result")
    
    assert vc.get_version("idea", idea_id) is not None
    assert vc.get_version("method", method_id) is not None
    assert vc.get_version("result", result_id) is not None
    
    # Each should have separate version lists
    idea_versions = vc.list_versions("idea")
    method_versions = vc.list_versions("method")
    
    assert len(idea_versions) == 1
    assert len(method_versions) == 1


def test_get_latest_version_nonexistent(temp_dir):
    """Test get_latest_version for component with no versions."""
    vc = VersionControl(temp_dir)
    
    latest = vc.get_latest_version("nonexistent")
    assert latest is None


def test_version_timestamps(temp_dir):
    """Test that versions have proper timestamps."""
    vc = VersionControl(temp_dir)
    
    import time
    v1_id = vc.create_version("idea", "Idea 1")
    time.sleep(0.01)
    v2_id = vc.create_version("idea", "Idea 2")
    
    v1 = vc.get_version("idea", v1_id)
    v2 = vc.get_version("idea", v2_id)
    
    assert v2.timestamp > v1.timestamp

