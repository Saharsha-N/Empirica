"""Unit tests for checkpoint manager."""
import pytest
from pathlib import Path

from empirica.research import Research
from empirica.reproducibility.checkpoints import CheckpointManager


def test_create_checkpoint(temp_dir):
    """Test create_checkpoint."""
    cm = CheckpointManager(temp_dir)
    
    research = Research(idea="Test idea", methodology="Test method")
    checkpoint_id = cm.create_checkpoint(
        stage="idea_generated",
        research=research,
        metadata={"test": True}
    )
    
    assert checkpoint_id is not None
    assert isinstance(checkpoint_id, str)


def test_get_checkpoint(temp_dir):
    """Test get_checkpoint and get_latest_checkpoint."""
    cm = CheckpointManager(temp_dir)
    
    research = Research(idea="Test idea")
    checkpoint_id = cm.create_checkpoint("idea_generated", research)
    
    checkpoint = cm.get_checkpoint(checkpoint_id)
    assert checkpoint is not None
    assert checkpoint.checkpoint_id == checkpoint_id
    assert checkpoint.stage == "idea_generated"
    assert checkpoint.research.idea == "Test idea"
    
    latest = cm.get_latest_checkpoint()
    assert latest is not None
    assert latest.checkpoint_id == checkpoint_id


def test_list_checkpoints(temp_dir):
    """Test list_checkpoints with stage filter."""
    cm = CheckpointManager(temp_dir)
    
    research1 = Research(idea="Idea 1")
    research2 = Research(methodology="Method 1")
    
    cm.create_checkpoint("idea_generated", research1)
    cm.create_checkpoint("method_generated", research2)
    cm.create_checkpoint("idea_generated", research1)
    
    all_checkpoints = cm.list_checkpoints()
    assert len(all_checkpoints) == 3
    
    idea_checkpoints = cm.list_checkpoints(stage="idea_generated")
    assert len(idea_checkpoints) == 2
    
    method_checkpoints = cm.list_checkpoints(stage="method_generated")
    assert len(method_checkpoints) == 1


def test_restore_checkpoint(temp_dir):
    """Test restore_checkpoint."""
    cm = CheckpointManager(temp_dir)
    
    original_research = Research(
        idea="Original idea",
        methodology="Original method",
        results="Original results"
    )
    checkpoint_id = cm.create_checkpoint("results_generated", original_research)
    
    restored = cm.restore_checkpoint(checkpoint_id)
    
    assert restored is not None
    assert restored.idea == "Original idea"
    assert restored.methodology == "Original method"
    assert restored.results == "Original results"
    # Should be a copy, not the same object
    assert restored is not original_research


def test_delete_checkpoint(temp_dir):
    """Test delete_checkpoint."""
    cm = CheckpointManager(temp_dir)
    
    research = Research(idea="Test")
    checkpoint_id = cm.create_checkpoint("idea_generated", research)
    
    deleted = cm.delete_checkpoint(checkpoint_id)
    assert deleted is True
    
    # Verify deletion
    checkpoint = cm.get_checkpoint(checkpoint_id)
    assert checkpoint is None


def test_delete_nonexistent_checkpoint(temp_dir):
    """Test deleting a non-existent checkpoint."""
    cm = CheckpointManager(temp_dir)
    
    deleted = cm.delete_checkpoint("nonexistent-id")
    assert deleted is False


def test_auto_checkpoint(temp_dir):
    """Test auto_checkpoint at key stages."""
    cm = CheckpointManager(temp_dir)
    
    research = Research(idea="Test idea")
    
    # Key stage should create checkpoint
    checkpoint_id = cm.auto_checkpoint("idea_generated", research)
    assert checkpoint_id is not None
    
    # Non-key stage should return None
    checkpoint_id = cm.auto_checkpoint("custom_stage", research)
    assert checkpoint_id is None


def test_checkpoint_metadata_preservation(temp_dir):
    """Test that checkpoint metadata is preserved."""
    cm = CheckpointManager(temp_dir)
    
    metadata = {"execution_time": 45.0, "success": True}
    research = Research(idea="Test")
    checkpoint_id = cm.create_checkpoint(
        "idea_generated",
        research,
        metadata=metadata
    )
    
    checkpoint = cm.get_checkpoint(checkpoint_id)
    assert checkpoint.metadata == metadata


def test_checkpoint_timestamps(temp_dir):
    """Test that checkpoints have proper timestamps."""
    cm = CheckpointManager(temp_dir)
    
    import time
    research = Research(idea="Test")
    
    c1_id = cm.create_checkpoint("idea_generated", research)
    time.sleep(0.01)
    c2_id = cm.create_checkpoint("method_generated", research)
    
    c1 = cm.get_checkpoint(c1_id)
    c2 = cm.get_checkpoint(c2_id)
    
    assert c2.timestamp > c1.timestamp


def test_get_latest_checkpoint_by_stage(temp_dir):
    """Test get_latest_checkpoint filtered by stage."""
    cm = CheckpointManager(temp_dir)
    
    research = Research(idea="Idea")
    cm.create_checkpoint("idea_generated", research)
    
    research2 = Research(methodology="Method")
    cm.create_checkpoint("method_generated", research2)
    
    latest_idea = cm.get_latest_checkpoint(stage="idea_generated")
    assert latest_idea is not None
    assert latest_idea.stage == "idea_generated"
    
    latest_method = cm.get_latest_checkpoint(stage="method_generated")
    assert latest_method is not None
    assert latest_method.stage == "method_generated"


def test_checkpoint_research_isolation(temp_dir):
    """Test that restored research objects are independent."""
    cm = CheckpointManager(temp_dir)
    
    research = Research(idea="Original")
    checkpoint_id = cm.create_checkpoint("idea_generated", research)
    
    restored1 = cm.restore_checkpoint(checkpoint_id)
    restored2 = cm.restore_checkpoint(checkpoint_id)
    
    # Should be independent copies
    assert restored1 is not restored2
    restored1.idea = "Modified"
    assert restored2.idea == "Original"

