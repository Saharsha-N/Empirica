"""
Experiment checkpointing system.

Provides save/restore functionality for experiment state at key stages.
    # improve documentation
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Optional, Any, List
from datetime import datetime
from uuid import UUID, uuid4

from ..research import Research
from ..logger import get_logger

logger = get_logger(__name__)


class Checkpoint:
    """Represents an experiment checkpoint."""
    
    def __init__(
        self,
        checkpoint_id: str,
        stage: str,
        research: Research,
        timestamp: datetime,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a checkpoint.
        
        Args:
            checkpoint_id: Unique checkpoint identifier
            stage: Stage name (e.g., "idea_generated", "method_generated", "results_generated")
            research: Research object state
            timestamp: Checkpoint timestamp
            metadata: Optional checkpoint metadata
        """
        self.checkpoint_id = checkpoint_id
        self.stage = stage
        self.research = research
        self.timestamp = timestamp
        self.metadata = metadata or {}


class CheckpointManager:
    """
    Manages experiment checkpoints for save/restore functionality.
    
    Provides automatic checkpointing at key stages and manual checkpoint
    creation/restoration.
    """
    
    def __init__(self, project_dir: str | Path):
        """
        Initialize checkpoint manager.
        
        Args:
            project_dir: Project directory
        """
        self.project_dir = Path(project_dir)
        self.checkpoints_dir = self.project_dir / ".checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        self._checkpoints_file = self.checkpoints_dir / "checkpoints.json"
        self._checkpoints: Dict[str, Checkpoint] = {}
        self._load_checkpoints()
    
    def _load_checkpoints(self) -> None:
        """Load checkpoints from disk."""
        if not self._checkpoints_file.exists():
            self._checkpoints = {}
            return
        
        try:
            with open(self._checkpoints_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self._checkpoints = {}
            for checkpoint_id, checkpoint_data in data.items():
                research_data = checkpoint_data["research"]
                research = Research(
                    data_description=research_data.get("data_description", ""),
                    idea=research_data.get("idea", ""),
                    methodology=research_data.get("methodology", ""),
                    results=research_data.get("results", ""),
                    plot_paths=research_data.get("plot_paths", []),
                    keywords=research_data.get("keywords", {}),
                )
                
                checkpoint = Checkpoint(
                    checkpoint_id=checkpoint_id,
                    stage=checkpoint_data["stage"],
                    research=research,
                    timestamp=datetime.fromisoformat(checkpoint_data["timestamp"]),
                    metadata=checkpoint_data.get("metadata", {}),
                )
                self._checkpoints[checkpoint_id] = checkpoint
        except Exception as e:
            logger.error(f"Failed to load checkpoints: {e}", exc_info=True)
            self._checkpoints = {}
    
    def _save_checkpoints(self) -> None:
        """Save checkpoints to disk."""
        try:
            data = {}
            for checkpoint_id, checkpoint in self._checkpoints.items():
                data[checkpoint_id] = {
                    "stage": checkpoint.stage,
                    "timestamp": checkpoint.timestamp.isoformat(),
                    "research": {
                        "data_description": checkpoint.research.data_description,
                        "idea": checkpoint.research.idea,
                        "methodology": checkpoint.research.methodology,
                        "results": checkpoint.research.results,
                        "plot_paths": checkpoint.research.plot_paths,
                        "keywords": checkpoint.research.keywords,
                    },
                    "metadata": checkpoint.metadata,
                }
            
            with open(self._checkpoints_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save checkpoints: {e}", exc_info=True)
    
    def create_checkpoint(
        self,
        stage: str,
        research: Research,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a checkpoint.
        
        Args:
            stage: Stage name
            research: Research object to checkpoint
            metadata: Optional checkpoint metadata
            
        Returns:
            Checkpoint ID
        """
        checkpoint_id = str(uuid4())
        
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            stage=stage,
            research=research,
            timestamp=datetime.now(),
            metadata=metadata or {},
        )
        
        self._checkpoints[checkpoint_id] = checkpoint
        self._save_checkpoints()
        
        logger.info(f"Created checkpoint {checkpoint_id} at stage '{stage}'")
        return checkpoint_id
    
    def get_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """
        Get a checkpoint by ID.
        
        Args:
            checkpoint_id: Checkpoint ID
            
        Returns:
            Checkpoint object or None if not found
        """
        return self._checkpoints.get(checkpoint_id)
    
    def get_latest_checkpoint(self, stage: Optional[str] = None) -> Optional[Checkpoint]:
        """
        Get the latest checkpoint, optionally filtered by stage.
        
        Args:
            stage: Optional stage filter
            
        Returns:
            Latest checkpoint or None
        """
        checkpoints = list(self._checkpoints.values())
        
        if stage:
            checkpoints = [c for c in checkpoints if c.stage == stage]
        
        if not checkpoints:
            return None
        
        return max(checkpoints, key=lambda c: c.timestamp)
    
    def list_checkpoints(self, stage: Optional[str] = None) -> List[Checkpoint]:
        """
        List all checkpoints, optionally filtered by stage.
        
        Args:
            stage: Optional stage filter
            
        Returns:
            List of checkpoints (sorted by timestamp, newest first)
        """
        checkpoints = list(self._checkpoints.values())
        
        if stage:
            checkpoints = [c for c in checkpoints if c.stage == stage]
        
        return sorted(checkpoints, key=lambda c: c.timestamp, reverse=True)
    
    def restore_checkpoint(self, checkpoint_id: str) -> Optional[Research]:
        """
        Restore a checkpoint.
        
        Args:
            checkpoint_id: Checkpoint ID to restore
            
        Returns:
            Research object from checkpoint, or None if not found
        """
        checkpoint = self.get_checkpoint(checkpoint_id)
        if not checkpoint:
            return None
        
        logger.info(f"Restoring checkpoint {checkpoint_id} from stage '{checkpoint.stage}'")
        
        # Return a copy of the research object
        return Research(
            data_description=checkpoint.research.data_description,
            idea=checkpoint.research.idea,
            methodology=checkpoint.research.methodology,
            results=checkpoint.research.results,
            plot_paths=checkpoint.research.plot_paths.copy() if checkpoint.research.plot_paths else [],
            keywords=checkpoint.research.keywords.copy() if isinstance(checkpoint.research.keywords, dict) else checkpoint.research.keywords,
        )
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint.
        
        Args:
            checkpoint_id: Checkpoint ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        if checkpoint_id in self._checkpoints:
            del self._checkpoints[checkpoint_id]
            self._save_checkpoints()
            logger.info(f"Deleted checkpoint {checkpoint_id}")
            return True
        return False
    
    def auto_checkpoint(
        self,
        stage: str,
        research: Research,
    ) -> Optional[str]:
        """
        Automatically create checkpoint at key stages.
        
        Args:
            stage: Stage name
            research: Research object
            
        Returns:
            Checkpoint ID if created, None if skipped
        """
        # Define key stages that should be checkpointed
        key_stages = [
            "idea_generated",
            "method_generated",
            "results_generated",
            "paper_generated",
        ]
        
        if stage in key_stages:
            return self.create_checkpoint(stage, research)
        
        return None

