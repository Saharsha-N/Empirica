"""
Version control system for ideas, methods, and results.

Provides git-like versioning with branching for experiment variations.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from uuid import UUID, uuid4

from ..logger import get_logger

logger = get_logger(__name__)


class Version:
    """Represents a version of a research component."""
    
    def __init__(
        self,
        version_id: str,
        content: str,
        timestamp: datetime,
        parent_version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a version.
        
        Args:
            version_id: Unique version identifier
            content: Version content
            timestamp: Version timestamp
            parent_version: Optional parent version ID
            metadata: Optional version metadata
        """
        self.version_id = version_id
        self.content = content
        self.timestamp = timestamp
        self.parent_version = parent_version
        self.metadata = metadata or {}


class VersionControl:
    """
    Version control system for research components.
    
    Tracks versions of ideas, methods, and results with support for
    branching and diff visualization.
    """
    
    def __init__(self, project_dir: str | Path):
        """
        Initialize version control.
        
        Args:
            project_dir: Project directory
        """
        self.project_dir = Path(project_dir)
        self.versions_dir = self.project_dir / ".versions"
        self.versions_dir.mkdir(exist_ok=True)
        self._versions_file = self.versions_dir / "versions.json"
        self._versions: Dict[str, List[Version]] = {}
        self._load_versions()
    
    def _load_versions(self) -> None:
        """Load versions from disk."""
        if not self._versions_file.exists():
            self._versions = {
                "idea": [],
                "method": [],
                "result": [],
            }
            return
        
        try:
            with open(self._versions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self._versions = {}
            for component_type, version_list in data.items():
                self._versions[component_type] = [
                    Version(
                        version_id=v["version_id"],
                        content=v["content"],
                        timestamp=datetime.fromisoformat(v["timestamp"]),
                        parent_version=v.get("parent_version"),
                        metadata=v.get("metadata", {}),
                    )
                    for v in version_list
                ]
        except Exception as e:
            logger.error(f"Failed to load versions: {e}", exc_info=True)
            self._versions = {"idea": [], "method": [], "result": []}
    
    def _save_versions(self) -> None:
        """Save versions to disk."""
        try:
            data = {}
            for component_type, version_list in self._versions.items():
                data[component_type] = [
                    {
                        "version_id": v.version_id,
                        "content": v.content,
                        "timestamp": v.timestamp.isoformat(),
                        "parent_version": v.parent_version,
                        "metadata": v.metadata,
                    }
                    for v in version_list
                ]
            
            with open(self._versions_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save versions: {e}", exc_info=True)
    
    def create_version(
        self,
        component_type: str,
        content: str,
        parent_version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new version of a component.
        
        Args:
            component_type: Type of component ("idea", "method", "result")
            content: Component content
            parent_version: Optional parent version ID
            metadata: Optional version metadata
            
        Returns:
            Version ID
        """
        if component_type not in self._versions:
            self._versions[component_type] = []
        
        version_id = str(uuid4())
        version = Version(
            version_id=version_id,
            content=content,
            timestamp=datetime.now(),
            parent_version=parent_version,
            metadata=metadata or {},
        )
        
        self._versions[component_type].append(version)
        self._save_versions()
        
        logger.debug(f"Created version {version_id} for {component_type}")
        return version_id
    
    def get_version(self, component_type: str, version_id: str) -> Optional[Version]:
        """
        Get a specific version.
        
        Args:
            component_type: Type of component
            version_id: Version ID
            
        Returns:
            Version object or None if not found
        """
        if component_type not in self._versions:
            return None
        
        for version in self._versions[component_type]:
            if version.version_id == version_id:
                return version
        
        return None
    
    def get_latest_version(self, component_type: str) -> Optional[Version]:
        """
        Get the latest version of a component.
        
        Args:
            component_type: Type of component
            
        Returns:
            Latest version or None if no versions exist
        """
        if component_type not in self._versions or not self._versions[component_type]:
            return None
        
        versions = self._versions[component_type]
        return max(versions, key=lambda v: v.timestamp)
    
    def list_versions(self, component_type: str) -> List[Version]:
        """
        List all versions of a component.
        
        Args:
            component_type: Type of component
            
        Returns:
            List of versions (sorted by timestamp, newest first)
        """
        if component_type not in self._versions:
            return []
        
        return sorted(
            self._versions[component_type],
            key=lambda v: v.timestamp,
            reverse=True,
        )
    
    def diff_versions(
        self,
        component_type: str,
        version_id1: str,
        version_id2: str,
    ) -> Dict[str, Any]:
        """
        Calculate diff between two versions.
        
        Args:
            component_type: Type of component
            version_id1: First version ID
            version_id2: Second version ID
            
        Returns:
            Dictionary with diff information
        """
        v1 = self.get_version(component_type, version_id1)
        v2 = self.get_version(component_type, version_id2)
        
        if not v1 or not v2:
            return {"error": "One or both versions not found"}
        
        content1 = v1.content
        content2 = v2.content
        
        # Simple diff calculation
        lines1 = content1.split('\n')
        lines2 = content2.split('\n')
        
        added = [line for line in lines2 if line not in lines1]
        removed = [line for line in lines1 if line not in lines2]
        
        return {
            "version1": version_id1,
            "version2": version_id2,
            "added_lines": len(added),
            "removed_lines": len(removed),
            "added": added[:20],  # Limit to first 20
            "removed": removed[:20],
            "length_diff": len(content2) - len(content1),
        }
    
    def rollback_to_version(
        self,
        component_type: str,
        version_id: str,
    ) -> Optional[str]:
        """
        Rollback to a specific version.
        
        Args:
            component_type: Type of component
            version_id: Version ID to rollback to
            
        Returns:
            Content of the rolled-back version, or None if not found
        """
        version = self.get_version(component_type, version_id)
        if not version:
            return None
        
        # Create a new version from the rollback (for history)
        new_version_id = self.create_version(
            component_type,
            version.content,
            parent_version=version_id,
            metadata={"rollback": True, "rolled_back_from": version_id},
        )
        
        logger.info(f"Rolled back {component_type} to version {version_id}")
        return version.content
    
    def create_branch(
        self,
        component_type: str,
        base_version_id: str,
        branch_name: str,
        content: str,
    ) -> str:
        """
        Create a branch (experiment variation) from a base version.
        
        Args:
            component_type: Type of component
            base_version_id: Base version ID
            branch_name: Name of the branch
            content: Content for the branch
            
        Returns:
            New version ID
        """
        version_id = self.create_version(
            component_type,
            content,
            parent_version=base_version_id,
            metadata={"branch": branch_name, "is_branch": True},
        )
        
        logger.info(f"Created branch '{branch_name}' from version {base_version_id}")
        return version_id
    
    def get_version_history(self, component_type: str) -> List[Dict[str, Any]]:
        """
        Get version history with relationships.
        
        Args:
            component_type: Type of component
            
        Returns:
            List of version information with relationships
        """
        versions = self.list_versions(component_type)
        
        history = []
        for version in versions:
            history.append({
                "version_id": version.version_id,
                "timestamp": version.timestamp.isoformat(),
                "parent_version": version.parent_version,
                "metadata": version.metadata,
                "content_length": len(version.content),
            })
        
        return history

