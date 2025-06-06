"""
Experiment provenance tracking.

Tracks complete experiment provenance including inputs, code, environment,
and configurations for full reproducibility.
"""

import os
import sys
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from uuid import UUID

from ..logger import get_logger

logger = get_logger(__name__)


class ProvenanceTracker:
    """
    Tracks complete experiment provenance for reproducibility.
    
    Captures all information needed to reproduce an experiment:
    - Input data versions and checksums
    - Code/script versions
    - Environment details
    - Agent configurations
    - Execution metadata
    """
    
    def __init__(self, project_dir: str | Path):
        """
        Initialize provenance tracker.
        
        Args:
            project_dir: Project directory to track
        """
        self.project_dir = Path(project_dir)
        self.provenance_file = self.project_dir / "provenance.json"
        self._provenance_data: Optional[Dict[str, Any]] = None
    
    def capture_provenance(
        self,
        research_idea: str,
        methodology: str,
        data_description: str,
        agent_config: Dict[str, Any],
        execution_start: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Capture complete experiment provenance.
        
        Args:
            research_idea: The research idea
            methodology: The methodology
            data_description: Data description
            agent_config: Agent configuration (models, parameters)
            execution_start: Optional execution start time
            
        Returns:
            Dictionary with provenance data
        """
        provenance = {
            "timestamp": datetime.now().isoformat(),
            "execution_start": execution_start.isoformat() if execution_start else datetime.now().isoformat(),
            "environment": self._capture_environment(),
            "inputs": self._capture_inputs(research_idea, methodology, data_description),
            "agent_config": agent_config,
            "data_checksums": self._capture_data_checksums(),
            "code_versions": self._capture_code_versions(),
        }
        
        self._provenance_data = provenance
        self._save_provenance(provenance)
        
        logger.info("Captured experiment provenance")
        return provenance
    
    def _capture_environment(self) -> Dict[str, Any]:
        """Capture environment details."""
        env = {
            "python_version": sys.version,
            "platform": sys.platform,
            "cwd": str(Path.cwd()),
        }
        
        # Capture package versions
        try:
            import importlib.metadata
            packages = {}
            for dist in importlib.metadata.distributions():
                try:
                    packages[dist.metadata["Name"]] = dist.version
                except Exception:
                    pass
            
            # Filter to relevant packages
            relevant_packages = [
                "empirica", "cmbagent", "langchain", "langgraph",
                "numpy", "pandas", "scipy", "matplotlib",
            ]
            env["packages"] = {
                pkg: packages.get(pkg, "unknown")
                for pkg in relevant_packages
                if pkg in packages
            }
        except Exception as e:
            logger.warning(f"Failed to capture package versions: {e}")
            env["packages"] = {}
        
        # Capture environment variables (filtered)
        env_vars = {}
        relevant_vars = ["PATH", "PYTHONPATH", "CONDA_DEFAULT_ENV", "VIRTUAL_ENV"]
        for var in relevant_vars:
            if var in os.environ:
                env_vars[var] = os.environ[var]
        env["environment_variables"] = env_vars
        
        return env
    
    def _capture_inputs(
        self,
        research_idea: str,
        methodology: str,
        data_description: str,
    ) -> Dict[str, Any]:
        """Capture input data with checksums."""
        inputs = {
            "research_idea": {
                "content": research_idea,
                "checksum": self._calculate_checksum(research_idea),
                "length": len(research_idea),
            },
            "methodology": {
                "content": methodology,
                "checksum": self._calculate_checksum(methodology),
                "length": len(methodology),
            },
            "data_description": {
                "content": data_description,
                "checksum": self._calculate_checksum(data_description),
                "length": len(data_description),
            },
        }
        
        return inputs
    
    def _capture_data_checksums(self) -> Dict[str, str]:
        """Capture checksums of data files."""
        checksums = {}
        
        # Look for data files in project directory
        data_patterns = ["*.csv", "*.txt", "*.json", "*.h5", "*.hdf5", "*.fits"]
        
        for pattern in data_patterns:
            for file_path in self.project_dir.rglob(pattern):
                try:
                    checksum = self._calculate_file_checksum(file_path)
                    checksums[str(file_path.relative_to(self.project_dir))] = checksum
                except Exception as e:
                    logger.warning(f"Failed to calculate checksum for {file_path}: {e}")
        
        return checksums
    
    def _capture_code_versions(self) -> Dict[str, Any]:
        """Capture code/script versions."""
        code_info = {}
        
        # Try to get git information if available
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=self.project_dir,
                timeout=5,
            )
            if result.returncode == 0:
                code_info["git_commit"] = result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        
        # Capture empirica version
        try:
            from .. import __version__
            code_info["empirica_version"] = __version__
        except Exception:
            code_info["empirica_version"] = "unknown"
        
        return code_info
    
    def _calculate_checksum(self, content: str) -> str:
        """Calculate SHA256 checksum of content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _save_provenance(self, provenance: Dict[str, Any]) -> None:
        """Save provenance to file."""
        try:
            with open(self.provenance_file, 'w', encoding='utf-8') as f:
                json.dump(provenance, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save provenance: {e}", exc_info=True)
    
    def load_provenance(self) -> Optional[Dict[str, Any]]:
        """Load provenance from file."""
        if not self.provenance_file.exists():
            return None
        
        try:
            with open(self.provenance_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load provenance: {e}", exc_info=True)
            return None
    
    def update_execution_end(
        self,
        execution_time: float,
        success: bool,
        results_checksum: Optional[str] = None,
    ) -> None:
        """
        Update provenance with execution end information.
        
        Args:
            execution_time: Total execution time in seconds
            success: Whether execution was successful
            results_checksum: Optional checksum of results
        """
        provenance = self.load_provenance()
        if not provenance:
            logger.warning("No provenance to update")
            return
        
        provenance["execution_end"] = datetime.now().isoformat()
        provenance["execution_time_seconds"] = execution_time
        provenance["success"] = success
        
        if results_checksum:
            provenance["results_checksum"] = results_checksum
        
        self._save_provenance(provenance)
        self._provenance_data = provenance

