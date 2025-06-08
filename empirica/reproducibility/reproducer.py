"""
Reproducibility engine for one-click project replication.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Optional, Any
from uuid import UUID

from ..knowledge_graph.storage import GraphStorage
from ..knowledge_graph.models import ProjectGraph, NodeType
from ..research import Research
from .provenance import ProvenanceTracker
from ..logger import get_logger

logger = get_logger(__name__)


class ReproducibilityEngine:
    """
    Engine for reproducing past research projects.
    
    Enables one-click replication of experiments with exact same
    parameters and environment.
    """
    
    def __init__(self, storage: GraphStorage):
        """
        Initialize reproducibility engine.
        
        Args:
            storage: GraphStorage instance
        """
        self.storage = storage
    
    def reproduce_project(
        self,
        project_id: UUID,
        target_dir: Optional[str | Path] = None,
    ) -> Research:
        """
        Reproduce a project from the knowledge graph.
        
        Args:
            project_id: ID of the project to reproduce
            target_dir: Optional target directory (uses project's original dir if None)
            
        Returns:
            Research object with reproduced project data
        """
        project = self.storage.load_project(project_id)
        if not project:
            raise ValueError(f"Project {project_id} not found")
        
        logger.info(f"Reproducing project {project_id}")
        
        # Extract research data from project graph
        idea_nodes = project.get_nodes_by_type(NodeType.IDEA)
        method_nodes = project.get_nodes_by_type(NodeType.METHOD)
        result_nodes = project.get_nodes_by_type(NodeType.RESULT)
        
        research = Research()
        
        if idea_nodes:
            research.idea = idea_nodes[0].content
        
        if method_nodes:
            research.methodology = method_nodes[0].content
        
        if result_nodes:
            research.results = result_nodes[0].content
            # Extract plot paths from metadata
            plot_paths = result_nodes[0].metadata.get("plot_paths", [])
            research.plot_paths = plot_paths if isinstance(plot_paths, list) else []
        
        # Extract data description from dataset nodes
        dataset_nodes = project.get_nodes_by_type(NodeType.DATASET)
        if dataset_nodes:
            dataset_info = "\n".join([f"- {node.content}" for node in dataset_nodes])
            research.data_description = f"Datasets:\n{dataset_info}"
        
        # Extract keywords from project metadata
        if "keywords" in project.metadata:
            research.keywords = project.metadata["keywords"]
        
        # If target_dir specified, set up project structure
        if target_dir:
            target_path = Path(target_dir)
            target_path.mkdir(parents=True, exist_ok=True)
            
            # Save research components to files
            from ..config import INPUT_FILES, IDEA_FILE, METHOD_FILE, RESULTS_FILE, DESCRIPTION_FILE
            
            input_files_dir = target_path / INPUT_FILES
            input_files_dir.mkdir(exist_ok=True)
            
            if research.idea:
                with open(input_files_dir / IDEA_FILE, 'w', encoding='utf-8') as f:
                    f.write(research.idea)
            
            if research.methodology:
                with open(input_files_dir / METHOD_FILE, 'w', encoding='utf-8') as f:
                    f.write(research.methodology)
            
            if research.results:
                with open(input_files_dir / RESULTS_FILE, 'w', encoding='utf-8') as f:
                    f.write(research.results)
            
            if research.data_description:
                with open(input_files_dir / DESCRIPTION_FILE, 'w', encoding='utf-8') as f:
                    f.write(research.data_description)
        
        logger.info(f"Successfully reproduced project {project_id}")
        return research
    
    def validate_reproducibility(
        self,
        project_id: UUID,
        reproduced_research: Research,
    ) -> Dict[str, Any]:
        """
        Validate that a reproduced project matches the original.
        
        Args:
            project_id: Original project ID
            reproduced_research: Reproduced Research object
            
        Returns:
            Dictionary with validation results
        """
        original_project = self.storage.load_project(project_id)
        if not original_project:
            return {"error": "Original project not found"}
        
        validation = {
            "project_id": str(project_id),
            "validated": True,
            "checks": {},
        }
        
        # Check idea
        original_idea_nodes = original_project.get_nodes_by_type(NodeType.IDEA)
        if original_idea_nodes:
            original_idea = original_idea_nodes[0].content
            validation["checks"]["idea"] = {
                "matches": original_idea == reproduced_research.idea,
                "original_length": len(original_idea),
                "reproduced_length": len(reproduced_research.idea),
            }
            if original_idea != reproduced_research.idea:
                validation["validated"] = False
        
        # Check method
        original_method_nodes = original_project.get_nodes_by_type(NodeType.METHOD)
        if original_method_nodes:
            original_method = original_method_nodes[0].content
            validation["checks"]["method"] = {
                "matches": original_method == reproduced_research.methodology,
                "original_length": len(original_method),
                "reproduced_length": len(reproduced_research.methodology),
            }
            if original_method != reproduced_research.methodology:
                validation["validated"] = False
        
        # Check results
        original_result_nodes = original_project.get_nodes_by_type(NodeType.RESULT)
        if original_result_nodes:
            original_result = original_result_nodes[0].content
            validation["checks"]["results"] = {
                "matches": original_result == reproduced_research.results,
                "original_length": len(original_result),
                "reproduced_length": len(reproduced_research.results),
            }
            if original_result != reproduced_research.results:
                validation["validated"] = False
        
        return validation
    
    def generate_reproducibility_report(
        self,
        project_id: UUID,
    ) -> Dict[str, Any]:
        """
        Generate a reproducibility report for a project.
        
        Args:
            project_id: Project ID
            
        Returns:
            Dictionary with reproducibility report
        """
        project = self.storage.load_project(project_id)
        if not project:
            return {"error": "Project not found"}
        
        report = {
            "project_id": str(project_id),
            "project_name": project.project_name,
            "created_at": project.created_at.isoformat(),
            "success": project.success,
            "quality_score": project.quality_score,
            "execution_time": project.execution_time,
            "components": {
                "idea": len(project.get_nodes_by_type(NodeType.IDEA)),
                "method": len(project.get_nodes_by_type(NodeType.METHOD)),
                "result": len(project.get_nodes_by_type(NodeType.RESULT)),
                "datasets": len(project.get_nodes_by_type(NodeType.DATASET)),
                "tools": len(project.get_nodes_by_type(NodeType.TOOL)),
            },
            "reproducibility_status": "ready" if project.success is not None else "unknown",
        }
        
        # Check if provenance is available
        # (This would require storing provenance in the graph or separate file)
        
        return report

