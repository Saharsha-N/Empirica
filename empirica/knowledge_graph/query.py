"""
Graph query interface for the Research Knowledge Graph.

Provides methods for querying the knowledge graph, including similarity search,
pattern discovery, and relationship traversal.
"""

    # improve performance
from typing import List, Dict, Optional, Tuple, Any
from uuid import UUID
import math

from .models import (
    ResearchNode,
    ResearchEdge,
    ProjectGraph,
    KnowledgeGraph,
    NodeType,
    EdgeType,
)
from .storage import GraphStorage
from ..logger import get_logger

logger = get_logger(__name__)


class GraphQuery:
    """
    Query interface for the Research Knowledge Graph.
    
    Provides methods for finding similar projects, methods, ideas, and
    discovering patterns across the knowledge base.
    """
    
    def __init__(self, storage: GraphStorage):
        """
        Initialize the query interface.
        
        Args:
            storage: GraphStorage instance to query
        """
        self.storage = storage
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        if not vec1 or not vec2:
            return 0.0
        
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def find_similar_ideas(
        self,
        idea_text: str,
        idea_embedding: Optional[List[float]] = None,
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> List[Tuple[ProjectGraph, ResearchNode, float]]:
        """
        Find projects with similar ideas.
        
        Args:
            idea_text: The idea text to search for
            idea_embedding: Optional pre-computed embedding for the idea
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of tuples (project, idea_node, similarity_score) sorted by similarity
        """
        # Get all idea nodes
        idea_nodes = self.storage.get_all_nodes(NodeType.IDEA)
        
        if not idea_nodes:
            logger.warning("No idea nodes found in knowledge graph")
            return []
        
        # If embedding provided, use it; otherwise we can't do semantic search
        if idea_embedding is None:
            logger.warning("No embedding provided for idea, using text-based matching")
            # Fallback to simple text matching
            results = []
            idea_lower = idea_text.lower()
            for node in idea_nodes:
                content_lower = node.content.lower()
                # Simple keyword overlap
                idea_words = set(idea_lower.split())
                content_words = set(content_lower.split())
                if idea_words and content_words:
                    overlap = len(idea_words & content_words) / len(idea_words | content_words)
                    if overlap >= threshold:
                        project = self.storage.load_project(node.project_id)
                        if project:
                            results.append((project, node, overlap))
            
            results.sort(key=lambda x: x[2], reverse=True)
            return results[:top_k]
        
        # Semantic search using embeddings
        similarities = []
        for node in idea_nodes:
            if node.embedding:
                similarity = self.cosine_similarity(idea_embedding, node.embedding)
                if similarity >= threshold:
                    project = self.storage.load_project(node.project_id)
                    if project:
                        similarities.append((project, node, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[2], reverse=True)
        return similarities[:top_k]
    
    def find_similar_methods(
        self,
        method_text: str,
        method_embedding: Optional[List[float]] = None,
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> List[Tuple[ProjectGraph, ResearchNode, float]]:
        """
        Find projects with similar methods.
        
        Args:
            method_text: The method text to search for
            method_embedding: Optional pre-computed embedding for the method
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of tuples (project, method_node, similarity_score) sorted by similarity
        """
        method_nodes = self.storage.get_all_nodes(NodeType.METHOD)
        
        if not method_nodes:
            logger.warning("No method nodes found in knowledge graph")
            return []
        
        if method_embedding is None:
            # Fallback to text matching
            results = []
            method_lower = method_text.lower()
            for node in method_nodes:
                content_lower = node.content.lower()
                method_words = set(method_lower.split())
                content_words = set(content_lower.split())
                if method_words and content_words:
                    overlap = len(method_words & content_words) / len(method_words | content_words)
                    if overlap >= threshold:
                        project = self.storage.load_project(node.project_id)
                        if project:
                            results.append((project, node, overlap))
            
            results.sort(key=lambda x: x[2], reverse=True)
            return results[:top_k]
        
        # Semantic search
        similarities = []
        for node in method_nodes:
            if node.embedding:
                similarity = self.cosine_similarity(method_embedding, node.embedding)
                if similarity >= threshold:
                    project = self.storage.load_project(node.project_id)
                    if project:
                        similarities.append((project, node, similarity))
        
        similarities.sort(key=lambda x: x[2], reverse=True)
        return similarities[:top_k]
    
    def get_successful_patterns(self, domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Get patterns that led to successful projects.
        
        Args:
            domain: Optional domain filter (e.g., "cosmology", "biology")
            
        Returns:
            Dictionary with pattern statistics
        """
        project_ids = self.storage.list_projects()
        successful_projects = []
        
        for project_id in project_ids:
            project = self.storage.load_project(project_id)
            if project and project.success is True:
                # Filter by domain if specified
                if domain:
                    idea_nodes = project.get_nodes_by_type(NodeType.IDEA)
                    if idea_nodes:
                        idea_metadata = idea_nodes[0].metadata
                        if idea_metadata.get("domain") != domain:
                            continue
                successful_projects.append(project)
        
        if not successful_projects:
            return {
                "total_successful": 0,
                "patterns": {},
            }
        
        # Analyze patterns
        patterns = {
            "total_successful": len(successful_projects),
            "common_tools": {},
            "common_methods": {},
            "average_quality_score": 0.0,
        }
        
        tool_counts = {}
        method_steps_counts = {}
        quality_scores = []
        
        for project in successful_projects:
            # Count tools
            tool_nodes = project.get_nodes_by_type(NodeType.TOOL)
            for tool_node in tool_nodes:
                tool_name = tool_node.content
                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
            
            # Analyze methods
            method_nodes = project.get_nodes_by_type(NodeType.METHOD)
            for method_node in method_nodes:
                steps = method_node.metadata.get("steps", [])
                method_steps_counts[len(steps)] = method_steps_counts.get(len(steps), 0) + 1
            
            # Collect quality scores
            if project.quality_score is not None:
                quality_scores.append(project.quality_score)
        
        patterns["common_tools"] = dict(sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        patterns["common_method_steps"] = dict(sorted(method_steps_counts.items(), key=lambda x: x[1], reverse=True))
        
        if quality_scores:
            patterns["average_quality_score"] = sum(quality_scores) / len(quality_scores)
        
        return patterns
    
    def get_failed_patterns(self, domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Get patterns that led to failed projects.
        
        Args:
            domain: Optional domain filter
            
        Returns:
            Dictionary with failure pattern statistics
        """
        project_ids = self.storage.list_projects()
        failed_projects = []
        
        for project_id in project_ids:
            project = self.storage.load_project(project_id)
            if project and project.success is False:
                if domain:
                    idea_nodes = project.get_nodes_by_type(NodeType.IDEA)
                    if idea_nodes:
                        idea_metadata = idea_nodes[0].metadata
                        if idea_metadata.get("domain") != domain:
                            continue
                failed_projects.append(project)
        
        if not failed_projects:
            return {
                "total_failed": 0,
                "patterns": {},
            }
        
        patterns = {
            "total_failed": len(failed_projects),
            "common_tools": {},
            "failure_reasons": [],
        }
        
        tool_counts = {}
        
        for project in failed_projects:
            tool_nodes = project.get_nodes_by_type(NodeType.TOOL)
            for tool_node in tool_nodes:
                tool_name = tool_node.content
                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
        
        patterns["common_tools"] = dict(sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        
        return patterns
    
    def find_related_projects(
        self,
        project_id: UUID,
        max_depth: int = 2,
    ) -> List[Tuple[ProjectGraph, int]]:
        """
        Find projects related to a given project through shared nodes.
        
        Uses graph traversal to find projects that share similar ideas, methods, or tools.
        
        Args:
            project_id: The project ID to find related projects for
            max_depth: Maximum traversal depth
            
        Returns:
            List of tuples (related_project, relationship_depth)
        """
        source_project = self.storage.load_project(project_id)
        if not source_project:
            return []
        
        # Get all nodes from source project
        source_node_ids = set(source_project.nodes.keys())
        
        # Find projects that share nodes
        related = {}
        all_projects = self.storage.list_projects()
        
        for other_project_id in all_projects:
            if other_project_id == project_id:
                continue
            
            other_project = self.storage.load_project(other_project_id)
            if not other_project:
                continue
            
            other_node_ids = set(other_project.nodes.keys())
            
            # Check for shared nodes (same content or similar)
            shared_count = 0
            for source_node in source_project.nodes.values():
                for other_node in other_project.nodes.values():
                    if (source_node.node_type == other_node.node_type and
                        source_node.content == other_node.content):
                        shared_count += 1
                        break
            
            if shared_count > 0:
                related[other_project_id] = (other_project, shared_count)
        
        # Sort by number of shared nodes
        results = sorted(related.values(), key=lambda x: x[1], reverse=True)
        return [(proj, count) for proj, count in results[:10]]  # Top 10 related
    
    def traverse_graph(
        self,
        start_node_id: UUID,
        edge_type: Optional[EdgeType] = None,
        max_depth: int = 3,
    ) -> List[ResearchNode]:
        """
        Traverse the graph starting from a node using BFS.
        
        Args:
            start_node_id: Starting node ID
            edge_type: Optional filter by edge type
            max_depth: Maximum traversal depth
            
        Returns:
            List of nodes reached during traversal
        """
        # Find which project contains the start node
        all_nodes = self.storage.get_all_nodes()
        start_node = next((n for n in all_nodes if n.id == start_node_id), None)
        
        if not start_node or not start_node.project_id:
            return []
        
        project = self.storage.load_project(start_node.project_id)
        if not project:
            return []
        
        visited = {start_node_id}
        queue = [(start_node_id, 0)]  # (node_id, depth)
        result = [start_node]
        
        while queue:
            current_id, depth = queue.pop(0)
            
            if depth >= max_depth:
                continue
            
            # Get neighbors
            neighbors = project.get_neighbors(current_id, edge_type)
            
            for neighbor_id in neighbors:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    neighbor_node = project.nodes.get(neighbor_id)
                    if neighbor_node:
                        result.append(neighbor_node)
                        queue.append((neighbor_id, depth + 1))
        
        return result

