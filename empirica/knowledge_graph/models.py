"""
Core data models for the Research Knowledge Graph.

Defines the structure for nodes, edges, and graphs that represent
research projects and their relationships.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime
from uuid import uuid4, UUID
from pydantic import BaseModel, Field


class NodeType(str, Enum):
    """Types of nodes in the research knowledge graph."""
    IDEA = "idea"
    METHOD = "method"
    DATASET = "dataset"
    TOOL = "tool"
    RESULT = "result"
    FINDING = "finding"
    AGENT = "agent"
    CITATION = "citation"
    PROJECT = "project"


class EdgeType(str, Enum):
    """Types of relationships between nodes."""
    USES = "uses"  # method uses tool, idea uses dataset
    GENERATES = "generates"  # method generates result, idea generates method
    LEADS_TO = "leads_to"  # idea leads to result, method leads to finding
    SIMILAR_TO = "similar_to"  # idea similar to idea, method similar to method
    CITES = "cites"  # idea cites citation, result cites citation
    PART_OF = "part_of"  # finding part of result, method part of project
    PRECEDES = "precedes"  # method precedes method (in workflow)
    FAILED_WITH = "failed_with"  # idea failed with method (negative relationship)
    SUCCEEDED_WITH = "succeeded_with"  # idea succeeded with method (positive relationship)


class ResearchNode(BaseModel):
    """
    A node in the research knowledge graph.
    
    Represents a single entity (idea, method, dataset, etc.) with its
    content, metadata, and embedding for semantic search.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the node")
    node_type: NodeType = Field(description="Type of the node")
    content: str = Field(description="Main content/text of the node")
    title: Optional[str] = Field(default=None, description="Title or name of the node")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    # Embedding for semantic search
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding for semantic search")
    embedding_model: Optional[str] = Field(default=None, description="Model used to generate embedding")
    
    # Project association
    project_id: Optional[UUID] = Field(default=None, description="ID of the project this node belongs to")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    def update_timestamp(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now()
    
    def set_embedding(self, embedding: List[float], model: str) -> None:
        """
        Set the embedding for this node.
        
        Args:
            embedding: Vector embedding as a list of floats
            model: Name/identifier of the embedding model used
        """
        self.embedding = embedding
        self.embedding_model = model
        self.update_timestamp()


class ResearchEdge(BaseModel):
    """
    An edge (relationship) between two nodes in the knowledge graph.
    
    Represents a relationship such as "idea uses dataset" or "method generates result".
    """
    
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the edge")
    source_id: UUID = Field(description="ID of the source node")
    target_id: UUID = Field(description="ID of the target node")
    edge_type: EdgeType = Field(description="Type of relationship")
    
    # Edge metadata
    weight: float = Field(default=1.0, description="Weight/strength of the relationship (0.0 to 1.0)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional edge metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    
    # Project association
    project_id: Optional[UUID] = Field(default=None, description="ID of the project this edge belongs to")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    def __hash__(self) -> int:
        """Make edge hashable for use in sets."""
        return hash((self.source_id, self.target_id, self.edge_type))
    
    def __eq__(self, other: object) -> bool:
        """Equality based on source, target, and edge type."""
        if not isinstance(other, ResearchEdge):
            return False
        return (self.source_id == other.source_id and
                self.target_id == other.target_id and
                self.edge_type == other.edge_type)


class ProjectGraph(BaseModel):
    """
    A complete graph representing a single research project.
    
    Contains all nodes and edges for one project, along with project-level metadata.
    """
    
    project_id: UUID = Field(default_factory=uuid4, description="Unique identifier for the project")
    project_name: Optional[str] = Field(default=None, description="Name of the project")
    
    # Graph structure
    nodes: Dict[UUID, ResearchNode] = Field(default_factory=dict, description="Nodes in the project graph")
    edges: List[ResearchEdge] = Field(default_factory=list, description="Edges in the project graph")
    
    # Project metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Project-level metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="Project creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    # Success metrics
    success: Optional[bool] = Field(default=None, description="Whether the project was successful")
    quality_score: Optional[float] = Field(default=None, description="Quality score (0.0 to 1.0)")
    execution_time: Optional[float] = Field(default=None, description="Total execution time in seconds")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    def add_node(self, node: ResearchNode) -> None:
        """
        Add a node to the project graph.
        
        Args:
            node: The node to add
        """
        node.project_id = self.project_id
        self.nodes[node.id] = node
        self.update_timestamp()
    
    def add_edge(self, edge: ResearchEdge) -> None:
        """
        Add an edge to the project graph.
        
        Args:
            edge: The edge to add
        """
        # Validate that source and target nodes exist
        if edge.source_id not in self.nodes:
            raise ValueError(f"Source node {edge.source_id} not found in graph")
        if edge.target_id not in self.nodes:
            raise ValueError(f"Target node {edge.target_id} not found in graph")
        
        edge.project_id = self.project_id
        
        # Avoid duplicate edges
        if edge not in self.edges:
            self.edges.append(edge)
            self.update_timestamp()
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[ResearchNode]:
        """
        Get all nodes of a specific type.
        
        Args:
            node_type: The type of nodes to retrieve
            
        Returns:
            List of nodes matching the type
        """
        return [node for node in self.nodes.values() if node.node_type == node_type]
    
    def get_edges_by_type(self, edge_type: EdgeType) -> List[ResearchEdge]:
        """
        Get all edges of a specific type.
        
        Args:
            edge_type: The type of edges to retrieve
            
        Returns:
            List of edges matching the type
        """
        return [edge for edge in self.edges if edge.edge_type == edge_type]
    
    def get_neighbors(self, node_id: UUID, edge_type: Optional[EdgeType] = None) -> List[UUID]:
        """
        Get neighbor node IDs for a given node.
        
        Args:
            node_id: The node to get neighbors for
            edge_type: Optional filter by edge type
            
        Returns:
            List of neighbor node IDs
        """
        neighbors = []
        for edge in self.edges:
            if edge.source_id == node_id:
                if edge_type is None or edge.edge_type == edge_type:
                    neighbors.append(edge.target_id)
            elif edge.target_id == node_id:
                if edge_type is None or edge.edge_type == edge_type:
                    neighbors.append(edge.source_id)
        return neighbors
    
    def update_timestamp(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now()


class KnowledgeGraph(BaseModel):
    """
    Global knowledge graph storing all research projects.
    
    This is the main container for the entire knowledge base, containing
    multiple project graphs and providing query capabilities.
    """
    
    graph_id: UUID = Field(default_factory=uuid4, description="Unique identifier for the knowledge graph")
    projects: Dict[UUID, ProjectGraph] = Field(default_factory=dict, description="All project graphs")
    
    # Global metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Global graph metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="Graph creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    def add_project(self, project: ProjectGraph) -> None:
        """
        Add a project graph to the knowledge graph.
        
        Args:
            project: The project graph to add
        """
        self.projects[project.project_id] = project
        self.update_timestamp()
    
    def get_project(self, project_id: UUID) -> Optional[ProjectGraph]:
        """
        Get a project graph by ID.
        
        Args:
            project_id: The ID of the project to retrieve
            
        Returns:
            The project graph, or None if not found
        """
        return self.projects.get(project_id)
    
    def get_all_nodes(self, node_type: Optional[NodeType] = None) -> List[ResearchNode]:
        """
        Get all nodes across all projects, optionally filtered by type.
        
        Args:
            node_type: Optional filter by node type
            
        Returns:
            List of all matching nodes
        """
        all_nodes = []
        for project in self.projects.values():
            if node_type is None:
                all_nodes.extend(project.nodes.values())
            else:
                all_nodes.extend(project.get_nodes_by_type(node_type))
        return all_nodes
    
    def get_all_edges(self, edge_type: Optional[EdgeType] = None) -> List[ResearchEdge]:
        """
        Get all edges across all projects, optionally filtered by type.
        
        Args:
            edge_type: Optional filter by edge type
            
        Returns:
            List of all matching edges
        """
        all_edges = []
        for project in self.projects.values():
            if edge_type is None:
                all_edges.extend(project.edges)
            else:
                all_edges.extend(project.get_edges_by_type(edge_type))
        return all_edges
    
    def update_timestamp(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now()

