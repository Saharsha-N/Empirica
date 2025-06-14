"""Unit tests for knowledge graph models."""
import pytest
from uuid import uuid4
from datetime import datetime

from empirica.knowledge_graph.models import (
    ResearchNode, ResearchEdge, ProjectGraph, KnowledgeGraph,
    NodeType, EdgeType
)


def test_research_node_creation():
    """Test creating a ResearchNode."""
    node = ResearchNode(
        node_type=NodeType.IDEA,
        content="Test idea content",
        title="Test Idea"
    )
    assert node.node_type == NodeType.IDEA
    assert node.content == "Test idea content"
    assert node.title == "Test Idea"
    assert node.id is not None
    assert isinstance(node.created_at, datetime)


def test_research_node_embedding():
    """Test setting embeddings on a node."""
    node = ResearchNode(
        node_type=NodeType.IDEA,
        content="Test content"
    )
    embedding = [0.1, 0.2, 0.3] * 100  # Mock embedding
    node.set_embedding(embedding, "test-model")
    assert node.embedding == embedding
    assert node.embedding_model == "test-model"


def test_research_node_update_timestamp():
    """Test timestamp update functionality."""
    node = ResearchNode(node_type=NodeType.IDEA, content="Test")
    original_time = node.updated_at
    import time
    time.sleep(0.01)  # Small delay
    node.update_timestamp()
    assert node.updated_at > original_time


def test_research_edge_creation():
    """Test creating a ResearchEdge."""
    source_id = uuid4()
    target_id = uuid4()
    edge = ResearchEdge(
        source_id=source_id,
        target_id=target_id,
        edge_type=EdgeType.GENERATES
    )
    assert edge.source_id == source_id
    assert edge.target_id == target_id
    assert edge.edge_type == EdgeType.GENERATES
    assert edge.weight == 1.0


def test_research_edge_equality():
    """Test ResearchEdge equality and hashing."""
    source_id = uuid4()
    target_id = uuid4()
    edge1 = ResearchEdge(
        source_id=source_id,
        target_id=target_id,
        edge_type=EdgeType.GENERATES
    )
    edge2 = ResearchEdge(
        source_id=source_id,
        target_id=target_id,
        edge_type=EdgeType.GENERATES
    )
    assert edge1 == edge2
    assert hash(edge1) == hash(edge2)


def test_research_edge_different_types():
    """Test edges with different types are not equal."""
    source_id = uuid4()
    target_id = uuid4()
    edge1 = ResearchEdge(
        source_id=source_id,
        target_id=target_id,
        edge_type=EdgeType.GENERATES
    )
    edge2 = ResearchEdge(
        source_id=source_id,
        target_id=target_id,
        edge_type=EdgeType.USES
    )
    assert edge1 != edge2


def test_project_graph_add_node():
    """Test adding nodes to a ProjectGraph."""
    project = ProjectGraph(project_name="Test Project")
    node = ResearchNode(
        node_type=NodeType.IDEA,
        content="Test idea"
    )
    project.add_node(node)
    assert len(project.nodes) == 1
    assert node.id in project.nodes
    assert project.nodes[node.id].project_id == project.project_id


def test_project_graph_add_edge():
    """Test adding edges to a ProjectGraph."""
    project = ProjectGraph()
    idea_node = ResearchNode(node_type=NodeType.IDEA, content="Idea")
    method_node = ResearchNode(node_type=NodeType.METHOD, content="Method")
    
    project.add_node(idea_node)
    project.add_node(method_node)
    
    edge = ResearchEdge(
        source_id=idea_node.id,
        target_id=method_node.id,
        edge_type=EdgeType.GENERATES
    )
    project.add_edge(edge)
    
    assert len(project.edges) == 1
    assert edge in project.edges


def test_project_graph_add_edge_missing_node():
    """Test adding edge with missing node raises error."""
    project = ProjectGraph()
    idea_node = ResearchNode(node_type=NodeType.IDEA, content="Idea")
    project.add_node(idea_node)
    
    method_node = ResearchNode(node_type=NodeType.METHOD, content="Method")
    edge = ResearchEdge(
        source_id=idea_node.id,
        target_id=method_node.id,
        edge_type=EdgeType.GENERATES
    )
    
    with pytest.raises(ValueError, match="not found in graph"):
        project.add_edge(edge)


def test_project_graph_get_nodes_by_type():
    """Test filtering nodes by type."""
    project = ProjectGraph()
    idea_node = ResearchNode(node_type=NodeType.IDEA, content="Idea")
    method_node = ResearchNode(node_type=NodeType.METHOD, content="Method")
    result_node = ResearchNode(node_type=NodeType.RESULT, content="Result")
    
    project.add_node(idea_node)
    project.add_node(method_node)
    project.add_node(result_node)
    
    idea_nodes = project.get_nodes_by_type(NodeType.IDEA)
    assert len(idea_nodes) == 1
    assert idea_nodes[0] == idea_node
    
    method_nodes = project.get_nodes_by_type(NodeType.METHOD)
    assert len(method_nodes) == 1
    assert method_nodes[0] == method_node


def test_project_graph_get_edges_by_type():
    """Test filtering edges by type."""
    project = ProjectGraph()
    idea_node = ResearchNode(node_type=NodeType.IDEA, content="Idea")
    method_node = ResearchNode(node_type=NodeType.METHOD, content="Method")
    result_node = ResearchNode(node_type=NodeType.RESULT, content="Result")
    
    project.add_node(idea_node)
    project.add_node(method_node)
    project.add_node(result_node)
    
    edge1 = ResearchEdge(
        source_id=idea_node.id,
        target_id=method_node.id,
        edge_type=EdgeType.GENERATES
    )
    edge2 = ResearchEdge(
        source_id=method_node.id,
        target_id=result_node.id,
        edge_type=EdgeType.LEADS_TO
    )
    
    project.add_edge(edge1)
    project.add_edge(edge2)
    
    generates_edges = project.get_edges_by_type(EdgeType.GENERATES)
    assert len(generates_edges) == 1
    assert generates_edges[0] == edge1


def test_project_graph_get_neighbors():
    """Test getting neighbor nodes."""
    project = ProjectGraph()
    idea_node = ResearchNode(node_type=NodeType.IDEA, content="Idea")
    method_node = ResearchNode(node_type=NodeType.METHOD, content="Method")
    tool_node = ResearchNode(node_type=NodeType.TOOL, content="pandas")
    
    project.add_node(idea_node)
    project.add_node(method_node)
    project.add_node(tool_node)
    
    edge1 = ResearchEdge(
        source_id=idea_node.id,
        target_id=method_node.id,
        edge_type=EdgeType.GENERATES
    )
    edge2 = ResearchEdge(
        source_id=method_node.id,
        target_id=tool_node.id,
        edge_type=EdgeType.USES
    )
    
    project.add_edge(edge1)
    project.add_edge(edge2)
    
    neighbors = project.get_neighbors(idea_node.id)
    assert method_node.id in neighbors
    
    neighbors_filtered = project.get_neighbors(method_node.id, EdgeType.USES)
    assert tool_node.id in neighbors_filtered


def test_project_graph_duplicate_edges():
    """Test that duplicate edges are not added."""
    project = ProjectGraph()
    idea_node = ResearchNode(node_type=NodeType.IDEA, content="Idea")
    method_node = ResearchNode(node_type=NodeType.METHOD, content="Method")
    
    project.add_node(idea_node)
    project.add_node(method_node)
    
    edge = ResearchEdge(
        source_id=idea_node.id,
        target_id=method_node.id,
        edge_type=EdgeType.GENERATES
    )
    
    project.add_edge(edge)
    project.add_edge(edge)  # Try to add again
    
    assert len(project.edges) == 1


def test_knowledge_graph_add_project():
    """Test adding projects to KnowledgeGraph."""
    kg = KnowledgeGraph()
    project = ProjectGraph(project_name="Test Project")
    
    kg.add_project(project)
    assert project.project_id in kg.projects
    assert kg.projects[project.project_id] == project


def test_knowledge_graph_get_project():
    """Test retrieving projects from KnowledgeGraph."""
    kg = KnowledgeGraph()
    project = ProjectGraph(project_name="Test Project")
    
    kg.add_project(project)
    retrieved = kg.get_project(project.project_id)
    assert retrieved == project
    
    non_existent = kg.get_project(uuid4())
    assert non_existent is None


def test_knowledge_graph_get_all_nodes():
    """Test getting all nodes across projects."""
    kg = KnowledgeGraph()
    
    project1 = ProjectGraph(project_name="Project 1")
    node1 = ResearchNode(node_type=NodeType.IDEA, content="Idea 1")
    project1.add_node(node1)
    kg.add_project(project1)
    
    project2 = ProjectGraph(project_name="Project 2")
    node2 = ResearchNode(node_type=NodeType.IDEA, content="Idea 2")
    project2.add_node(node2)
    kg.add_project(project2)
    
    all_nodes = kg.get_all_nodes()
    assert len(all_nodes) == 2
    
    idea_nodes = kg.get_all_nodes(NodeType.IDEA)
    assert len(idea_nodes) == 2


def test_knowledge_graph_get_all_edges():
    """Test getting all edges across projects."""
    kg = KnowledgeGraph()
    
    project1 = ProjectGraph(project_name="Project 1")
    node1 = ResearchNode(node_type=NodeType.IDEA, content="Idea 1")
    node2 = ResearchNode(node_type=NodeType.METHOD, content="Method 1")
    project1.add_node(node1)
    project1.add_node(node2)
    edge1 = ResearchEdge(
        source_id=node1.id,
        target_id=node2.id,
        edge_type=EdgeType.GENERATES
    )
    project1.add_edge(edge1)
    kg.add_project(project1)
    
    all_edges = kg.get_all_edges()
    assert len(all_edges) == 1
    
    generates_edges = kg.get_all_edges(EdgeType.GENERATES)
    assert len(generates_edges) == 1


def test_node_type_enum():
    """Test NodeType enum values."""
    assert NodeType.IDEA == "idea"
    assert NodeType.METHOD == "method"
    assert NodeType.DATASET == "dataset"
    assert NodeType.TOOL == "tool"
    assert NodeType.RESULT == "result"
    assert NodeType.FINDING == "finding"


def test_edge_type_enum():
    """Test EdgeType enum values."""
    assert EdgeType.USES == "uses"
    assert EdgeType.GENERATES == "generates"
    assert EdgeType.LEADS_TO == "leads_to"
    assert EdgeType.SIMILAR_TO == "similar_to"
    assert EdgeType.CITES == "cites"
    assert EdgeType.PART_OF == "part_of"

