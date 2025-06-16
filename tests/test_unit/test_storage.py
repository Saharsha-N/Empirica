"""Unit tests for graph storage."""
import pytest
from uuid import uuid4
from pathlib import Path

from empirica.knowledge_graph.storage import GraphStorage
from empirica.knowledge_graph.models import (
    ProjectGraph, ResearchNode, ResearchEdge, NodeType, EdgeType
)


def test_storage_initialization(storage):
    """Test storage initialization creates database."""
    assert storage.db_path.exists()
    # Database should be initialized
    project_ids = storage.list_projects()
    assert isinstance(project_ids, list)


def test_storage_save_and_load_project(storage, sample_research):
    """Test saving and loading a project."""
    from empirica.knowledge_graph.extractor import KnowledgeExtractor
    
    extractor = KnowledgeExtractor(generate_embeddings=False)
    project = extractor.extract_from_research(
        sample_research,
        project_name="Test Project"
    )
    
    # Save project
    storage.save_project(project)
    
    # Load project
    loaded_project = storage.load_project(project.project_id)
    assert loaded_project is not None
    assert loaded_project.project_id == project.project_id
    assert loaded_project.project_name == "Test Project"
    assert len(loaded_project.nodes) > 0
    assert len(loaded_project.edges) > 0


def test_storage_list_projects(storage, sample_research):
    """Test listing all projects."""
    from empirica.knowledge_graph.extractor import KnowledgeExtractor
    
    extractor = KnowledgeExtractor(generate_embeddings=False)
    
    # Create and save multiple projects
    project_ids = []
    for i in range(3):
        project = extractor.extract_from_research(
            sample_research,
            project_name=f"Project {i}"
        )
        storage.save_project(project)
        project_ids.append(project.project_id)
    
    listed_ids = storage.list_projects()
    assert len(listed_ids) >= 3
    for pid in project_ids:
        assert pid in listed_ids


def test_storage_delete_project(storage, sample_research):
    """Test deleting a project."""
    from empirica.knowledge_graph.extractor import KnowledgeExtractor
    
    extractor = KnowledgeExtractor(generate_embeddings=False)
    project = extractor.extract_from_research(sample_research)
    
    storage.save_project(project)
    project_id = project.project_id
    
    # Verify project exists
    loaded = storage.load_project(project_id)
    assert loaded is not None
    
    # Delete project
    deleted = storage.delete_project(project_id)
    assert deleted is True
    
    # Verify deletion
    loaded = storage.load_project(project_id)
    assert loaded is None


def test_storage_delete_nonexistent_project(storage):
    """Test deleting a non-existent project."""
    fake_id = uuid4()
    deleted = storage.delete_project(fake_id)
    assert deleted is False


def test_storage_get_all_nodes(storage, sample_research):
    """Test getting all nodes with filters."""
    from empirica.knowledge_graph.extractor import KnowledgeExtractor
    
    extractor = KnowledgeExtractor(generate_embeddings=False)
    project = extractor.extract_from_research(sample_research)
    storage.save_project(project)
    
    # Get all nodes
    all_nodes = storage.get_all_nodes()
    assert len(all_nodes) > 0
    
    # Get nodes by type
    idea_nodes = storage.get_all_nodes(NodeType.IDEA)
    assert len(idea_nodes) > 0
    assert all(node.node_type == NodeType.IDEA for node in idea_nodes)
    
    method_nodes = storage.get_all_nodes(NodeType.METHOD)
    assert len(method_nodes) > 0
    assert all(node.node_type == NodeType.METHOD for node in method_nodes)


def test_storage_get_all_edges(storage, sample_research):
    """Test getting all edges with filters."""
    from empirica.knowledge_graph.extractor import KnowledgeExtractor
    
    extractor = KnowledgeExtractor(generate_embeddings=False)
    project = extractor.extract_from_research(sample_research)
    storage.save_project(project)
    
    # Get all edges
    all_edges = storage.get_all_edges()
    assert len(all_edges) > 0
    
    # Get edges by type
    generates_edges = storage.get_all_edges(EdgeType.GENERATES)
    assert len(generates_edges) > 0
    assert all(edge.edge_type == EdgeType.GENERATES for edge in generates_edges)


def test_storage_embedding_serialization(storage):
    """Test embedding serialization and deserialization."""
    from empirica.knowledge_graph.models import ProjectGraph, ResearchNode
    
    project = ProjectGraph(project_name="Embedding Test")
    node = ResearchNode(
        node_type=NodeType.IDEA,
        content="Test idea with embedding"
    )
    
    # Set embedding
    embedding = [0.1, 0.2, 0.3] * 100  # 300-dim embedding
    node.set_embedding(embedding, "test-model")
    project.add_node(node)
    
    # Save and load
    storage.save_project(project)
    loaded_project = storage.load_project(project.project_id)
    
    assert loaded_project is not None
    loaded_node = list(loaded_project.nodes.values())[0]
    assert loaded_node.embedding == embedding
    assert loaded_node.embedding_model == "test-model"


def test_storage_load_nonexistent_project(storage):
    """Test loading a non-existent project returns None."""
    fake_id = uuid4()
    project = storage.load_project(fake_id)
    assert project is None


def test_storage_multiple_projects_isolation(storage, sample_research):
    """Test that projects are stored and loaded independently."""
    from empirica.knowledge_graph.extractor import KnowledgeExtractor
    
    extractor = KnowledgeExtractor(generate_embeddings=False)
    
    # Create two projects
    project1 = extractor.extract_from_research(
        sample_research,
        project_name="Project 1"
    )
    project2 = extractor.extract_from_research(
        sample_research,
        project_name="Project 2"
    )
    
    storage.save_project(project1)
    storage.save_project(project2)
    
    # Load each independently
    loaded1 = storage.load_project(project1.project_id)
    loaded2 = storage.load_project(project2.project_id)
    
    assert loaded1.project_name == "Project 1"
    assert loaded2.project_name == "Project 2"
    assert loaded1.project_id != loaded2.project_id


def test_storage_update_existing_project(storage, sample_research):
    """Test updating an existing project."""
    from empirica.knowledge_graph.extractor import KnowledgeExtractor
    
    extractor = KnowledgeExtractor(generate_embeddings=False)
    project = extractor.extract_from_research(
        sample_research,
        project_name="Original Name"
    )
    
    storage.save_project(project)
    
    # Update project
    project.project_name = "Updated Name"
    project.success = True
    project.quality_score = 0.85
    storage.save_project(project)
    
    # Load and verify update
    loaded = storage.load_project(project.project_id)
    assert loaded.project_name == "Updated Name"
    assert loaded.success is True
    assert loaded.quality_score == 0.85

