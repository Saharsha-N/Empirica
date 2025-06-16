"""Unit tests for graph query."""
import pytest
import math
from uuid import uuid4

from empirica.knowledge_graph.query import GraphQuery
from empirica.knowledge_graph.models import NodeType, EdgeType


def test_cosine_similarity_calculation():
    """Test cosine similarity calculation."""
    storage = GraphQuery.__new__(GraphQuery)  # Create instance without __init__
    storage.storage = None  # Not needed for this test
    
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]
    similarity = storage.cosine_similarity(vec1, vec2)
    assert abs(similarity - 1.0) < 0.001  # Should be 1.0 (identical)
    
    vec3 = [1.0, 0.0, 0.0]
    vec4 = [0.0, 1.0, 0.0]
    similarity = storage.cosine_similarity(vec3, vec4)
    assert abs(similarity - 0.0) < 0.001  # Should be 0.0 (orthogonal)
    
    vec5 = [1.0, 1.0, 0.0]
    vec6 = [1.0, 0.0, 0.0]
    similarity = storage.cosine_similarity(vec5, vec6)
    assert 0.0 < similarity < 1.0  # Should be between 0 and 1


def test_cosine_similarity_empty_vectors():
    """Test cosine similarity with empty vectors."""
    storage = GraphQuery.__new__(GraphQuery)
    storage.storage = None
    
    similarity = storage.cosine_similarity([], [])
    assert similarity == 0.0
    
    similarity = storage.cosine_similarity([1.0, 2.0], [])
    assert similarity == 0.0


def test_cosine_similarity_different_lengths():
    """Test cosine similarity with different length vectors."""
    storage = GraphQuery.__new__(GraphQuery)
    storage.storage = None
    
    similarity = storage.cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])
    assert similarity == 0.0  # Should return 0 for mismatched lengths


def test_find_similar_ideas_with_embeddings(storage, sample_research):
    """Test find_similar_ideas with embeddings."""
    from empirica.knowledge_graph.extractor import KnowledgeExtractor
    from empirica.knowledge_graph.query import GraphQuery
    
    # Create extractor with embeddings disabled for speed
    extractor = KnowledgeExtractor(generate_embeddings=False)
    project = extractor.extract_from_research(sample_research)
    storage.save_project(project)
    
    query = GraphQuery(storage)
    
    # Add embedding manually to idea node for testing
    idea_nodes = project.get_nodes_by_type(NodeType.IDEA)
    if idea_nodes:
        idea_node = idea_nodes[0]
        test_embedding = [0.1] * 384  # Mock embedding
        idea_node.set_embedding(test_embedding, "test-model")
        storage.save_project(project)
    
    # Search with embedding
    similar = query.find_similar_ideas(
        "Test research idea about analyzing time-series data",
        idea_embedding=test_embedding,
        top_k=5
    )
    
    # Should find at least the project we just created
    assert len(similar) > 0


def test_find_similar_ideas_fallback_text_matching(storage, sample_research):
    """Test find_similar_ideas fallback to text matching."""
    from empirica.knowledge_graph.extractor import KnowledgeExtractor
    from empirica.knowledge_graph.query import GraphQuery
    
    extractor = KnowledgeExtractor(generate_embeddings=False)
    project = extractor.extract_from_research(sample_research)
    storage.save_project(project)
    
    query = GraphQuery(storage)
    
    # Search without embedding (should use text matching)
    similar = query.find_similar_ideas(
        "analyzing time-series data patterns",
        idea_embedding=None,
        top_k=5,
        threshold=0.1  # Lower threshold for text matching
    )
    
    # Should find similar ideas based on keyword overlap
    assert isinstance(similar, list)


def test_find_similar_methods(storage, sample_research):
    """Test find_similar_methods."""
    from empirica.knowledge_graph.extractor import KnowledgeExtractor
    from empirica.knowledge_graph.query import GraphQuery
    
    extractor = KnowledgeExtractor(generate_embeddings=False)
    project = extractor.extract_from_research(sample_research)
    storage.save_project(project)
    
    query = GraphQuery(storage)
    
    similar = query.find_similar_methods(
        "Load data using pandas and analyze with numpy",
        method_embedding=None,
        top_k=5,
        threshold=0.1
    )
    
    assert isinstance(similar, list)


def test_get_successful_patterns(storage, sample_research):
    """Test get_successful_patterns."""
    from empirica.knowledge_graph.extractor import KnowledgeExtractor
    from empirica.knowledge_graph.query import GraphQuery
    
    extractor = KnowledgeExtractor(generate_embeddings=False)
    
    # Create multiple projects with different success statuses
    for i in range(3):
        project = extractor.extract_from_research(sample_research, project_name=f"Project {i}")
        project.success = (i % 2 == 0)  # Alternate success
        project.quality_score = 0.8 if project.success else 0.3
        storage.save_project(project)
    
    query = GraphQuery(storage)
    patterns = query.get_successful_patterns()
    
    assert "total_successful" in patterns
    assert patterns["total_successful"] > 0
    assert "patterns" in patterns or "common_tools" in patterns


def test_get_failed_patterns(storage, sample_research):
    """Test get_failed_patterns."""
    from empirica.knowledge_graph.extractor import KnowledgeExtractor
    from empirica.knowledge_graph.query import GraphQuery
    
    extractor = KnowledgeExtractor(generate_embeddings=False)
    
    # Create a failed project
    project = extractor.extract_from_research(sample_research)
    project.success = False
    storage.save_project(project)
    
    query = GraphQuery(storage)
    patterns = query.get_failed_patterns()
    
    assert "total_failed" in patterns
    assert isinstance(patterns["total_failed"], int)


def test_find_related_projects(storage, sample_research):
    """Test find_related_projects."""
    from empirica.knowledge_graph.extractor import KnowledgeExtractor
    from empirica.knowledge_graph.query import GraphQuery
    
    extractor = KnowledgeExtractor(generate_embeddings=False)
    
    # Create multiple projects
    projects = []
    for i in range(3):
        project = extractor.extract_from_research(sample_research, project_name=f"Project {i}")
        storage.save_project(project)
        projects.append(project)
    
    query = GraphQuery(storage)
    related = query.find_related_projects(projects[0].project_id, max_depth=2)
    
    assert isinstance(related, list)
    # Should find other projects that share nodes


def test_traverse_graph_bfs(storage, sample_research):
    """Test traverse_graph BFS algorithm."""
    from empirica.knowledge_graph.extractor import KnowledgeExtractor
    from empirica.knowledge_graph.query import GraphQuery
    
    extractor = KnowledgeExtractor(generate_embeddings=False)
    project = extractor.extract_from_research(sample_research)
    storage.save_project(project)
    
    query = GraphQuery(storage)
    
    # Get a starting node
    idea_nodes = project.get_nodes_by_type(NodeType.IDEA)
    if idea_nodes:
        start_node_id = idea_nodes[0].id
        traversed = query.traverse_graph(start_node_id, max_depth=2)
        
        assert isinstance(traversed, list)
        assert len(traversed) > 0
        # Should include the start node
        assert any(node.id == start_node_id for node in traversed)


def test_traverse_graph_with_edge_filter(storage, sample_research):
    """Test traverse_graph with edge type filter."""
    from empirica.knowledge_graph.extractor import KnowledgeExtractor
    from empirica.knowledge_graph.query import GraphQuery
    
    extractor = KnowledgeExtractor(generate_embeddings=False)
    project = extractor.extract_from_research(sample_research)
    storage.save_project(project)
    
    query = GraphQuery(storage)
    
    idea_nodes = project.get_nodes_by_type(NodeType.IDEA)
    if idea_nodes:
        start_node_id = idea_nodes[0].id
        # Traverse only GENERATES edges
        traversed = query.traverse_graph(
            start_node_id,
            edge_type=EdgeType.GENERATES,
            max_depth=2
        )
        
        assert isinstance(traversed, list)


def test_find_similar_ideas_empty_graph(storage):
    """Test find_similar_ideas with empty knowledge graph."""
    from empirica.knowledge_graph.query import GraphQuery
    
    query = GraphQuery(storage)
    similar = query.find_similar_ideas("Test idea", top_k=5)
    
    assert isinstance(similar, list)
    assert len(similar) == 0


def test_get_successful_patterns_empty(storage):
    """Test get_successful_patterns with no successful projects."""
    from empirica.knowledge_graph.query import GraphQuery
    
    query = GraphQuery(storage)
    patterns = query.get_successful_patterns()
    
    assert "total_successful" in patterns
    assert patterns["total_successful"] == 0

