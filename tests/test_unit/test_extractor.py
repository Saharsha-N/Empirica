"""Unit tests for knowledge extractor."""
import pytest
from empirica.research import Research
from empirica.knowledge_graph.extractor import KnowledgeExtractor
from empirica.knowledge_graph.models import NodeType, EdgeType


def test_extract_from_research_creates_complete_graph(extractor, sample_research):
    """Test extract_from_research creates complete graph."""
    project = extractor.extract_from_research(
        sample_research,
        project_name="Test Project"
    )
    
    assert project is not None
    assert len(project.nodes) > 0
    assert len(project.edges) > 0
    
    # Should have at least idea, method, result nodes
    idea_nodes = project.get_nodes_by_type(NodeType.IDEA)
    method_nodes = project.get_nodes_by_type(NodeType.METHOD)
    result_nodes = project.get_nodes_by_type(NodeType.RESULT)
    
    assert len(idea_nodes) > 0
    assert len(method_nodes) > 0
    assert len(result_nodes) > 0


def test_extract_idea_creates_idea_node(extractor, sample_research):
    """Test _extract_idea creates IDEA node correctly."""
    project_id = extractor.extract_from_research(
        sample_research,
        project_name="Test"
    ).project_id
    
    idea_node = extractor._extract_idea(sample_research.idea, project_id)
    
    assert idea_node.node_type == NodeType.IDEA
    assert idea_node.content == sample_research.idea
    assert idea_node.project_id == project_id
    assert "text_length" in idea_node.metadata


def test_extract_method_creates_method_node(extractor, sample_research):
    """Test _extract_method creates METHOD node with steps."""
    project_id = extractor.extract_from_research(
        sample_research,
        project_name="Test"
    ).project_id
    
    method_node = extractor._extract_method(sample_research.methodology, project_id)
    
    assert method_node.node_type == NodeType.METHOD
    assert method_node.content == sample_research.methodology
    assert method_node.project_id == project_id
    assert "num_steps" in method_node.metadata
    assert "steps" in method_node.metadata


def test_extract_datasets_finds_dataset_nodes(extractor, sample_research):
    """Test _extract_datasets finds dataset nodes."""
    project_id = extractor.extract_from_research(
        sample_research,
        project_name="Test"
    ).project_id
    
    dataset_nodes = extractor._extract_datasets(sample_research.data_description, project_id)
    
    assert len(dataset_nodes) > 0
    assert all(node.node_type == NodeType.DATASET for node in dataset_nodes)
    # Should find .csv and .h5 files mentioned in data_description
    dataset_contents = [node.content for node in dataset_nodes]
    assert any(".csv" in content or ".h5" in content for content in dataset_contents)


def test_extract_tools_identifies_tools(extractor, sample_research):
    """Test _extract_tools identifies tools from text."""
    project_id = extractor.extract_from_research(
        sample_research,
        project_name="Test"
    ).project_id
    
    # Extract from data description
    tool_nodes = extractor._extract_tools(sample_research.data_description, project_id)
    
    assert len(tool_nodes) > 0
    assert all(node.node_type == NodeType.TOOL for node in tool_nodes)
    
    # Should find pandas and numpy mentioned in data_description
    tool_contents = [node.content.lower() for node in tool_nodes]
    assert "pandas" in tool_contents or "numpy" in tool_contents


def test_extract_result_creates_result_node(extractor, sample_research):
    """Test _extract_result creates RESULT node with metadata."""
    project_id = extractor.extract_from_research(
        sample_research,
        project_name="Test"
    ).project_id
    
    result_node = extractor._extract_result(
        sample_research.results,
        sample_research.plot_paths,
        project_id
    )
    
    assert result_node.node_type == NodeType.RESULT
    assert result_node.content == sample_research.results
    assert result_node.project_id == project_id
    assert result_node.metadata["num_plots"] == len(sample_research.plot_paths)
    assert result_node.metadata["plot_paths"] == sample_research.plot_paths


def test_extract_findings_extracts_finding_nodes(extractor, sample_research):
    """Test _extract_findings extracts finding nodes."""
    project_id = extractor.extract_from_research(
        sample_research,
        project_name="Test"
    ).project_id
    
    finding_nodes = extractor._extract_findings(sample_research.results, project_id)
    
    # Should find findings with key phrases like "we found", "results show"
    assert len(finding_nodes) > 0
    assert all(node.node_type == NodeType.FINDING for node in finding_nodes)
    assert all(len(node.content) > 20 for node in finding_nodes)  # Reasonable sentence length


def test_extract_creates_edges_between_nodes(extractor, sample_research):
    """Test edge creation between nodes."""
    project = extractor.extract_from_research(
        sample_research,
        project_name="Test"
    )
    
    # Should have edges connecting idea -> method, method -> result, etc.
    assert len(project.edges) > 0
    
    # Check for idea -> method edge
    idea_nodes = project.get_nodes_by_type(NodeType.IDEA)
    method_nodes = project.get_nodes_by_type(NodeType.METHOD)
    
    if idea_nodes and method_nodes:
        generates_edges = project.get_edges_by_type(EdgeType.GENERATES)
        assert len(generates_edges) > 0


def test_extract_with_empty_research(extractor):
    """Test extraction with empty research object."""
    empty_research = Research()
    project = extractor.extract_from_research(empty_research)
    
    # Should still create a project, but with minimal nodes
    assert project is not None
    assert len(project.nodes) == 0
    assert len(project.edges) == 0


def test_extract_with_partial_research(extractor):
    """Test extraction with partial research data."""
    partial_research = Research(
        data_description="Test data",
        idea="Test idea"
    )
    project = extractor.extract_from_research(partial_research)
    
    # Should have idea and dataset nodes, but no method/result
    idea_nodes = project.get_nodes_by_type(NodeType.IDEA)
    assert len(idea_nodes) > 0
    
    method_nodes = project.get_nodes_by_type(NodeType.METHOD)
    assert len(method_nodes) == 0


def test_extract_domain_detection(extractor):
    """Test domain detection in idea extraction."""
    cosmology_research = Research(
        idea="Analyze cosmic microwave background radiation patterns in the universe"
    )
    project = extractor.extract_from_research(cosmology_research)
    
    idea_nodes = project.get_nodes_by_type(NodeType.IDEA)
    if idea_nodes:
        metadata = idea_nodes[0].metadata
        # Should detect cosmology domain
        assert "domain" in metadata
        assert metadata["domain"] == "cosmology"


def test_extract_tool_deduplication(extractor):
    """Test that tools are deduplicated."""
    research = Research(
        data_description="Use pandas for data analysis. Also use pandas for visualization. Use numpy for calculations.",
        methodology="import pandas\nimport numpy\nimport pandas as pd"
    )
    project = extractor.extract_from_research(research)
    
    tool_nodes = project.get_nodes_by_type(NodeType.TOOL)
    tool_contents = [node.content for node in tool_nodes]
    
    # pandas should appear only once despite multiple mentions
    pandas_count = tool_contents.count("pandas")
    assert pandas_count == 1


def test_extract_metadata_preservation(extractor, sample_research):
    """Test that project metadata is preserved."""
    metadata = {"custom_key": "custom_value", "test": True}
    project = extractor.extract_from_research(
        sample_research,
        project_name="Test",
        metadata=metadata
    )
    
    assert project.metadata["custom_key"] == "custom_value"
    assert project.metadata["test"] is True

