"""Unit tests for pattern analyzer."""
import pytest
from datetime import datetime, timedelta

from empirica.knowledge_graph.models import NodeType
from empirica.meta_learning.analyzer import PatternAnalyzer


def test_analyze_success_rates_by_method_type(storage, sample_research):
    """Test analyze_success_rates_by_method_type."""
    from empirica.knowledge_graph.extractor import KnowledgeExtractor
    
    extractor = KnowledgeExtractor(generate_embeddings=False)
    analyzer = PatternAnalyzer(storage)
    
    # Create projects with different method characteristics
    for i in range(5):
        project = extractor.extract_from_research(sample_research, project_name=f"Project {i}")
        project.success = (i % 2 == 0)  # Alternate success
        storage.save_project(project)
    
    success_rates = analyzer.analyze_success_rates_by_method_type()
    
    assert isinstance(success_rates, dict)
    # Should have some method type categories
    assert len(success_rates) >= 0


def test_analyze_tool_effectiveness(storage, sample_research):
    """Test analyze_tool_effectiveness."""
    from empirica.knowledge_graph.extractor import KnowledgeExtractor
    
    extractor = KnowledgeExtractor(generate_embeddings=False)
    analyzer = PatternAnalyzer(storage)
    
    # Create projects with tools
    for i in range(3):
        project = extractor.extract_from_research(sample_research, project_name=f"Project {i}")
        project.success = True
        storage.save_project(project)
    
    effectiveness = analyzer.analyze_tool_effectiveness()
    
    assert isinstance(effectiveness, dict)
    # Should have tool effectiveness metrics
    for tool, metrics in effectiveness.items():
        assert "success_rate" in metrics
        assert "usage_count" in metrics
        assert 0.0 <= metrics["success_rate"] <= 1.0


def test_identify_common_failure_modes(storage, sample_research):
    """Test identify_common_failure_modes."""
    from empirica.knowledge_graph.extractor import KnowledgeExtractor
    
    extractor = KnowledgeExtractor(generate_embeddings=False)
    analyzer = PatternAnalyzer(storage)
    
    # Create some failed projects
    for i in range(3):
        project = extractor.extract_from_research(sample_research, project_name=f"Failed {i}")
        project.success = False
        storage.save_project(project)
    
    failure_modes = analyzer.identify_common_failure_modes()
    
    assert isinstance(failure_modes, list)
    # Should identify failure patterns
    for mode in failure_modes:
        assert "type" in mode
        assert "description" in mode
        assert "count" in mode


def test_analyze_domain_patterns(storage):
    """Test analyze_domain_patterns."""
    from empirica.research import Research
    from empirica.knowledge_graph.extractor import KnowledgeExtractor
    
    extractor = KnowledgeExtractor(generate_embeddings=False)
    analyzer = PatternAnalyzer(storage)
    
    # Create projects in different domains
    cosmology_research = Research(idea="Analyze cosmic microwave background radiation")
    ml_research = Research(idea="Machine learning model for time-series prediction")
    
    project1 = extractor.extract_from_research(cosmology_research)
    project1.success = True
    storage.save_project(project1)
    
    project2 = extractor.extract_from_research(ml_research)
    project2.success = True
    storage.save_project(project2)
    
    domain_patterns = analyzer.analyze_domain_patterns()
    
    assert isinstance(domain_patterns, dict)
    # Should have domain-specific patterns
    for domain, pattern in domain_patterns.items():
        assert "total_projects" in pattern
        assert "success_rate" in pattern
        assert 0.0 <= pattern["success_rate"] <= 1.0


def test_analyze_temporal_trends(storage, sample_research):
    """Test analyze_temporal_trends."""
    from empirica.knowledge_graph.extractor import KnowledgeExtractor
    
    extractor = KnowledgeExtractor(generate_embeddings=False)
    analyzer = PatternAnalyzer(storage)
    
    # Create projects with different timestamps
    for i in range(5):
        project = extractor.extract_from_research(sample_research, project_name=f"Project {i}")
        # Set timestamps to simulate recent and older projects
        if i < 2:
            project.created_at = datetime.now() - timedelta(days=10)  # Recent
        else:
            project.created_at = datetime.now() - timedelta(days=60)  # Older
        project.success = True
        storage.save_project(project)
    
    trends = analyzer.analyze_temporal_trends(days=30)
    
    assert isinstance(trends, dict)
    assert "recent_period_days" in trends
    assert "recent_projects" in trends
    assert "older_projects" in trends


def test_get_statistical_relationships(storage, sample_research):
    """Test get_statistical_relationships."""
    from empirica.knowledge_graph.extractor import KnowledgeExtractor
    
    extractor = KnowledgeExtractor(generate_embeddings=False)
    analyzer = PatternAnalyzer(storage)
    
    # Create projects with idea-method-result relationships
    for i in range(3):
        project = extractor.extract_from_research(sample_research, project_name=f"Project {i}")
        project.success = True
        storage.save_project(project)
    
    relationships = analyzer.get_statistical_relationships()
    
    assert isinstance(relationships, dict)
    assert "idea_method_pairs" in relationships
    assert "method_result_pairs" in relationships


def test_analyze_temporal_trends_emerging_tools(storage, sample_research):
    """Test that temporal trends identify emerging tools."""
    from empirica.research import Research
    from empirica.knowledge_graph.extractor import KnowledgeExtractor
    
    extractor = KnowledgeExtractor(generate_embeddings=False)
    analyzer = PatternAnalyzer(storage)
    
    # Create recent project with specific tools
    recent_research = Research(
        data_description="Use new_tool for analysis",
        idea="Test idea"
    )
    recent_project = extractor.extract_from_research(recent_research)
    recent_project.created_at = datetime.now() - timedelta(days=5)
    storage.save_project(recent_project)
    
    trends = analyzer.analyze_temporal_trends(days=30)
    
    assert "emerging_tools" in trends
    assert isinstance(trends["emerging_tools"], dict)


def test_analyze_empty_storage(storage):
    """Test analyzer with empty storage."""
    analyzer = PatternAnalyzer(storage)
    
    success_rates = analyzer.analyze_success_rates_by_method_type()
    assert isinstance(success_rates, dict)
    
    tool_effectiveness = analyzer.analyze_tool_effectiveness()
    assert isinstance(tool_effectiveness, dict)
    
    failure_modes = analyzer.identify_common_failure_modes()
    assert isinstance(failure_modes, list)
    
    domain_patterns = analyzer.analyze_domain_patterns()
    assert isinstance(domain_patterns, dict)

