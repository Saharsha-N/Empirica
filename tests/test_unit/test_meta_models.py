"""Unit tests for meta-learning models."""
import pytest

from empirica.meta_learning.models import MetaLearningModels


def test_predict_success_probability(meta_agent, sample_research):
    """Test predict_success_probability."""
    models = meta_agent.models
    
    probability = models.predict_success_probability(
        idea_text=sample_research.idea,
        method_text=sample_research.methodology,
        tools=["pandas", "numpy"]
    )
    
    assert 0.0 <= probability <= 1.0


def test_predict_success_probability_with_domain(meta_agent):
    """Test predict_success_probability with domain."""
    models = meta_agent.models
    
    probability = models.predict_success_probability(
        idea_text="Analyze cosmic microwave background radiation",
        domain="cosmology"
    )
    
    assert 0.0 <= probability <= 1.0


def test_recommend_methods(meta_agent, sample_research):
    """Test recommend_methods."""
    from empirica.knowledge_graph.extractor import KnowledgeExtractor
    
    # Create a successful project to base recommendations on
    extractor = KnowledgeExtractor(generate_embeddings=False)
    project = extractor.extract_from_research(sample_research)
    project.success = True
    meta_agent.storage.save_project(project)
    
    models = meta_agent.models
    recommendations = models.recommend_methods(
        idea_text=sample_research.idea,
        top_k=3
    )
    
    assert isinstance(recommendations, list)
    for method_text, confidence in recommendations:
        assert isinstance(method_text, str)
        assert 0.0 <= confidence <= 1.0


def test_recommend_methods_empty_graph(meta_agent):
    """Test recommend_methods with empty knowledge graph."""
    models = meta_agent.models
    recommendations = models.recommend_methods("Test idea", top_k=5)
    
    assert isinstance(recommendations, list)
    # Should return empty list if no similar projects


def test_predict_failure(meta_agent, sample_research):
    """Test predict_failure."""
    models = meta_agent.models
    
    likely_to_fail, reasons = models.predict_failure(
        idea_text=sample_research.idea,
        method_text=sample_research.methodology,
        tools=["pandas"]
    )
    
    assert isinstance(likely_to_fail, bool)
    assert isinstance(reasons, list)
    assert all(isinstance(reason, str) for reason in reasons)


def test_estimate_execution_time(meta_agent, sample_research):
    """Test estimate_execution_time."""
    from empirica.knowledge_graph.extractor import KnowledgeExtractor
    
    # Create project with execution time
    extractor = KnowledgeExtractor(generate_embeddings=False)
    project = extractor.extract_from_research(sample_research)
    project.execution_time = 45.0  # 45 minutes
    meta_agent.storage.save_project(project)
    
    models = meta_agent.models
    estimated_time, confidence_range = models.estimate_execution_time(
        method_text=sample_research.methodology
    )
    
    assert estimated_time > 0
    assert confidence_range >= 0
    assert isinstance(estimated_time, float)
    assert isinstance(confidence_range, float)


def test_estimate_execution_time_no_similar(meta_agent):
    """Test estimate_execution_time with no similar methods."""
    models = meta_agent.models
    estimated_time, confidence_range = models.estimate_execution_time(
        method_text="Completely unique method that doesn't exist"
    )
    
    # Should return default estimate
    assert estimated_time > 0
    assert confidence_range >= 0


def test_predict_quality_score(meta_agent, sample_research):
    """Test predict_quality_score."""
    models = meta_agent.models
    
    quality = models.predict_quality_score(
        idea_text=sample_research.idea,
        method_text=sample_research.methodology,
        tools=["pandas", "numpy"]
    )
    
    assert 0.0 <= quality <= 1.0


def test_pattern_caching(meta_agent, sample_research):
    """Test that patterns are cached for performance."""
    from empirica.knowledge_graph.extractor import KnowledgeExtractor
    
    extractor = KnowledgeExtractor(generate_embeddings=False)
    project = extractor.extract_from_research(sample_research)
    project.success = True
    meta_agent.storage.save_project(project)
    
    models = meta_agent.models
    
    # First call should cache patterns
    models._cache_patterns()
    assert hasattr(models, '_tool_effectiveness')
    assert hasattr(models, '_success_rates')
    assert hasattr(models, '_domain_patterns')
    
    # Second call should use cached patterns
    effectiveness1 = models._tool_effectiveness
    models._cache_patterns()
    effectiveness2 = models._tool_effectiveness
    
    # Should be the same (cached)
    assert effectiveness1 == effectiveness2


def test_predict_success_with_tool_effectiveness(meta_agent, sample_research):
    """Test success prediction considers tool effectiveness."""
    from empirica.knowledge_graph.extractor import KnowledgeExtractor
    
    # Create projects using specific tools
    extractor = KnowledgeExtractor(generate_embeddings=False)
    for i in range(3):
        project = extractor.extract_from_research(sample_research)
        project.success = True  # All successful
        meta_agent.storage.save_project(project)
    
    models = meta_agent.models
    models._cache_patterns()
    
    # Predict with tools that were used in successful projects
    probability = models.predict_success_probability(
        idea_text=sample_research.idea,
        tools=["pandas", "numpy"]  # Tools from sample_research
    )
    
    assert 0.0 <= probability <= 1.0


def test_recommend_methods_confidence_scores(meta_agent, sample_research):
    """Test that recommended methods have confidence scores."""
    from empirica.knowledge_graph.extractor import KnowledgeExtractor
    
    extractor = KnowledgeExtractor(generate_embeddings=False)
    project = extractor.extract_from_research(sample_research)
    project.success = True
    meta_agent.storage.save_project(project)
    
    models = meta_agent.models
    recommendations = models.recommend_methods(sample_research.idea, top_k=5)
    
    if recommendations:
        # Check that recommendations are sorted by confidence
        confidences = [conf for _, conf in recommendations]
        assert confidences == sorted(confidences, reverse=True)

