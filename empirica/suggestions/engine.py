"""
Suggestion engine for intelligent research recommendations.
"""

from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from ..knowledge_graph.storage import GraphStorage
from ..knowledge_graph.query import GraphQuery
from ..knowledge_graph.models import NodeType
from ..meta_learning.agent import MetaLearningAgent
from ..meta_learning.models import MetaLearningModels
from ..logger import get_logger

logger = get_logger(__name__)


class SuggestionType(str, Enum):
    """Types of suggestions."""
    METHOD = "method"
    TOOL = "tool"
    DATASET = "dataset"
    WARNING = "warning"
    OPTIMIZATION = "optimization"


class Suggestion:
    """A single suggestion with metadata."""
    
    def __init__(
        self,
        suggestion_type: SuggestionType,
        content: str,
        confidence: float,
        reason: str,
        source: Optional[str] = None,
    ):
        """
        Initialize a suggestion.
        
        Args:
            suggestion_type: Type of suggestion
            content: The suggestion content/text
            confidence: Confidence score (0.0 to 1.0)
            reason: Explanation for the suggestion
            source: Optional source of the suggestion (e.g., "similar_project_42")
        """
        self.suggestion_type = suggestion_type
        self.content = content
        self.confidence = confidence
        self.reason = reason
        self.source = source


class SuggestionEngine:
    """
    Engine for generating intelligent research suggestions.
    
    Uses the knowledge graph and meta-learning to provide recommendations
    based on similar past projects and learned patterns.
    """
    
    def __init__(self, storage: GraphStorage, meta_agent: MetaLearningAgent):
        """
        Initialize the suggestion engine.
        
        Args:
            storage: GraphStorage instance
            meta_agent: MetaLearningAgent instance
        """
        self.storage = storage
        self.meta_agent = meta_agent
        self.query = GraphQuery(storage)
        self.models = meta_agent.models
    
    def suggest_methods(
        self,
        idea_text: str,
        current_method: Optional[str] = None,
        top_k: int = 3,
    ) -> List[Suggestion]:
        """
        Suggest methods based on similar successful projects.
        
        Args:
            idea_text: The research idea
            current_method: Optional current method to compare against
            top_k: Number of suggestions to return
            
        Returns:
            List of method suggestions
        """
        suggestions = []
        
        # Get method recommendations from meta-learning
        recommended_methods = self.models.recommend_methods(idea_text, top_k=top_k)
        
        for method_text, confidence in recommended_methods:
            suggestion = Suggestion(
                suggestion_type=SuggestionType.METHOD,
                content=method_text[:500] + "..." if len(method_text) > 500 else method_text,
                confidence=confidence,
                reason=f"Similar successful projects used this method (confidence: {confidence:.1%})",
            )
            suggestions.append(suggestion)
        
        return suggestions
    
    def suggest_tools(
        self,
        idea_text: str,
        domain: Optional[str] = None,
        current_tools: Optional[List[str]] = None,
    ) -> List[Suggestion]:
        """
        Suggest tools based on effectiveness and domain patterns.
        
        Args:
            idea_text: The research idea
            domain: Optional research domain
            current_tools: Optional list of currently planned tools
            
        Returns:
            List of tool suggestions
        """
        suggestions = []
        
        # Get tool effectiveness data
        tool_effectiveness = self.meta_agent.analyzer.analyze_tool_effectiveness()
        
        # Get domain-specific tools if domain provided
        domain_tools = []
        if domain:
            domain_patterns = self.meta_agent.analyzer.analyze_domain_patterns()
            if domain in domain_patterns:
                domain_tools = list(domain_patterns[domain].get("common_tools", {}).keys())
        
        # Rank tools by effectiveness
        tool_scores = []
        for tool, metrics in tool_effectiveness.items():
            if current_tools and tool in current_tools:
                continue  # Skip already planned tools
            
            success_rate = metrics.get("success_rate", 0.5)
            usage_count = metrics.get("usage_count", 0)
            
            # Boost score if tool is common in domain
            domain_boost = 0.1 if tool in domain_tools else 0.0
            
            # Score based on success rate and usage (more usage = more reliable)
            score = success_rate + (min(usage_count / 10, 0.2)) + domain_boost
            tool_scores.append((tool, score, success_rate, usage_count))
        
        # Sort by score and take top tools
        tool_scores.sort(key=lambda x: x[1], reverse=True)
        
        for tool, score, success_rate, usage_count in tool_scores[:5]:
            suggestion = Suggestion(
                suggestion_type=SuggestionType.TOOL,
                content=tool,
                confidence=min(score, 1.0),
                reason=f"High success rate ({success_rate:.1%}) with {usage_count} uses",
            )
            suggestions.append(suggestion)
        
        return suggestions
    
    def check_dataset_compatibility(
        self,
        idea_text: str,
        dataset_paths: List[str],
    ) -> List[Suggestion]:
        """
        Check dataset compatibility and suggest alternatives.
        
        Args:
            idea_text: The research idea
            dataset_paths: List of dataset file paths
            
        Returns:
            List of compatibility suggestions/warnings
        """
        suggestions = []
        
        # Find similar projects
        similar_projects = self.query.find_similar_ideas(idea_text, top_k=5)
        
        if not similar_projects:
            return suggestions
        
        # Check what datasets similar projects used
        used_datasets = set()
        for project, _, _ in similar_projects:
            dataset_nodes = project.get_nodes_by_type(NodeType.DATASET)
            for dataset_node in dataset_nodes:
                used_datasets.add(dataset_node.content)
        
        # Check if current datasets match
        current_datasets = set(dataset_paths)
        if not current_datasets.intersection(used_datasets):
            suggestions.append(Suggestion(
                suggestion_type=SuggestionType.DATASET,
                content="Consider datasets used in similar projects",
                confidence=0.6,
                reason="Your datasets differ from those used in similar successful projects",
            ))
        
        return suggestions
    
    def generate_warnings(
        self,
        idea_text: str,
        method_text: Optional[str] = None,
        tools: Optional[List[str]] = None,
    ) -> List[Suggestion]:
        """
        Generate warnings for known failure patterns.
        
        Args:
            idea_text: The research idea
            method_text: Optional methodology
            tools: Optional list of tools
            
        Returns:
            List of warning suggestions
        """
        warnings = []
        
        # Check for failure prediction
        likely_to_fail, reasons = self.models.predict_failure(idea_text, method_text, tools)
        
        if likely_to_fail:
            for reason in reasons:
                warnings.append(Suggestion(
                    suggestion_type=SuggestionType.WARNING,
                    content=reason,
                    confidence=0.7,
                    reason="Based on analysis of failed projects",
                ))
        
        # Check success probability
        success_prob = self.models.predict_success_probability(idea_text, method_text, tools)
        if success_prob < 0.4:
            warnings.append(Suggestion(
                suggestion_type=SuggestionType.WARNING,
                content=f"Low predicted success probability ({success_prob:.1%})",
                confidence=0.8,
                reason="Historical data suggests low success rate for similar projects",
            ))
        
        return warnings
    
    def suggest_optimizations(
        self,
        method_text: str,
        execution_time: Optional[float] = None,
    ) -> List[Suggestion]:
        """
        Suggest optimizations based on past experiments.
        
        Args:
            method_text: The methodology
            execution_time: Optional current execution time
            
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        
        # Estimate execution time
        estimated_time, confidence_range = self.models.estimate_execution_time(method_text)
        
        if execution_time and execution_time > estimated_time * 1.5:
            suggestions.append(Suggestion(
                suggestion_type=SuggestionType.OPTIMIZATION,
                content=f"Execution time ({execution_time:.1f} min) is higher than estimated ({estimated_time:.1f} min)",
                confidence=0.6,
                reason="Consider reviewing method steps for efficiency",
            ))
        
        # Check method length
        method_length = len(method_text)
        if method_length > 5000:
            suggestions.append(Suggestion(
                suggestion_type=SuggestionType.OPTIMIZATION,
                content="Method is very long - consider breaking into smaller steps",
                confidence=0.5,
                reason="Shorter, focused methods often perform better",
            ))
        
        return suggestions
    
    def get_all_suggestions(
        self,
        idea_text: str,
        method_text: Optional[str] = None,
        dataset_paths: Optional[List[str]] = None,
        tools: Optional[List[str]] = None,
        domain: Optional[str] = None,
    ) -> Dict[str, List[Suggestion]]:
        """
        Get all suggestions for a research project.
        
        Args:
            idea_text: The research idea
            method_text: Optional methodology
            dataset_paths: Optional list of dataset paths
            tools: Optional list of tools
            domain: Optional research domain
            
        Returns:
            Dictionary mapping suggestion types to lists of suggestions
        """
        all_suggestions = {
            "methods": self.suggest_methods(idea_text, method_text),
            "tools": self.suggest_tools(idea_text, domain, tools),
            "warnings": self.generate_warnings(idea_text, method_text, tools),
            "optimizations": [],
        }
        
        if dataset_paths:
            all_suggestions["datasets"] = self.check_dataset_compatibility(idea_text, dataset_paths)
        else:
            all_suggestions["datasets"] = []
        
        if method_text:
            all_suggestions["optimizations"] = self.suggest_optimizations(method_text)
        
        return all_suggestions

