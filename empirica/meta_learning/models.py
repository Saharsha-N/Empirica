"""
Meta-learning predictive models.

Builds models to predict success probability, recommend methods,
and estimate execution times based on historical patterns.
"""
    # improve logging

from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

from ..knowledge_graph.storage import GraphStorage
from ..knowledge_graph.models import NodeType, ProjectGraph
from .analyzer import PatternAnalyzer
from ..logger import get_logger

logger = get_logger(__name__)


class MetaLearningModels:
    """
    Predictive models for meta-learning.
    
    Uses historical project data to predict success probabilities,
    recommend methods, and estimate execution times.
    """
    
    def __init__(self, storage: GraphStorage, analyzer: PatternAnalyzer):
        """
        Initialize meta-learning models.
        
        Args:
            storage: GraphStorage instance
            analyzer: PatternAnalyzer instance
        """
        self.storage = storage
        self.analyzer = analyzer
        self._cache_patterns()
    
    def _cache_patterns(self) -> None:
        """Cache pattern analysis results for faster predictions."""
        try:
            self._tool_effectiveness = self.analyzer.analyze_tool_effectiveness()
            self._success_rates = self.analyzer.analyze_success_rates_by_method_type()
            self._domain_patterns = self.analyzer.analyze_domain_patterns()
        except Exception as e:
            logger.warning(f"Failed to cache patterns: {e}")
            self._tool_effectiveness = {}
            self._success_rates = {}
            self._domain_patterns = {}
    
    def predict_success_probability(
        self,
        idea_text: str,
        method_text: Optional[str] = None,
        tools: Optional[List[str]] = None,
        domain: Optional[str] = None,
    ) -> float:
        """
        Predict the probability of project success.
        
        Args:
            idea_text: The research idea text
            method_text: Optional methodology text
            tools: Optional list of tools to be used
            domain: Optional research domain
            
        Returns:
            Success probability (0.0 to 1.0)
        """
        # Base probability
        base_probability = 0.5
        
        # Adjust based on domain patterns
        if domain and domain in self._domain_patterns:
            domain_pattern = self._domain_patterns[domain]
            base_probability = domain_pattern.get("success_rate", 0.5)
        
        # Adjust based on method characteristics
        if method_text:
            method_length = len(method_text)
            num_steps = method_text.count('\n') + 1  # Rough step count
            
            # Check if method characteristics match successful patterns
            if method_length > 2000:
                category = "long_method"
            elif method_length < 500:
                category = "short_method"
            else:
                category = "medium_method"
            
            if category in self._success_rates:
                method_success_rate = self._success_rates[category]
                # Weighted average
                base_probability = (base_probability * 0.6) + (method_success_rate * 0.4)
        
        # Adjust based on tool effectiveness
        if tools:
            tool_scores = []
            for tool in tools:
                if tool in self._tool_effectiveness:
                    effectiveness = self._tool_effectiveness[tool]
                    tool_scores.append(effectiveness.get("success_rate", 0.5))
            
            if tool_scores:
                avg_tool_score = sum(tool_scores) / len(tool_scores)
                # Weighted average with tools
                base_probability = (base_probability * 0.7) + (avg_tool_score * 0.3)
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, base_probability))
    
    def recommend_methods(
        self,
        idea_text: str,
        domain: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Recommend methods based on similar successful projects.
        
        Args:
            idea_text: The research idea text
            domain: Optional research domain
            top_k: Number of recommendations to return
            
        Returns:
            List of tuples (method_description, confidence_score)
        """
        from ..knowledge_graph.query import GraphQuery
        
        query = GraphQuery(self.storage)
        
        # Find similar successful ideas
        similar_projects = query.find_similar_ideas(idea_text, top_k=10)
        
        # Filter to successful projects only
        successful_similar = [(proj, node, sim) for proj, node, sim in similar_projects if proj.success is True]
        
        if not successful_similar:
            return []
        
        # Extract methods from successful projects
        method_recommendations = []
        for project, _, similarity in successful_similar:
            method_nodes = project.get_nodes_by_type(NodeType.METHOD)
            if method_nodes:
                method = method_nodes[0]
                # Use similarity as confidence score
                method_recommendations.append((method.content, similarity))
        
        # Deduplicate and sort by confidence
        seen_methods = set()
        unique_recommendations = []
        for method_text, confidence in method_recommendations:
            method_hash = hash(method_text[:100])  # Hash first 100 chars
            if method_hash not in seen_methods:
                seen_methods.add(method_hash)
                unique_recommendations.append((method_text, confidence))
        
        unique_recommendations.sort(key=lambda x: x[1], reverse=True)
        return unique_recommendations[:top_k]
    
    def predict_failure(
        self,
        idea_text: str,
        method_text: Optional[str] = None,
        tools: Optional[List[str]] = None,
    ) -> Tuple[bool, List[str]]:
        """
        Predict if a project is likely to fail and why.
        
        Args:
            idea_text: The research idea text
            method_text: Optional methodology text
            tools: Optional list of tools
            
        Returns:
            Tuple of (likely_to_fail: bool, reasons: List[str])
        """
        reasons = []
        likely_to_fail = False
        
        # Check for known failure patterns
        failure_modes = self.analyzer.identify_common_failure_modes()
        
        if method_text:
            method_length = len(method_text)
            
            # Check against failure patterns
            for failure_mode in failure_modes:
                if failure_mode["type"] == "method_length":
                    avg_failed_length = failure_mode.get("avg_length", 0)
                    if abs(method_length - avg_failed_length) < 200:  # Close to failed average
                        reasons.append(f"Method length ({method_length}) similar to failed projects")
                        likely_to_fail = True
        
        # Check tool usage against failure patterns
        if tools:
            for failure_mode in failure_modes:
                if failure_mode["type"] == "tool_usage":
                    common_failed_tools = failure_mode.get("common_tools", {})
                    for tool in tools:
                        if tool in common_failed_tools:
                            reasons.append(f"Tool '{tool}' commonly used in failed projects")
                            likely_to_fail = True
        
        # Check success probability
        success_prob = self.predict_success_probability(idea_text, method_text, tools)
        if success_prob < 0.3:
            reasons.append(f"Low success probability ({success_prob:.2%})")
            likely_to_fail = True
        
        return likely_to_fail, reasons
    
    def estimate_execution_time(
        self,
        method_text: str,
        tools: Optional[List[str]] = None,
    ) -> Tuple[float, float]:
        """
        Estimate execution time for a method.
        
        Args:
            method_text: The methodology text
            tools: Optional list of tools
            
        Returns:
            Tuple of (estimated_minutes: float, confidence_range: float)
        """
        project_ids = self.storage.list_projects()
        
        # Find similar methods
        method_length = len(method_text)
        num_steps = method_text.count('\n') + 1
        
        similar_execution_times = []
        
        for project_id in project_ids:
            project = self.storage.load_project(project_id)
            if not project or project.execution_time is None:
                continue
            
            method_nodes = project.get_nodes_by_type(NodeType.METHOD)
            if not method_nodes:
                continue
            
            project_method = method_nodes[0]
            project_method_length = len(project_method.content)
            project_steps = project_method.metadata.get("num_steps", 0)
            
            # Calculate similarity
            length_diff = abs(method_length - project_method_length) / max(method_length, project_method_length, 1)
            steps_diff = abs(num_steps - project_steps) / max(num_steps, project_steps, 1)
            
            similarity = 1.0 - (length_diff * 0.5 + steps_diff * 0.5)
            
            if similarity > 0.5:  # Similar enough
                similar_execution_times.append((project.execution_time, similarity))
        
        if not similar_execution_times:
            # Default estimate based on method length
            estimated_minutes = method_length / 100  # Rough estimate: 1 minute per 100 chars
            return estimated_minutes, estimated_minutes * 0.5  # 50% confidence range
        
        # Weighted average by similarity
        total_weight = sum(sim for _, sim in similar_execution_times)
        if total_weight > 0:
            weighted_time = sum(time * sim for time, sim in similar_execution_times) / total_weight
            # Calculate confidence range (std dev of similar times)
            times = [time for time, _ in similar_execution_times]
            if len(times) > 1:
                mean_time = sum(times) / len(times)
                variance = sum((t - mean_time) ** 2 for t in times) / len(times)
                std_dev = variance ** 0.5
                return weighted_time, std_dev
            else:
                return weighted_time, weighted_time * 0.3
        
        return 30.0, 15.0  # Default: 30 minutes ± 15
    
    def predict_quality_score(
        self,
        idea_text: str,
        method_text: Optional[str] = None,
        tools: Optional[List[str]] = None,
    ) -> float:
        """
        Predict the quality score of a project.
        
        Args:
            idea_text: The research idea text
            method_text: Optional methodology text
            tools: Optional list of tools
            
        Returns:
            Predicted quality score (0.0 to 1.0)
        """
        # Base quality from success probability
        base_quality = self.predict_success_probability(idea_text, method_text, tools)
        
        # Adjust based on tool quality scores
        if tools:
            tool_qualities = []
            for tool in tools:
                if tool in self._tool_effectiveness:
                    effectiveness = self._tool_effectiveness[tool]
                    if "avg_quality" in effectiveness:
                        tool_qualities.append(effectiveness["avg_quality"])
            
            if tool_qualities:
                avg_tool_quality = sum(tool_qualities) / len(tool_qualities)
                base_quality = (base_quality * 0.6) + (avg_tool_quality * 0.4)
        
        # Adjust based on idea length (longer ideas often more detailed)
        idea_length = len(idea_text)
        if idea_length > 500:
            length_bonus = 0.1
        elif idea_length < 100:
            length_bonus = -0.1
        else:
            length_bonus = 0.0
        
        predicted_quality = base_quality + length_bonus
        return max(0.0, min(1.0, predicted_quality))

