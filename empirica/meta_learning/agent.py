"""
Meta-learning agent for continuous pattern analysis and insight generation.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime

from ..knowledge_graph.storage import GraphStorage
from .analyzer import PatternAnalyzer
from .models import MetaLearningModels
from ..logger import get_logger

logger = get_logger(__name__)


class MetaLearningAgent:
    """
    Meta-learning agent that continuously analyzes projects and generates insights.
    
    Identifies emerging best practices, detects anti-patterns, and provides
    recommendations based on accumulated knowledge.
    """
    
    def __init__(self, storage: GraphStorage):
        """
        Initialize the meta-learning agent.
        
        Args:
            storage: GraphStorage instance
        """
        self.storage = storage
        self.analyzer = PatternAnalyzer(storage)
        self.models = MetaLearningModels(storage, self.analyzer)
        self.last_analysis_time: Optional[datetime] = None
    
    def analyze_new_projects(self) -> Dict[str, Any]:
        """
        Analyze newly added projects and update pattern knowledge.
        
        Returns:
            Dictionary with analysis results and insights
        """
        logger.info("Analyzing new projects for meta-learning insights")
        
        # Refresh cached patterns
        self.models._cache_patterns()
        
        # Perform comprehensive analysis
        insights = {
            "timestamp": datetime.now().isoformat(),
            "success_rates": self.analyzer.analyze_success_rates_by_method_type(),
            "tool_effectiveness": self.analyzer.analyze_tool_effectiveness(),
            "failure_modes": self.analyzer.identify_common_failure_modes(),
            "domain_patterns": self.analyzer.analyze_domain_patterns(),
            "temporal_trends": self.analyzer.analyze_temporal_trends(),
            "statistical_relationships": self.analyzer.get_statistical_relationships(),
        }
        
        self.last_analysis_time = datetime.now()
        logger.info("Meta-learning analysis complete")
        
        return insights
    
    def identify_best_practices(self) -> List[Dict[str, Any]]:
        """
        Identify emerging best practices from successful projects.
        
        Returns:
            List of best practice recommendations
        """
        practices = []
        
        # Analyze successful projects
        tool_effectiveness = self.analyzer.analyze_tool_effectiveness()
        success_rates = self.analyzer.analyze_success_rates_by_method_type()
        domain_patterns = self.analyzer.analyze_domain_patterns()
        
        # Best practice: Use effective tools
        effective_tools = [
            (tool, metrics["success_rate"])
            for tool, metrics in tool_effectiveness.items()
            if metrics.get("success_rate", 0) > 0.7 and metrics.get("usage_count", 0) >= 3
        ]
        if effective_tools:
            practices.append({
                "type": "tool_selection",
                "description": "Use tools with high success rates",
                "recommendations": [tool for tool, _ in sorted(effective_tools, key=lambda x: x[1], reverse=True)[:5]],
                "confidence": "high",
            })
        
        # Best practice: Method characteristics
        if success_rates:
            best_method_type = max(success_rates.items(), key=lambda x: x[1])
            if best_method_type[1] > 0.6:
                practices.append({
                    "type": "method_structure",
                    "description": f"Methods with {best_method_type[0]} tend to succeed",
                    "recommendation": best_method_type[0],
                    "success_rate": best_method_type[1],
                    "confidence": "medium",
                })
        
        # Best practice: Domain-specific patterns
        for domain, pattern in domain_patterns.items():
            if pattern.get("success_rate", 0) > 0.7:
                practices.append({
                    "type": "domain_specific",
                    "description": f"Successful patterns in {domain} domain",
                    "domain": domain,
                    "common_tools": pattern.get("common_tools", {}),
                    "success_rate": pattern["success_rate"],
                    "confidence": "medium",
                })
        
        return practices
    
    def detect_anti_patterns(self) -> List[Dict[str, Any]]:
        """
        Detect anti-patterns that lead to failures.
        
        Returns:
            List of anti-pattern warnings
        """
        anti_patterns = []
        
        failure_modes = self.analyzer.identify_common_failure_modes()
        
        for failure_mode in failure_modes:
            if failure_mode["type"] == "tool_usage":
                common_tools = failure_mode.get("common_tools", {})
                if common_tools:
                    anti_patterns.append({
                        "type": "tool_anti_pattern",
                        "description": "Tools commonly associated with failures",
                        "tools": list(common_tools.keys())[:5],
                        "severity": "medium",
                    })
            
            elif failure_mode["type"] == "method_length":
                avg_length = failure_mode.get("avg_length", 0)
                anti_patterns.append({
                    "type": "method_length_anti_pattern",
                    "description": f"Methods with length around {avg_length} often fail",
                    "warning_length": avg_length,
                    "severity": "low",
                })
        
        return anti_patterns
    
    def generate_insights(self) -> Dict[str, Any]:
        """
        Generate comprehensive insights from the knowledge base.
        
        Returns:
            Dictionary with insights and recommendations
        """
        logger.info("Generating meta-learning insights")
        
        insights = {
            "timestamp": datetime.now().isoformat(),
            "best_practices": self.identify_best_practices(),
            "anti_patterns": self.detect_anti_patterns(),
            "summary": {
                "total_projects": len(self.storage.list_projects()),
                "successful_projects": sum(
                    1 for pid in self.storage.list_projects()
                    if (proj := self.storage.load_project(pid)) and proj.success is True
                ),
            },
        }
        
        # Add temporal trends
        trends = self.analyzer.analyze_temporal_trends()
        insights["trends"] = trends
        
        return insights
    
    def get_recommendations(
        self,
        idea_text: str,
        domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get personalized recommendations for a research idea.
        
        Args:
            idea_text: The research idea text
            domain: Optional research domain
            
        Returns:
            Dictionary with recommendations
        """
        recommendations = {
            "idea": idea_text[:100] + "..." if len(idea_text) > 100 else idea_text,
            "success_probability": self.models.predict_success_probability(idea_text, domain=domain),
            "recommended_methods": self.models.recommend_methods(idea_text, domain=domain),
            "estimated_time": self.models.estimate_execution_time(""),  # Will be updated when method is known
            "warnings": [],
        }
        
        # Check for failure prediction
        likely_to_fail, reasons = self.models.predict_failure(idea_text)
        if likely_to_fail:
            recommendations["warnings"].extend(reasons)
        
        # Add domain-specific recommendations
        if domain:
            domain_patterns = self.analyzer.analyze_domain_patterns()
            if domain in domain_patterns:
                pattern = domain_patterns[domain]
                recommendations["domain_specific"] = {
                    "common_tools": pattern.get("common_tools", {}),
                    "success_rate": pattern.get("success_rate", 0.5),
                }
        
        return recommendations

