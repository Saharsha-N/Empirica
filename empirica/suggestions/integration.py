"""
Integration of suggestions into the existing Empirica workflow.
"""

from typing import Optional, Dict, Any
import os

from .engine import SuggestionEngine, Suggestion
from ..knowledge_graph.storage import GraphStorage
from ..meta_learning.agent import MetaLearningAgent
from ..logger import get_logger

logger = get_logger(__name__)


class SuggestionIntegration:
    """
    Integrates suggestions into the Empirica workflow.
    
    Provides hooks and callbacks for displaying suggestions at appropriate
    points in the research process.
    """
    
    def __init__(
        self,
        storage: GraphStorage,
        meta_agent: MetaLearningAgent,
        enabled: bool = True,
    ):
        """
        Initialize suggestion integration.
        
        Args:
            storage: GraphStorage instance
            meta_agent: MetaLearningAgent instance
            enabled: Whether suggestions are enabled
        """
        self.storage = storage
        self.meta_agent = meta_agent
        self.enabled = enabled
        self.engine = SuggestionEngine(storage, meta_agent) if enabled else None
        self._suggestion_history: list[Dict[str, Any]] = []
    
    def get_idea_suggestions(
        self,
        idea_text: str,
        data_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get suggestions when generating an idea.
        
        Args:
            idea_text: The generated or proposed idea
            data_description: Optional data description for context
            
        Returns:
            Dictionary with suggestions and warnings
        """
        if not self.enabled or not self.engine:
            return {}
        
        try:
            # Extract domain from idea if possible
            domain = None
            idea_lower = idea_text.lower()
            domain_keywords = {
                "cosmology": ["cosmology", "cosmic", "universe", "galaxy"],
                "biology": ["biology", "biological", "organism", "cell"],
                "chemistry": ["chemistry", "chemical", "molecule"],
                "material science": ["material", "crystal", "lattice"],
                "machine learning": ["machine learning", "neural network", "model"],
            }
            
            for dom, keywords in domain_keywords.items():
                if any(kw in idea_lower for kw in keywords):
                    domain = dom
                    break
            
            # Find similar ideas
            similar_projects = self.engine.query.find_similar_ideas(idea_text, top_k=3)
            
            suggestions = {
                "similar_projects": [
                    {
                        "project_id": str(proj.project_id),
                        "similarity": float(sim),
                        "success": proj.success,
                    }
                    for proj, node, sim in similar_projects
                ],
                "warnings": [s.content for s in self.engine.generate_warnings(idea_text)],
                "success_probability": self.engine.models.predict_success_probability(idea_text, domain=domain),
            }
            
            # Log suggestion usage
            self._log_suggestion("idea", suggestions)
            
            return suggestions
        
        except Exception as e:
            logger.warning(f"Failed to generate idea suggestions: {e}")
            return {}
    
    def get_method_suggestions(
        self,
        idea_text: str,
        method_text: Optional[str] = None,
        data_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get suggestions when generating or setting a method.
        
        Args:
            idea_text: The research idea
            method_text: Optional current method
            data_description: Optional data description
            
        Returns:
            Dictionary with method suggestions
        """
        if not self.enabled or not self.engine:
            return {}
        
        try:
            # Extract tools from data description
            tools = []
            if data_description:
                # Simple tool extraction
                import re
                tool_patterns = [
                    r'\b(pandas|numpy|scipy|sklearn|matplotlib|seaborn)\b',
                    r'\b(tensorflow|pytorch|keras)\b',
                ]
                for pattern in tool_patterns:
                    matches = re.findall(pattern, data_description, re.IGNORECASE)
                    tools.extend([m.lower() for m in matches])
            
            all_suggestions = self.engine.get_all_suggestions(
                idea_text=idea_text,
                method_text=method_text,
                tools=tools if tools else None,
            )
            
            suggestions = {
                "method_recommendations": [
                    {
                        "content": s.content,
                        "confidence": s.confidence,
                        "reason": s.reason,
                    }
                    for s in all_suggestions.get("methods", [])
                ],
                "tool_recommendations": [
                    {
                        "tool": s.content,
                        "confidence": s.confidence,
                        "reason": s.reason,
                    }
                    for s in all_suggestions.get("tools", [])
                ],
                "warnings": [s.content for s in all_suggestions.get("warnings", [])],
            }
            
            self._log_suggestion("method", suggestions)
            return suggestions
        
        except Exception as e:
            logger.warning(f"Failed to generate method suggestions: {e}")
            return {}
    
    def get_results_suggestions(
        self,
        idea_text: str,
        method_text: str,
        results_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get suggestions when generating results.
        
        Args:
            idea_text: The research idea
            method_text: The methodology used
            results_text: Optional current results
            
        Returns:
            Dictionary with optimization suggestions
        """
        if not self.enabled or not self.engine:
            return {}
        
        try:
            optimizations = self.engine.suggest_optimizations(method_text)
            
            suggestions = {
                "optimizations": [
                    {
                        "content": s.content,
                        "confidence": s.confidence,
                        "reason": s.reason,
                    }
                    for s in optimizations
                ],
            }
            
            self._log_suggestion("results", suggestions)
            return suggestions
        
        except Exception as e:
            logger.warning(f"Failed to generate results suggestions: {e}")
            return {}
    
    def _log_suggestion(self, stage: str, suggestions: Dict[str, Any]) -> None:
        """
        Log suggestion usage for feedback loop.
        
        Args:
            stage: Stage where suggestion was shown (idea, method, results)
            suggestions: The suggestions provided
        """
        self._suggestion_history.append({
            "stage": stage,
            "suggestions": suggestions,
            "timestamp": os.environ.get("EMPTIMESTAMP", ""),  # Could use actual timestamp
        })
        
        # Keep only last 100 entries
        if len(self._suggestion_history) > 100:
            self._suggestion_history = self._suggestion_history[-100:]

