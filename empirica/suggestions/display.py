"""
Display system for research suggestions.
"""

from typing import List, Dict, Any, Optional
from enum import Enum

from .engine import Suggestion, SuggestionType
from ..logger import get_logger

logger = get_logger(__name__)


class DisplayFormat(str, Enum):
    """Display formats for suggestions."""
    TEXT = "text"
    RICH = "rich"
    JSON = "json"


class SuggestionDisplay:
    """
    Displays suggestions in various formats with confidence scores.
    """
    
    @staticmethod
    def format_suggestion(suggestion: Suggestion, format_type: DisplayFormat = DisplayFormat.TEXT) -> str:
        """
        Format a single suggestion.
        
        Args:
            suggestion: The suggestion to format
            format_type: Display format
            
        Returns:
            Formatted suggestion string
        """
        if format_type == DisplayFormat.JSON:
            import json
            return json.dumps({
                "type": suggestion.suggestion_type.value,
                "content": suggestion.content,
                "confidence": suggestion.confidence,
                "reason": suggestion.reason,
                "source": suggestion.source,
            }, indent=2)
        
        # Text format
        confidence_bar = "█" * int(suggestion.confidence * 10) + "░" * (10 - int(suggestion.confidence * 10))
        
        if format_type == DisplayFormat.RICH:
            # Rich formatting (could use rich library if available)
            return (
                f"[{suggestion.suggestion_type.value.upper()}] {suggestion.content}\n"
                f"  Confidence: {confidence_bar} {suggestion.confidence:.1%}\n"
                f"  Reason: {suggestion.reason}"
            )
        else:
            # Plain text
            return (
                f"[{suggestion.suggestion_type.value.upper()}] {suggestion.content}\n"
                f"  Confidence: {suggestion.confidence:.1%} | {suggestion.reason}"
            )
    
    @staticmethod
    def display_suggestions(
        suggestions: List[Suggestion],
        title: str = "Suggestions",
        format_type: DisplayFormat = DisplayFormat.TEXT,
        min_confidence: float = 0.3,
    ) -> None:
        """
        Display a list of suggestions.
        
        Args:
            suggestions: List of suggestions to display
            title: Title for the suggestion list
            format_type: Display format
            min_confidence: Minimum confidence to display
        """
        filtered = [s for s in suggestions if s.confidence >= min_confidence]
        
        if not filtered:
            logger.info(f"No suggestions above confidence threshold {min_confidence:.1%}")
            return
        
        logger.info(f"\n{'='*60}")
        logger.info(f"{title}")
        logger.info(f"{'='*60}")
        
        # Group by type
        by_type = {}
        for suggestion in filtered:
            stype = suggestion.suggestion_type.value
            if stype not in by_type:
                by_type[stype] = []
            by_type[stype].append(suggestion)
        
        # Display by type
        for stype, type_suggestions in by_type.items():
            logger.info(f"\n{stype.upper()} Suggestions:")
            logger.info("-" * 60)
            for suggestion in sorted(type_suggestions, key=lambda s: s.confidence, reverse=True):
                formatted = SuggestionDisplay.format_suggestion(suggestion, format_type)
                logger.info(formatted)
                logger.info("")
        
        logger.info(f"{'='*60}\n")
    
    @staticmethod
    def display_suggestion_dict(
        suggestions_dict: Dict[str, Any],
        format_type: DisplayFormat = DisplayFormat.TEXT,
    ) -> None:
        """
        Display suggestions from a dictionary (from integration).
        
        Args:
            suggestions_dict: Dictionary with suggestion data
            format_type: Display format
        """
        if not suggestions_dict:
            return
        
        # Display warnings first
        if "warnings" in suggestions_dict and suggestions_dict["warnings"]:
            logger.warning("\n[WARNINGS]")
            for warning in suggestions_dict["warnings"]:
                logger.warning(f"  - {warning}")
            logger.warning("")
        
        # Display method recommendations
        if "method_recommendations" in suggestions_dict:
            method_suggestions = []
            for rec in suggestions_dict["method_recommendations"]:
                suggestion = Suggestion(
                    suggestion_type=SuggestionType.METHOD,
                    content=rec["content"],
                    confidence=rec["confidence"],
                    reason=rec["reason"],
                )
                method_suggestions.append(suggestion)
            
            if method_suggestions:
                SuggestionDisplay.display_suggestions(
                    method_suggestions,
                    title="Method Recommendations",
                    format_type=format_type,
                )
        
        # Display tool recommendations
        if "tool_recommendations" in suggestions_dict:
            tool_suggestions = []
            for rec in suggestions_dict["tool_recommendations"]:
                suggestion = Suggestion(
                    suggestion_type=SuggestionType.TOOL,
                    content=rec["tool"],
                    confidence=rec["confidence"],
                    reason=rec["reason"],
                )
                tool_suggestions.append(suggestion)
            
            if tool_suggestions:
                SuggestionDisplay.display_suggestions(
                    tool_suggestions,
                    title="Tool Recommendations",
                    format_type=format_type,
                )
        
        # Display optimizations
        if "optimizations" in suggestions_dict:
            opt_suggestions = []
            for opt in suggestions_dict["optimizations"]:
                suggestion = Suggestion(
                    suggestion_type=SuggestionType.OPTIMIZATION,
                    content=opt["content"],
                    confidence=opt["confidence"],
                    reason=opt["reason"],
                )
                opt_suggestions.append(suggestion)
            
            if opt_suggestions:
                SuggestionDisplay.display_suggestions(
                    opt_suggestions,
                    title="Optimization Suggestions",
                    format_type=format_type,
                )
        
        # Display success probability if available
        if "success_probability" in suggestions_dict:
            prob = suggestions_dict["success_probability"]
            logger.info(f"\nPredicted Success Probability: {prob:.1%}")
            if prob < 0.4:
                logger.warning("  Low success probability - consider reviewing approach")
            elif prob > 0.7:
                logger.info("  High success probability - approach looks promising")

