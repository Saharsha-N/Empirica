"""
Pattern analysis engine for meta-learning.

    # improve error messages
Analyzes patterns across projects to identify what works, what doesn't,
    # optimize performance
and emerging best practices.
"""

from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, Counter
from datetime import datetime, timedelta

from ..knowledge_graph.storage import GraphStorage
from ..knowledge_graph.models import NodeType, EdgeType, ProjectGraph
from ..logger import get_logger

logger = get_logger(__name__)


class PatternAnalyzer:
    """
    Analyzes patterns across research projects to extract insights.
    
    Identifies success patterns, failure modes, tool effectiveness,
    and domain-specific trends.
    """
    
    def __init__(self, storage: GraphStorage):
        """
        Initialize the pattern analyzer.
        
        Args:
            storage: GraphStorage instance to analyze
        """
        self.storage = storage
    
    def analyze_success_rates_by_method_type(self) -> Dict[str, float]:
        """
        Analyze success rates by method characteristics.
        
        Returns:
            Dictionary mapping method characteristics to success rates
        """
        project_ids = self.storage.list_projects()
        
        method_stats = defaultdict(lambda: {"success": 0, "total": 0})
        
        for project_id in project_ids:
            project = self.storage.load_project(project_id)
            if not project:
                continue
            
            method_nodes = project.get_nodes_by_type(NodeType.METHOD)
            if not method_nodes:
                continue
            
            method_node = method_nodes[0]
            num_steps = method_node.metadata.get("num_steps", 0)
            method_length = len(method_node.content)
            
            # Categorize methods
            categories = []
            if num_steps > 0:
                categories.append(f"steps_{num_steps}")
            if method_length > 2000:
                categories.append("long_method")
            elif method_length < 500:
                categories.append("short_method")
            
            for category in categories:
                method_stats[category]["total"] += 1
                if project.success is True:
                    method_stats[category]["success"] += 1
        
        # Calculate success rates
        success_rates = {}
        for category, stats in method_stats.items():
            if stats["total"] > 0:
                success_rates[category] = stats["success"] / stats["total"]
        
        return success_rates
    
    def analyze_tool_effectiveness(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze effectiveness of different tools.
        
        Returns:
            Dictionary mapping tool names to effectiveness metrics
        """
        project_ids = self.storage.list_projects()
        
        tool_stats = defaultdict(lambda: {"success": 0, "total": 0, "quality_scores": []})
        
        for project_id in project_ids:
            project = self.storage.load_project(project_id)
            if not project:
                continue
            
            tool_nodes = project.get_nodes_by_type(NodeType.TOOL)
            for tool_node in tool_nodes:
                tool_name = tool_node.content
                tool_stats[tool_name]["total"] += 1
                
                if project.success is True:
                    tool_stats[tool_name]["success"] += 1
                
                if project.quality_score is not None:
                    tool_stats[tool_name]["quality_scores"].append(project.quality_score)
        
        # Calculate metrics
        tool_effectiveness = {}
        for tool_name, stats in tool_stats.items():
            if stats["total"] > 0:
                effectiveness = {
                    "success_rate": stats["success"] / stats["total"],
                    "usage_count": stats["total"],
                }
                
                if stats["quality_scores"]:
                    effectiveness["avg_quality"] = sum(stats["quality_scores"]) / len(stats["quality_scores"])
                
                tool_effectiveness[tool_name] = effectiveness
        
        return tool_effectiveness
    
    def identify_common_failure_modes(self) -> List[Dict[str, Any]]:
        """
        Identify common patterns in failed projects.
        
        Returns:
            List of failure mode descriptions with statistics
        """
        project_ids = self.storage.list_projects()
        failed_projects = []
        
        for project_id in project_ids:
            project = self.storage.load_project(project_id)
            if project and project.success is False:
                failed_projects.append(project)
        
        if not failed_projects:
            return []
        
        failure_modes = []
        
        # Analyze method characteristics of failures
        method_lengths = []
        method_steps = []
        tool_usage = Counter()
        
        for project in failed_projects:
            method_nodes = project.get_nodes_by_type(NodeType.METHOD)
            if method_nodes:
                method = method_nodes[0]
                method_lengths.append(len(method.content))
                method_steps.append(method.metadata.get("num_steps", 0))
            
            tool_nodes = project.get_nodes_by_type(NodeType.TOOL)
            for tool_node in tool_nodes:
                tool_usage[tool_node.content] += 1
        
        if method_lengths:
            failure_modes.append({
                "type": "method_length",
                "description": "Failed projects often have methods of unusual length",
                "avg_length": sum(method_lengths) / len(method_lengths),
                "count": len(failed_projects),
            })
        
        if method_steps:
            failure_modes.append({
                "type": "method_complexity",
                "description": "Failed projects may have too many or too few steps",
                "avg_steps": sum(method_steps) / len(method_steps),
                "count": len(failed_projects),
            })
        
        if tool_usage:
            failure_modes.append({
                "type": "tool_usage",
                "description": "Common tools in failed projects",
                "common_tools": dict(tool_usage.most_common(5)),
                "count": len(failed_projects),
            })
        
        return failure_modes
    
    def analyze_domain_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze patterns specific to different research domains.
        
        Returns:
            Dictionary mapping domains to their patterns
        """
        project_ids = self.storage.list_projects()
        
        domain_stats = defaultdict(lambda: {
            "projects": [],
            "success_count": 0,
            "total_count": 0,
            "tools": Counter(),
            "avg_quality": [],
        })
        
        for project_id in project_ids:
            project = self.storage.load_project(project_id)
            if not project:
                continue
            
            idea_nodes = project.get_nodes_by_type(NodeType.IDEA)
            if idea_nodes:
                domain = idea_nodes[0].metadata.get("domain")
                if domain:
                    domain_stats[domain]["projects"].append(project)
                    domain_stats[domain]["total_count"] += 1
                    
                    if project.success is True:
                        domain_stats[domain]["success_count"] += 1
                    
                    if project.quality_score is not None:
                        domain_stats[domain]["avg_quality"].append(project.quality_score)
                    
                    tool_nodes = project.get_nodes_by_type(NodeType.TOOL)
                    for tool_node in tool_nodes:
                        domain_stats[domain]["tools"][tool_node.content] += 1
        
        # Process domain statistics
        domain_patterns = {}
        for domain, stats in domain_stats.items():
            if stats["total_count"] > 0:
                pattern = {
                    "total_projects": stats["total_count"],
                    "success_rate": stats["success_count"] / stats["total_count"],
                    "common_tools": dict(stats["tools"].most_common(5)),
                }
                
                if stats["avg_quality"]:
                    pattern["avg_quality_score"] = sum(stats["avg_quality"]) / len(stats["avg_quality"])
                
                domain_patterns[domain] = pattern
        
        return domain_patterns
    
    def analyze_temporal_trends(self, days: int = 30) -> Dict[str, Any]:
        """
        Analyze trends over time.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Dictionary with temporal trend statistics
        """
        project_ids = self.storage.list_projects()
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_projects = []
        older_projects = []
        
        for project_id in project_ids:
            project = self.storage.load_project(project_id)
            if not project:
                continue
            
            if project.created_at >= cutoff_date:
                recent_projects.append(project)
            else:
                older_projects.append(project)
        
        trends = {
            "recent_period_days": days,
            "recent_projects": len(recent_projects),
            "older_projects": len(older_projects),
        }
        
        if recent_projects:
            recent_success = sum(1 for p in recent_projects if p.success is True)
            trends["recent_success_rate"] = recent_success / len(recent_projects)
        
        if older_projects:
            older_success = sum(1 for p in older_projects if p.success is True)
            trends["older_success_rate"] = older_success / len(older_projects)
        
        # Compare tool usage
        recent_tools = Counter()
        older_tools = Counter()
        
        for project in recent_projects:
            tool_nodes = project.get_nodes_by_type(NodeType.TOOL)
            for tool_node in tool_nodes:
                recent_tools[tool_node.content] += 1
        
        for project in older_projects:
            tool_nodes = project.get_nodes_by_type(NodeType.TOOL)
            for tool_node in tool_nodes:
                older_tools[tool_node.content] += 1
        
        # Find emerging tools (more common in recent projects)
        emerging_tools = {}
        for tool, recent_count in recent_tools.items():
            older_count = older_tools.get(tool, 0)
            if recent_count > 0 and (older_count == 0 or recent_count / max(older_count, 1) > 1.5):
                emerging_tools[tool] = {
                    "recent_usage": recent_count,
                    "older_usage": older_count,
                }
        
        trends["emerging_tools"] = emerging_tools
        
        return trends
    
    def get_statistical_relationships(self) -> Dict[str, Any]:
        """
        Perform statistical analysis of relationships in the graph.
        
        Returns:
            Dictionary with statistical insights
        """
        project_ids = self.storage.list_projects()
        
        # Analyze idea-method-result relationships
        idea_method_pairs = []
        method_result_pairs = []
        
        for project_id in project_ids:
            project = self.storage.load_project(project_id)
            if not project:
                continue
            
            idea_nodes = project.get_nodes_by_type(NodeType.IDEA)
            method_nodes = project.get_nodes_by_type(NodeType.METHOD)
            result_nodes = project.get_nodes_by_type(NodeType.RESULT)
            
            if idea_nodes and method_nodes:
                idea_method_pairs.append({
                    "idea_length": len(idea_nodes[0].content),
                    "method_length": len(method_nodes[0].content),
                    "success": project.success,
                })
            
            if method_nodes and result_nodes:
                method_result_pairs.append({
                    "method_length": len(method_nodes[0].content),
                    "result_length": len(result_nodes[0].content),
                    "num_plots": result_nodes[0].metadata.get("num_plots", 0),
                    "success": project.success,
                })
        
        relationships = {
            "idea_method_pairs": len(idea_method_pairs),
            "method_result_pairs": len(method_result_pairs),
        }
        
        # Calculate correlations if we have enough data
        if idea_method_pairs:
            successful_pairs = [p for p in idea_method_pairs if p["success"] is True]
            if successful_pairs:
                relationships["successful_idea_method_avg_lengths"] = {
                    "idea": sum(p["idea_length"] for p in successful_pairs) / len(successful_pairs),
                    "method": sum(p["method_length"] for p in successful_pairs) / len(successful_pairs),
                }
        
        return relationships

