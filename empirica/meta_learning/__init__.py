"""
Meta-learning module for analyzing patterns across research projects.
"""

from .analyzer import PatternAnalyzer
from .models import MetaLearningModels
from .agent import MetaLearningAgent

__all__ = [
    'PatternAnalyzer',
    'MetaLearningModels',
    'MetaLearningAgent',
]

