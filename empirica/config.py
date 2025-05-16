from pathlib import Path
import os

# Using pathlib (modern approach) to define the base directory as the directory that contains this file.
BASE_DIR = Path(__file__).resolve().parent

# REPO_DIR is defined as one directory above the package
REPO_DIR = BASE_DIR.parent
## in colab we need REPO_DIR = "/content/Empirica/"

LaTeX_DIR = BASE_DIR / "paper_agents" / "LaTeX"

DEFAUL_PROJECT_NAME = "project"
"""Default name of the project"""

# Constants for defining .md files and folder names
INPUT_FILES = "input_files"
PLOTS_FOLDER = "plots"
PAPER_FOLDER = "paper"

DESCRIPTION_FILE = "data_description.md"
IDEA_FILE = "idea.md"
METHOD_FILE = "methods.md"
RESULTS_FILE = "results.md"
LITERATURE_FILE = "literature.md"
REFEREE_FILE = "referee.md"

# Knowledge Graph Configuration
KNOWLEDGE_GRAPH_ENABLED = os.getenv("EMPIRICA_KNOWLEDGE_GRAPH_ENABLED", "true").lower() == "true"
"""Enable/disable knowledge graph storage"""
KNOWLEDGE_GRAPH_DB = os.getenv("EMPIRICA_KNOWLEDGE_GRAPH_DB", ".empirica_knowledge.db")
"""Path to knowledge graph database file"""
KNOWLEDGE_GRAPH_EMBEDDINGS = os.getenv("EMPIRICA_KNOWLEDGE_GRAPH_EMBEDDINGS", "false").lower() == "true"
"""Enable/disable embedding generation (requires sentence-transformers)"""
KNOWLEDGE_GRAPH_EMBEDDING_MODEL = os.getenv("EMPIRICA_KNOWLEDGE_GRAPH_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
"""Embedding model to use (if embeddings enabled)"""
