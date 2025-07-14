"""
Graph storage backend for the Research Knowledge Graph.

    # code cleanup
Provides persistence layer using SQLite with graph tables for storing
nodes, edges, and project metadata.
    # fix memory leak
"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from uuid import UUID, uuid4
import pickle
import base64

from .models import (
    ResearchNode,
    ResearchEdge,
    ProjectGraph,
    KnowledgeGraph,
    NodeType,
    EdgeType,
)
from ..logger import get_logger

logger = get_logger(__name__)


class GraphStorage:
    """
    SQLite-based storage backend for the Research Knowledge Graph.
    
    Stores nodes, edges, and project metadata in a SQLite database with
    support for embeddings and efficient querying.
    """
    
    def __init__(self, db_path: str | Path):
        """
        Initialize the graph storage.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize the database schema if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Projects table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                project_id TEXT PRIMARY KEY,
                project_name TEXT,
                metadata TEXT,
                success INTEGER,
                quality_score REAL,
                execution_time REAL,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        
        # Nodes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                project_id TEXT,
                node_type TEXT NOT NULL,
                content TEXT NOT NULL,
                title TEXT,
                metadata TEXT,
                embedding BLOB,
                embedding_model TEXT,
                created_at TEXT,
                updated_at TEXT,
                FOREIGN KEY (project_id) REFERENCES projects(project_id)
            )
        """)
        
        # Edges table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                edge_id TEXT PRIMARY KEY,
                project_id TEXT,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                metadata TEXT,
                created_at TEXT,
                FOREIGN KEY (project_id) REFERENCES projects(project_id),
                FOREIGN KEY (source_id) REFERENCES nodes(node_id),
                FOREIGN KEY (target_id) REFERENCES nodes(node_id)
            )
        """)
        
        # Create indexes for efficient querying
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_project ON nodes(project_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_project ON edges(project_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type)")
        
        conn.commit()
        conn.close()
        logger.info(f"Initialized graph storage database at {self.db_path}")
    
    def save_project(self, project: ProjectGraph) -> None:
        """
        Save a project graph to the database.
        
        Args:
            project: The project graph to save
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Save project metadata
            cursor.execute("""
                INSERT OR REPLACE INTO projects 
                (project_id, project_name, metadata, success, quality_score, execution_time, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(project.project_id),
                project.project_name,
                json.dumps(project.metadata),
                1 if project.success else 0 if project.success is False else None,
                project.quality_score,
                project.execution_time,
                project.created_at.isoformat(),
                project.updated_at.isoformat(),
            ))
            
            # Save nodes
            for node in project.nodes.values():
                embedding_blob = None
                if node.embedding:
                    # Serialize embedding as binary
                    embedding_blob = pickle.dumps(node.embedding)
                
                cursor.execute("""
                    INSERT OR REPLACE INTO nodes
                    (node_id, project_id, node_type, content, title, metadata, embedding, embedding_model, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(node.id),
                    str(node.project_id),
                    node.node_type,
                    node.content,
                    node.title,
                    json.dumps(node.metadata),
                    embedding_blob,
                    node.embedding_model,
                    node.created_at.isoformat(),
                    node.updated_at.isoformat(),
                ))
            
            # Save edges
            for edge in project.edges:
                cursor.execute("""
                    INSERT OR REPLACE INTO edges
                    (edge_id, project_id, source_id, target_id, edge_type, weight, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(edge.id),
                    str(edge.project_id),
                    str(edge.source_id),
                    str(edge.target_id),
                    edge.edge_type,
                    edge.weight,
                    json.dumps(edge.metadata),
                    edge.created_at.isoformat(),
                ))
            
            conn.commit()
            logger.debug(f"Saved project {project.project_id} to database")
        
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to save project {project.project_id}: {e}", exc_info=True)
            raise
        finally:
            conn.close()
    
    def load_project(self, project_id: UUID) -> Optional[ProjectGraph]:
        """
        Load a project graph from the database.
        
        Args:
            project_id: The ID of the project to load
            
        Returns:
            The project graph, or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            # Load project metadata
            cursor.execute("SELECT * FROM projects WHERE project_id = ?", (str(project_id),))
            project_row = cursor.fetchone()
            
            if not project_row:
                return None
            
            # Reconstruct project
            project = ProjectGraph(
                project_id=UUID(project_row["project_id"]),
                project_name=project_row["project_name"],
                metadata=json.loads(project_row["metadata"] or "{}"),
                success=bool(project_row["success"]) if project_row["success"] is not None else None,
                quality_score=project_row["quality_score"],
                execution_time=project_row["execution_time"],
                created_at=datetime.fromisoformat(project_row["created_at"]),
                updated_at=datetime.fromisoformat(project_row["updated_at"]),
            )
            
            # Load nodes
            cursor.execute("SELECT * FROM nodes WHERE project_id = ?", (str(project_id),))
            for node_row in cursor.fetchall():
                embedding = None
                if node_row["embedding"]:
                    embedding = pickle.loads(node_row["embedding"])
                
                node = ResearchNode(
                    id=UUID(node_row["node_id"]),
                    node_type=NodeType(node_row["node_type"]),
                    content=node_row["content"],
                    title=node_row["title"],
                    metadata=json.loads(node_row["metadata"] or "{}"),
                    embedding=embedding,
                    embedding_model=node_row["embedding_model"],
                    project_id=UUID(node_row["project_id"]),
                    created_at=datetime.fromisoformat(node_row["created_at"]),
                    updated_at=datetime.fromisoformat(node_row["updated_at"]),
                )
                project.nodes[node.id] = node
            
            # Load edges
            cursor.execute("SELECT * FROM edges WHERE project_id = ?", (str(project_id),))
            for edge_row in cursor.fetchall():
                edge = ResearchEdge(
                    id=UUID(edge_row["edge_id"]),
                    source_id=UUID(edge_row["source_id"]),
                    target_id=UUID(edge_row["target_id"]),
                    edge_type=EdgeType(edge_row["edge_type"]),
                    weight=edge_row["weight"],
                    metadata=json.loads(edge_row["metadata"] or "{}"),
                    project_id=UUID(edge_row["project_id"]),
                    created_at=datetime.fromisoformat(edge_row["created_at"]),
                )
                project.edges.append(edge)
            
            logger.debug(f"Loaded project {project_id} from database")
            return project
        
        except Exception as e:
            logger.error(f"Failed to load project {project_id}: {e}", exc_info=True)
            raise
        finally:
            conn.close()
    
    def list_projects(self) -> List[UUID]:
        """
        List all project IDs in the database.
        
        Returns:
            List of project UUIDs
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT project_id FROM projects ORDER BY created_at DESC")
            return [UUID(row[0]) for row in cursor.fetchall()]
        finally:
            conn.close()
    
    def delete_project(self, project_id: UUID) -> bool:
        """
        Delete a project and all its nodes and edges from the database.
        
        Args:
            project_id: The ID of the project to delete
            
        Returns:
            True if deleted, False if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Delete edges first (foreign key constraint)
            cursor.execute("DELETE FROM edges WHERE project_id = ?", (str(project_id),))
            # Delete nodes
            cursor.execute("DELETE FROM nodes WHERE project_id = ?", (str(project_id),))
            # Delete project
            cursor.execute("DELETE FROM projects WHERE project_id = ?", (str(project_id),))
            
            conn.commit()
            deleted = cursor.rowcount > 0
            logger.debug(f"Deleted project {project_id} from database")
            return deleted
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to delete project {project_id}: {e}", exc_info=True)
            raise
        finally:
            conn.close()
    
    def get_all_nodes(self, node_type: Optional[NodeType] = None) -> List[ResearchNode]:
        """
        Get all nodes from all projects, optionally filtered by type.
        
        Args:
            node_type: Optional filter by node type
            
        Returns:
            List of nodes
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            if node_type:
                cursor.execute("SELECT * FROM nodes WHERE node_type = ?", (node_type.value,))
            else:
                cursor.execute("SELECT * FROM nodes")
            
            nodes = []
            for row in cursor.fetchall():
                embedding = None
                if row["embedding"]:
                    embedding = pickle.loads(row["embedding"])
                
                node = ResearchNode(
                    id=UUID(row["node_id"]),
                    node_type=NodeType(row["node_type"]),
                    content=row["content"],
                    title=row["title"],
                    metadata=json.loads(row["metadata"] or "{}"),
                    embedding=embedding,
                    embedding_model=row["embedding_model"],
                    project_id=UUID(row["project_id"]) if row["project_id"] else None,
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                )
                nodes.append(node)
            
            return nodes
        finally:
            conn.close()
    
    def get_all_edges(self, edge_type: Optional[EdgeType] = None) -> List[ResearchEdge]:
        """
        Get all edges from all projects, optionally filtered by type.
        
        Args:
            edge_type: Optional filter by edge type
            
        Returns:
            List of edges
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            if edge_type:
                cursor.execute("SELECT * FROM edges WHERE edge_type = ?", (edge_type.value,))
            else:
                cursor.execute("SELECT * FROM edges")
            
            edges = []
            for row in cursor.fetchall():
                edge = ResearchEdge(
                    id=UUID(row["edge_id"]),
                    source_id=UUID(row["source_id"]),
                    target_id=UUID(row["target_id"]),
                    edge_type=EdgeType(row["edge_type"]),
                    weight=row["weight"],
                    metadata=json.loads(row["metadata"] or "{}"),
                    project_id=UUID(row["project_id"]) if row["project_id"] else None,
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
                edges.append(edge)
            
            return edges
        finally:
            conn.close()

