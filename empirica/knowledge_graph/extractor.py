"""
Knowledge extraction engine for the Research Knowledge Graph.

Extracts structured knowledge from Research objects and converts them
into graph structures with nodes and edges.
    # add retry logic
    # fix typo
"""

import re
    # refactor for better readability
from typing import Dict, List, Optional, Any
from uuid import uuid4
from pathlib import Path

from .models import (
    ResearchNode,
    ResearchEdge,
    ProjectGraph,
    NodeType,
    EdgeType,
)
from ..research import Research
from ..logger import get_logger

logger = get_logger(__name__)


class KnowledgeExtractor:
    """
    Extracts structured knowledge from Research objects into graph format.
    
    Parses ideas, methods, datasets, tools, and results from Research objects
    and creates a ProjectGraph with appropriate nodes and relationships.
    """
    
    def __init__(self, generate_embeddings: bool = False, embedding_model: Optional[str] = None):
        """
        Initialize the knowledge extractor.
        
        Args:
            generate_embeddings: Whether to generate embeddings for nodes
            embedding_model: Model to use for embeddings (if None, embeddings disabled)
        """
        self.generate_embeddings = generate_embeddings
        self.embedding_model = embedding_model
        self._embedding_function = None
        
        if generate_embeddings and embedding_model:
            self._init_embedding_model()
    
    def _init_embedding_model(self) -> None:
        """Initialize the embedding model."""
        try:
            # Try to use sentence-transformers if available
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_function = SentenceTransformer(self.embedding_model)
                logger.info(f"Initialized embedding model: {self.embedding_model}")
            except ImportError:
                logger.warning("sentence-transformers not available, embeddings disabled")
                self.generate_embeddings = False
        except Exception as e:
            logger.warning(f"Failed to initialize embedding model: {e}")
            self.generate_embeddings = False
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if generation fails
        """
        if not self.generate_embeddings or not self._embedding_function:
            return None
        
        try:
            embedding = self._embedding_function.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}")
            return None
    
    def extract_from_research(
        self,
        research: Research,
        project_id: Optional[str] = None,
        project_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProjectGraph:
        """
        Extract knowledge graph from a Research object.
        
        Args:
            research: The Research object to extract from
            project_id: Optional project ID (generated if not provided)
            project_name: Optional project name
            metadata: Optional project metadata
            
        Returns:
            A ProjectGraph containing all extracted knowledge
        """
        from uuid import UUID
        
        if project_id:
            project_uuid = UUID(project_id) if isinstance(project_id, str) else project_id
        else:
            project_uuid = uuid4()
        
        project = ProjectGraph(
            project_id=project_uuid,
            project_name=project_name,
            metadata=metadata or {},
        )
        
        # Extract idea node
        if research.idea:
            idea_node = self._extract_idea(research.idea, project_uuid)
            project.add_node(idea_node)
        
        # Extract method node
        if research.methodology:
            method_node = self._extract_method(research.methodology, project_uuid)
            project.add_node(method_node)
            
            # Create edge: idea -> method
            if research.idea:
                idea_id = next((n.id for n in project.nodes.values() if n.node_type == NodeType.IDEA), None)
                if idea_id:
                    edge = ResearchEdge(
                        source_id=idea_id,
                        target_id=method_node.id,
                        edge_type=EdgeType.GENERATES,
                        project_id=project_uuid,
                    )
                    project.add_edge(edge)
        
        # Extract dataset nodes from data description
        if research.data_description:
            dataset_nodes = self._extract_datasets(research.data_description, project_uuid)
            for dataset_node in dataset_nodes:
                project.add_node(dataset_node)
                
                # Create edge: idea uses dataset
                if research.idea:
                    idea_id = next((n.id for n in project.nodes.values() if n.node_type == NodeType.IDEA), None)
                    if idea_id:
                        edge = ResearchEdge(
                            source_id=idea_id,
                            target_id=dataset_node.id,
                            edge_type=EdgeType.USES,
                            project_id=project_uuid,
                        )
                        project.add_edge(edge)
        
        # Extract tool nodes from data description and methodology
        tool_nodes = []
        if research.data_description:
            tool_nodes.extend(self._extract_tools(research.data_description, project_uuid))
        if research.methodology:
            tool_nodes.extend(self._extract_tools(research.methodology, project_uuid))
        
        # Deduplicate tools
        seen_tools = set()
        for tool_node in tool_nodes:
            if tool_node.content not in seen_tools:
                project.add_node(tool_node)
                seen_tools.add(tool_node.content)
                
                # Create edge: method uses tool
                if research.methodology:
                    method_id = next((n.id for n in project.nodes.values() if n.node_type == NodeType.METHOD), None)
                    if method_id:
                        edge = ResearchEdge(
                            source_id=method_id,
                            target_id=tool_node.id,
                            edge_type=EdgeType.USES,
                            project_id=project_uuid,
                        )
                        project.add_edge(edge)
        
        # Extract result node
        if research.results:
            result_node = self._extract_result(research.results, research.plot_paths, project_uuid)
            project.add_node(result_node)
            
            # Create edge: method generates result
            if research.methodology:
                method_id = next((n.id for n in project.nodes.values() if n.node_type == NodeType.METHOD), None)
                if method_id:
                    edge = ResearchEdge(
                        source_id=method_id,
                        target_id=result_node.id,
                        edge_type=EdgeType.GENERATES,
                        project_id=project_uuid,
                    )
                    project.add_edge(edge)
            
            # Create edge: idea leads to result
            if research.idea:
                idea_id = next((n.id for n in project.nodes.values() if n.node_type == NodeType.IDEA), None)
                if idea_id:
                    edge = ResearchEdge(
                        source_id=idea_id,
                        target_id=result_node.id,
                        edge_type=EdgeType.LEADS_TO,
                        project_id=project_uuid,
                    )
                    project.add_edge(edge)
        
        # Extract findings from results
        if research.results:
            finding_nodes = self._extract_findings(research.results, project_uuid)
            for finding_node in finding_nodes:
                project.add_node(finding_node)
                
                # Create edge: result contains finding
                result_id = next((n.id for n in project.nodes.values() if n.node_type == NodeType.RESULT), None)
                if result_id:
                    edge = ResearchEdge(
                        source_id=result_id,
                        target_id=finding_node.id,
                        edge_type=EdgeType.PART_OF,
                        project_id=project_uuid,
                    )
                    project.add_edge(edge)
        
        logger.info(f"Extracted knowledge graph with {len(project.nodes)} nodes and {len(project.edges)} edges")
        return project
    
    def _extract_idea(self, idea_text: str, project_id) -> ResearchNode:
        """
        Extract an idea node from idea text.
        
        Args:
            idea_text: The idea text
            project_id: Project UUID
            
        Returns:
            ResearchNode of type IDEA
        """
        # Try to extract title (first line or sentence)
        title = None
        lines = idea_text.strip().split('\n')
        if lines:
            first_line = lines[0].strip()
            if len(first_line) < 200:  # Reasonable title length
                title = first_line
        
        # Extract domain/keywords from idea
        metadata = {
            "text_length": len(idea_text),
            "has_title": title is not None,
        }
        
        # Try to identify domain (simple keyword matching)
        domain_keywords = {
            "cosmology": ["cosmology", "cosmic", "universe", "galaxy", "dark matter", "dark energy"],
            "biology": ["biology", "biological", "organism", "cell", "protein", "dna"],
            "chemistry": ["chemistry", "chemical", "molecule", "reaction", "compound"],
            "material science": ["material", "crystal", "lattice", "structure"],
            "machine learning": ["machine learning", "neural network", "model", "training", "algorithm"],
        }
        
        idea_lower = idea_text.lower()
        for domain, keywords in domain_keywords.items():
            if any(kw in idea_lower for kw in keywords):
                metadata["domain"] = domain
                break
        
        node = ResearchNode(
            node_type=NodeType.IDEA,
            content=idea_text,
            title=title,
            metadata=metadata,
            project_id=project_id,
        )
        
        if self.generate_embeddings:
            embedding = self._generate_embedding(idea_text)
            if embedding:
                node.set_embedding(embedding, self.embedding_model or "unknown")
        
        return node
    
    def _extract_method(self, method_text: str, project_id) -> ResearchNode:
        """
        Extract a method node from methodology text.
        
        Args:
            method_text: The methodology text
            project_id: Project UUID
            
        Returns:
            ResearchNode of type METHOD
        """
        # Extract steps (numbered or bulleted lists)
        steps = []
        step_pattern = r'(?:^\d+[\.\)]\s+|^[-*]\s+)(.+?)(?=\n(?:^\d+[\.\)]\s+|^[-*]\s+)|$)'
        matches = re.findall(step_pattern, method_text, re.MULTILINE)
        if matches:
            steps = [m.strip() for m in matches]
        
        metadata = {
            "text_length": len(method_text),
            "num_steps": len(steps),
            "steps": steps[:10],  # Store first 10 steps
        }
        
        node = ResearchNode(
            node_type=NodeType.METHOD,
            content=method_text,
            title="Research Methodology",
            metadata=metadata,
            project_id=project_id,
        )
        
        if self.generate_embeddings:
            embedding = self._generate_embedding(method_text)
            if embedding:
                node.set_embedding(embedding, self.embedding_model or "unknown")
        
        return node
    
    def _extract_datasets(self, data_description: str, project_id) -> List[ResearchNode]:
        """
        Extract dataset nodes from data description.
        
        Args:
            data_description: The data description text
            project_id: Project UUID
            
        Returns:
            List of ResearchNode of type DATASET
        """
        nodes = []
        
        # Extract file paths (absolute paths mentioned in markdown or text)
        # Match paths like /data/file.csv or /path/to/file.h5
        file_pattern = r'(?:^|\s)(/[^\s,]+\.(?:csv|txt|json|h5|hdf5|fits|npy|npz|parquet|pkl|pickle))'
        matches = re.findall(file_pattern, data_description, re.IGNORECASE)
        
        for file_path in matches:
            path_obj = Path(file_path)
            node = ResearchNode(
                node_type=NodeType.DATASET,
                content=file_path,
                title=path_obj.name,
                metadata={
                    "file_path": file_path,
                    "file_extension": path_obj.suffix,
                    "file_name": path_obj.name,
                },
                project_id=project_id,
            )
            nodes.append(node)
        
        # Also look for dataset mentions in text
        dataset_keywords = ["dataset", "data file", "data set", "training data", "test data"]
        for keyword in dataset_keywords:
            if keyword.lower() in data_description.lower():
                # Extract context around keyword
                pattern = rf'{keyword}[:\s]+([^\n\.]+)'
                matches = re.findall(pattern, data_description, re.IGNORECASE)
                for match in matches[:3]:  # Limit to 3 matches
                    match = match.strip()
                    if len(match) > 10 and match not in [n.content for n in nodes]:
                        node = ResearchNode(
                            node_type=NodeType.DATASET,
                            content=match,
                            title=f"Dataset: {match[:50]}",
                            metadata={"extracted_from": "text_mention"},
                            project_id=project_id,
                        )
                        nodes.append(node)
        
        return nodes
    
    def _extract_tools(self, text: str, project_id) -> List[ResearchNode]:
        """
        Extract tool/library nodes from text.
        
        Args:
            text: Text to extract tools from
            project_id: Project UUID
            
        Returns:
            List of ResearchNode of type TOOL
        """
        nodes = []
        
        # Common scientific tools and libraries
        tool_patterns = [
            r'\b(pandas|numpy|scipy|sklearn|scikit-learn|matplotlib|seaborn|plotly)\b',
            r'\b(tensorflow|pytorch|keras|jax)\b',
            r'\b(astropy|scipy\.stats|scipy\.optimize)\b',
            r'\b(jupyter|ipython|notebook)\b',
            r'\b(git|docker|conda|pip)\b',
        ]
        
        seen_tools = set()
        for pattern in tool_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for tool in matches:
                tool_lower = tool.lower()
                if tool_lower not in seen_tools:
                    seen_tools.add(tool_lower)
                    node = ResearchNode(
                        node_type=NodeType.TOOL,
                        content=tool_lower,
                        title=tool,
                        metadata={"extracted_from": "pattern_match"},
                        project_id=project_id,
                    )
                    nodes.append(node)
        
        # Also look for import statements
        import_pattern = r'^(?:import|from)\s+([a-zA-Z0-9_\.]+)'
        matches = re.findall(import_pattern, text, re.MULTILINE)
        for module in matches[:20]:  # Limit to 20 imports
            module_clean = module.split('.')[0]  # Get base module
            if module_clean not in seen_tools and len(module_clean) > 2:
                seen_tools.add(module_clean)
                node = ResearchNode(
                    node_type=NodeType.TOOL,
                    content=module_clean,
                    title=module_clean,
                    metadata={"extracted_from": "import_statement"},
                    project_id=project_id,
                )
                nodes.append(node)
        
        return nodes
    
    def _extract_result(self, results_text: str, plot_paths: List[str], project_id) -> ResearchNode:
        """
        Extract a result node from results text.
        
        Args:
            results_text: The results text
            plot_paths: List of plot file paths
            project_id: Project UUID
            
        Returns:
            ResearchNode of type RESULT
        """
        metadata = {
            "text_length": len(results_text),
            "num_plots": len(plot_paths),
            "plot_paths": plot_paths,
        }
        
        # Try to extract key metrics/numbers
        number_pattern = r'\b(\d+\.?\d*)\s*(?:%|percent|sigma|σ)'
        metrics = re.findall(number_pattern, results_text)
        if metrics:
            metadata["metrics"] = [float(m) for m in metrics[:10]]
        
        node = ResearchNode(
            node_type=NodeType.RESULT,
            content=results_text,
            title="Research Results",
            metadata=metadata,
            project_id=project_id,
        )
        
        if self.generate_embeddings:
            embedding = self._generate_embedding(results_text)
            if embedding:
                node.set_embedding(embedding, self.embedding_model or "unknown")
        
        return node
    
    def _extract_findings(self, results_text: str, project_id) -> List[ResearchNode]:
        """
        Extract finding nodes from results text.
        
        Args:
            results_text: The results text
            project_id: Project UUID
            
        Returns:
            List of ResearchNode of type FINDING
        """
        nodes = []
        
        # Extract sentences that look like findings (contain key phrases)
        finding_indicators = [
            r'we found that',
            r'results show',
            r'analysis reveals',
            r'we observe',
            r'we conclude',
            r'significant',
            r'evidence suggests',
        ]
        
        sentences = re.split(r'[.!?]\s+', results_text)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Reasonable sentence length
                for indicator in finding_indicators:
                    if indicator.lower() in sentence.lower():
                        node = ResearchNode(
                            node_type=NodeType.FINDING,
                            content=sentence,
                            title=sentence[:60] + "..." if len(sentence) > 60 else sentence,
                            metadata={"extracted_from": "results_text"},
                            project_id=project_id,
                        )
                        nodes.append(node)
                        break
        
        # Limit to top 10 findings
        return nodes[:10]

