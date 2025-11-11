# Empirica

[![Version](https://img.shields.io/pypi/v/empirica.svg)](https://pypi.python.org/pypi/empirica) [![Python Version](https://img.shields.io/badge/python-%3E%3D3.12-blue.svg)](https://www.python.org/downloads/) [![PyPI - Downloads](https://img.shields.io/pypi/dm/empirica)](https://pypi.python.org/pypi/empirica) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

> **An intelligent multi-agent system that transforms scientific research workflows through AI-powered automation, knowledge graphs, and meta-learning capabilities.**

Empirica is a comprehensive research assistant that automates the entire scientific research pipeline—from initial data analysis to generating publication-ready papers. Built with advanced AI agents, it combines multiple LLM backends with a sophisticated knowledge graph system that learns from past research to improve future outcomes.

## Why Empirica?

- **End-to-End Automation**: Transform raw data into complete research papers with minimal manual intervention
- **Intelligent Knowledge Graph**: Builds a persistent knowledge base that captures relationships between ideas, methods, datasets, and results
- **Meta-Learning System**: Learns from past projects to predict success, recommend methods, and avoid common pitfalls
- **Reproducibility Built-In**: Automatic versioning, checkpointing, and provenance tracking for every experiment
- **Multi-Backend Support**: Choose between fast LangGraph mode or detailed cmbagent mode for different use cases
- **Smart Suggestions**: Get AI-powered recommendations at every stage of your research workflow

## Key Features

### 🤖 Multi-Agent Research Pipeline
- **Idea Generation**: AI agents collaborate to generate and refine research ideas
- **Methodology Development**: Automated creation of detailed research methodologies
- **Experiment Execution**: Intelligent agents execute experiments and generate results
- **Paper Writing**: Generate publication-ready papers in various journal styles

### 🧠 Research Knowledge Graph
- **Structured Knowledge**: Captures ideas, methods, datasets, tools, and results as interconnected nodes
- **Similarity Search**: Find related past projects and learn from similar research
- **Pattern Analysis**: Identify what works and what doesn't across your research portfolio
- **Semantic Embeddings**: Advanced similarity matching using vector embeddings

### 📊 Meta-Learning & Intelligence
- **Success Prediction**: Predict project success probability before starting
- **Method Recommendations**: Get AI-suggested methodologies based on past successes
- **Failure Prevention**: Early warnings about potential issues based on historical patterns
- **Execution Time Estimation**: Predict how long experiments will take

### 🔄 Reproducibility & Versioning
- **Automatic Versioning**: Git-like version control for ideas, methods, and results
- **Checkpoint System**: Save and restore experiment state at any point
- **Provenance Tracking**: Complete audit trail of inputs, code, and configurations
- **One-Click Reproduction**: Recreate any past project with a single command

### 💡 Intelligent Suggestions
- **Context-Aware Recommendations**: Get suggestions tailored to your current research stage
- **Tool Recommendations**: Discover useful tools and libraries for your domain
- **Optimization Hints**: Receive suggestions for improving your methodology
- **Warning System**: Alerts about potential issues before they become problems

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv empirica_env
source empirica_env/bin/activate  # On Windows: empirica_env\Scripts\activate

# Install Empirica
pip install empirica
```

For the GUI application:
```bash
pip install "empirica[app]"
```

### Basic Usage

```python
from empirica import Empirica

# Initialize Empirica
emp = Empirica(project_dir="my_research_project")

# Describe your data and tools
emp.set_data_description("""
    Analyze experimental data in data.csv using sklearn and pandas.
    Dataset contains 1000 time-series measurements from a particle detector.
""")

# Generate research idea (fast mode using LangGraph)
emp.get_idea(mode="fast")

# Generate methodology
emp.get_method(mode="fast")

# Execute experiments and get results (requires cmbagent)
emp.get_results()

# Generate publication-ready paper
from empirica import Journal
emp.get_paper(journal=Journal.APS)
```

### GUI Application

Launch the interactive web interface:

```bash
empirica run
```

This starts a Streamlit-based GUI where you can manage your research projects visually.

## Advanced Features

### Knowledge Graph Integration

Empirica automatically builds a knowledge graph of your research:

```python
# Find similar past projects
similar = emp.find_similar_projects("machine learning time series")

# Get intelligent suggestions
suggestions = emp.get_suggestions(stage="method")

# Reproduce a past project
reproduced = emp.reproduce_project(project_id="...")
```

### Version Control & Checkpoints

```python
# Automatic versioning happens during workflow
# Access versions programmatically
versions = emp.version_control.list_versions("idea")
diff = emp.version_control.diff_versions("idea", v1=1, v2=2)

# Manual checkpoints
emp.checkpoint_manager.save_checkpoint("before_experiment")
emp.checkpoint_manager.load_checkpoint("before_experiment")
```

### Meta-Learning Insights

```python
# Get predictions about your project
success_prob = emp.meta_agent.models.predict_success(
    idea="...",
    method="...",
    domain="machine learning"
)

# Get method recommendations
recommendations = emp.meta_agent.models.recommend_methods(
    idea="...",
    domain="..."
)
```

## Installation Options

### Standard Installation

```bash
pip install empirica
```

### With GUI

```bash
pip install "empirica[app]"
```

### Development Installation

```bash
git clone https://github.com/Saharsha-N/Empirica.git
cd Empirica
pip install -e .
```

### Using uv

```bash
uv add empirica
# or for development
uv sync
```

## Optional Dependencies

### cmbagent

For advanced planning and detailed experiment execution, install `cmbagent`:

```bash
pip install cmbagent>=0.0.1post63
```

**Note**: On Windows with Python 3.13, `cmbagent` installation may fail due to dependency build issues. We recommend using Python 3.12 for full compatibility. See [INSTALL_CMBAGENT.md](INSTALL_CMBAGENT.md) for detailed instructions.

The codebase works without `cmbagent`—you can use `mode="fast"` for idea and method generation, which uses LangGraph instead.

## Project Structure

```
empirica/
├── empirica.py          # Main Empirica class
├── idea.py              # Idea generation
├── method.py            # Methodology generation
├── experiment.py        # Experiment execution
├── knowledge_graph/     # Knowledge graph system
│   ├── models.py        # Graph data models
│   ├── storage.py       # SQLite storage backend
│   ├── extractor.py     # Knowledge extraction
│   └── query.py         # Graph queries
├── meta_learning/       # Meta-learning system
│   ├── analyzer.py      # Pattern analysis
│   ├── models.py        # Predictive models
│   └── agent.py         # Meta-learning agent
├── suggestions/         # Suggestion engine
├── reproducibility/    # Versioning & checkpoints
└── paper_agents/       # Paper generation
```

## Documentation

- **Full Documentation**: [https://empirica.readthedocs.io/](https://empirica.readthedocs.io/)
- **API Reference**: See `docs/api_ref/` for detailed API documentation
- **Examples**: Check `examples/` directory for usage examples
- **Tutorials**: Step-by-step guides in `docs/tutorials/`

## Examples

### Complete Workflow

```python
from empirica import Empirica, Journal

# Initialize
emp = Empirica(project_dir="research_project")

# Set up data description
emp.set_data_description("""
    Dataset: sensor_data.csv
    Tools: pandas, numpy, scikit-learn, matplotlib
    Domain: Time-series analysis, Anomaly detection
""")

# Generate idea
emp.get_idea(mode="fast")

# Generate methodology
emp.get_method(mode="fast")

# Execute (requires cmbagent)
emp.get_results(
    engineer_model="gpt-4o",
    researcher_model="gpt-4o"
)

# Generate paper
emp.get_paper(journal=Journal.APS)
```

### Using Knowledge Graph

```python
# After running several projects, query the knowledge graph
similar_projects = emp.find_similar_projects(
    idea="time series anomaly detection"
)

# Get suggestions for your current project
suggestions = emp.get_suggestions(stage="method")
print(suggestions)

# Access meta-learning insights
insights = emp.meta_agent.get_insights()
```

## Testing

Run the comprehensive test suite:

```bash
# Install test dependencies
pip install -e ".[test]"

# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_unit/          # Unit tests
pytest tests/test_integration/   # Integration tests
pytest tests/test_e2e/          # End-to-end tests
```

## Docker

Run Empirica in a Docker container:

```bash
# Pull the image
docker pull pablovd/empirica:latest

# Run GUI
docker run -p 8501:8501 --rm pablovd/empirica:latest

# Interactive mode
docker run --rm -it pablovd/empirica:latest bash
```

See the [Docker documentation](https://empirica.readthedocs.io/en/latest/docker/) for more details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue for bugs, questions, or suggestions.

## Citation

If you use Empirica in your research, please cite:

```bibtex
@article{villaescusanavarro2025empiricaprojectdeepknowledge,
         title={The Empirica project: Deep knowledge AI agents for scientific discovery}, 
         author={Francisco Villaescusa-Navarro and Boris Bolliet and Pablo Villanueva-Domingo and others},
         year={2025},
         eprint={2510.26887},
         archivePrefix={arXiv},
         primaryClass={cs.AI},
         url={https://arxiv.org/abs/2510.26887},
}

@software{Empirica_2025,
          author = {Pablo Villanueva-Domingo, Francisco Villaescusa-Navarro, Boris Bolliet},
          title = {Empirica: Modular Multi-Agent System for Scientific Research Assistance},
          year = {2025},
          url = {https://github.com/Saharsha-N/Empirica},
          version = {latest}
}
```

## License

[GNU GENERAL PUBLIC LICENSE (GPLv3)](https://www.gnu.org/licenses/gpl-3.0.html)

Copyright (C) 2025 - Empirica Contributors

## Acknowledgments

Empirica builds upon several excellent open-source projects:
- [cmbagent](https://github.com/CMBAgents/cmbagent) - Multi-agent planning and control
- [LangGraph](https://www.langchain.com/langgraph) - Agent orchestration
- [LangChain](https://www.langchain.com/) - LLM integration framework

## Support

For questions, issues, or contributions:
- **GitHub Issues**: [https://github.com/Saharsha-N/Empirica/issues](https://github.com/Saharsha-N/Empirica/issues)
- **Documentation**: [https://empirica.readthedocs.io/](https://empirica.readthedocs.io/)

---

**Empirica** - Transforming scientific research through intelligent automation.
