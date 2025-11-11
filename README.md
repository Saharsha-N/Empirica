# Empirica

Empirica is a multiagent system designed to be a scientific research assistant. Empirica implements AI agents with [AG2](https://ag2.ai/) and [LangGraph](https://www.langchain.com/langgraph), using [cmbagent](https://github.com/CMBAgents/cmbagent) as the research analysis backend.

## Installation

To install empirica create a virtual environment and pip install it. We recommend using Python 3.12:

```bash
python -m venv Empirica_env
source Empirica_env/bin/activate
pip install "empirica[app]"
```

Or alternatively install it with [uv](https://docs.astral.sh/uv/), initializing a project and installing it:

```bash
uv init
uv add empirica[app]
```

Then, run the gui with:

```
empirica run
```

### Optional: Installing cmbagent

`cmbagent` is an optional dependency that provides advanced planning and control for research analysis. The codebase works without it, but some features (like `get_results()` and `mode="cmbagent"`) require it. 

**Note**: On Windows with Python 3.13, `cmbagent` installation may fail due to dependency build issues. We recommend using Python 3.12 for full compatibility. See [INSTALL_CMBAGENT.md](INSTALL_CMBAGENT.md) for detailed installation instructions and troubleshooting.

## Get started

Initialize a `Empirica` instance and describe the data and tools to be employed.

```python
from empirica import Empirica

emp = Empirica(project_dir="project_dir")

prompt = """
Analyze the experimental data stored in data.csv using sklearn and pandas.
This data includes time-series measurements from a particle detector.
"""

emp.set_data_description(prompt)
```

Generate a research idea from that data specification.

```python
emp.get_idea()
```

Generate the methodology required for working on that idea.

```python
emp.get_method()
```

With the methodology setup, perform the required computations and get the plots and results.

```python
emp.get_results()
```

Finally, generate a latex article with the results. You can specify the journal style, in this example we choose the [APS (Physical Review Journals)](https://journals.aps.org/) style.

```python
from empirica import Journal

emp.get_paper(journal=Journal.APS)
```

You can also manually provide any info as a string or markdown file in an intermediate step, using the `set_idea`, `set_method` or `set_results` methods. For instance, for providing a file with the methodology developed by the user:

```python
emp.set_method(path_to_the_method_file.md)
```

## EmpiricaApp

You can run Empirica using a GUI through the [EmpiricaApp](https://github.com/AstroPilot-AI/EmpiricaApp).

The app is already installed with `pip install "empirica[app]"`, otherwise install it with `pip install empirica_app` or `uv sync --extra app`.

Then, launch the GUI with

```bash
empirica run
```

Test a [deployed demo of the app in HugginFace Spaces](https://huggingface.co/spaces/astropilot-ai/Empirica).

## Build from source

### pip

You will need python 3.12 or higher installed. Clone Empirica:

```bash
git clone https://github.com/AstroPilot-AI/Empirica.git
cd Empirica
```

Create and activate a virtual environment

```bash
python3 -m venv Empirica_env
source Empirica_env/bin/activate
```

And install the project

```bash
pip install -e .
```

### uv

You can also install the project using [uv](https://docs.astral.sh/uv/), just running:

```bash
uv sync
```

which will create the virtual environment and install the dependencies and project. Activate the virtual environment if needed with

```bash
source .venv/bin/activate
```

## Docker

You can run Empirica in a [Docker](https://www.docker.com/) image, which includes all the required dependencies for Empirica including LaTeX. Pull the image with:

```bash
docker pull pablovd/empirica:latest
```

Once built, you can run the GUI with

```bash
docker run -p 8501:8501 --rm pablovd/empirica:latest
```

or in interactive mode with

```bash
docker run --rm -it pablovd/empirica:latest bash
```

Share volumes with `-v $(pwd)/project:/app/project` for inputing data and accessing to it. You can also share the API keys with a `.env` file in the same folder with `-v $(pwd).env/app/.env`.

You can also build an image locally with

```bash
docker build -f docker/Dockerfile.dev -t empirica_src .
```

Read more information on how to use the Docker images in the [documentation](https://empirica.readthedocs.io/en/latest/docker/).

## Contributing

Pull requests are welcome! Feel free to open an issue for bugs, comments, questions and suggestions.

<!-- ## Citation

If you use this library please link this repository and cite [arXiv:2506.xxxxx](arXiv:x2506.xxxxx). -->

## Citation

If you make use of Empirica, please cite the following references:

```bibtex
@article{villaescusanavarro2025empiricaprojectdeepknowledge,
         title={The Empirica project: Deep knowledge AI agents for scientific discovery}, 
         author={Francisco Villaescusa-Navarro and Boris Bolliet and Pablo Villanueva-Domingo and Adrian E. Bayer and Aidan Acquah and Chetana Amancharla and Almog Barzilay-Siegal and Pablo Bermejo and Camille Bilodeau and Pablo Cárdenas Ramírez and Miles Cranmer and Urbano L. França and ChangHoon Hahn and Yan-Fei Jiang and Raul Jimenez and Jun-Young Lee and Antonio Lerario and Osman Mamun and Thomas Meier and Anupam A. Ojha and Pavlos Protopapas and Shimanto Roy and David N. Spergel and Pedro Tarancón-Álvarez and Ujjwal Tiwari and Matteo Viel and Digvijay Wadekar and Chi Wang and Bonny Y. Wang and Licong Xu and Yossi Yovel and Shuwen Yue and Wen-Han Zhou and Qiyao Zhu and Jiajun Zou and Íñigo Zubeldia},
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
          url = {https://github.com/AstroPilot-AI/Empirica},
          note = {Available at https://github.com/AstroPilot-AI/Empirica},
          version = {latest}
          }

@software{CMBAGENT_2025,
          author = {Boris Bolliet},
          title = {CMBAGENT: Open-Source Multi-Agent System for Science},
          year = {2025},
          url = {https://github.com/CMBAgents/cmbagent},
          note = {Available at https://github.com/CMBAgents/cmbagent},
          version = {latest}
          }
```

## License

[GNU GENERAL PUBLIC LICENSE (GPLv3)](https://www.gnu.org/licenses/gpl-3.0.html)

Empirica - Copyright (C) 2025 Pablo Villanueva-Domingo, Francisco Villaescusa-Navarro, Boris Bolliet

## Contact and enquieries

E-mail: [empirica.astropilot.ai@gmail.com](mailto:empirica.astropilot.ai@gmail.com)
