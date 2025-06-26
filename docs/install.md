# Installation

## Requirements

For running Empirica you will need different API keys for the LLMs. The supported ones are OpenAI, Gemini, Anthropic and Perplexity. [Read here](apikeys.md) more details on the LLMs required for each modules and how to get the different API keys.

Also, you will need [LaTeX](https://www.latex-project.org/) installed for the write paper module. This is not required for the other modules.

## Virtual environment

Given that Empirica could install packages and run code, we recommend to run it within a virtual environment. We recommend using Python 3.12. You can create a virtual environment through different otions.

### venv

Using `venv`:

```bash
python3 -m venv Empirica_env
```

Activate the virtual environment with

```bash
source Empirica_env/bin/activate
```

### conda

You can also use [conda](https://docs.conda.io/projects/conda/en/stable/index.html) instead:

```bash
conda create -n Empirica_env python==3.12
conda activate Empirica_env
```

### uv

Or also [uv](https://docs.astral.sh/uv/):

```bash
uv init --python 3.12
source .venv/bin/activate
```

## Install from PyPI

### pip

To install Empirica, just run

```bash
pip install "empirica[app]"
```

The `[app]` allow us to run the [GUI](docs/app.md). If we do not need that, we can also install just `pip install empirica`.

### uv

Alternatively, we can use [uv](https://docs.astral.sh/uv/) to install empirica as

```bash
uv add empirica[app]
```

or `uv add empirica` if we do not need GUI support.

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

which will create the virtual environment and install the dependencies and project. Activate the virtual environment, if needed, with

```bash
source .venv/bin/activate
```
