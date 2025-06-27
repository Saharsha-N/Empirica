# EmpiricaApp

EmpiricaApp is the GUI for multiagent research assitant system [Empirica](https://github.com/AstroPilot-AI/Empirica.git), powered by [streamlit](https://streamlit.io).

[Test a deployed demo of this app in HugginFace Spaces.](https://huggingface.co/spaces/astropilot-ai/Empirica)

<img width="1793" height="694" alt="Screenshot from 2025-09-10 18-30-46" src="https://github.com/user-attachments/assets/2c524601-13ff-492b-addb-173323aaa15b" />

## Launch the GUI

Install the app with

```bash
pip install "empirica[app]"
```

or, if Empirica is already installed, do:

```bash
pip install empirica_app
```

Then, launch the app with

```bash
empirica run
```

## Build the GUI from source

First, clone the app with

`git clone https://github.com/AstroPilot-AI/EmpiricaApp.git`

Install the GUI from source following one of the following steps:

1. Install with pip

   ```bash
   pip install -e .
   ```

2. Install with [uv](https://docs.astral.sh/uv/)

   ```bash
   uv sync
   ```

Run the app with:

```bash
empirica run
```

or

```bash
streamlit run src/empirica_app/app.py
```
