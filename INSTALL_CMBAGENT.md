# Installing cmbagent

`cmbagent` is an optional dependency for Empirica that provides advanced planning and control capabilities for research analysis. The codebase has been updated to work without `cmbagent` installed, but some features require it.

## Installation Issues

### Windows with Python 3.13

If you're using Python 3.13 on Windows, `cmbagent` installation may fail due to `healpy` (a dependency) not having pre-built wheels for Python 3.13 on Windows. The error will look like:

```
ERROR: Failed building wheel for healpy
error: [WinError 193] %1 is not a valid Win32 application
```

### Solutions

1. **Use Python 3.12 (Recommended)**
   - Empirica officially supports Python 3.12 (see `pyproject.toml`: `requires-python = ">=3.12,<3.14"`)
   - `cmbagent` and its dependencies have better support for Python 3.12
   - Create a new virtual environment with Python 3.12:
     ```bash
     python3.12 -m venv Empirica_env
     source Empirica_env/bin/activate  # On Windows: Empirica_env\Scripts\activate
     pip install empirica
     ```

2. **Install cmbagent separately**
   - If you have Python 3.12 available, you can install `cmbagent` in that environment:
     ```bash
     pip install cmbagent>=0.0.1post63
     ```

3. **Use without cmbagent**
   - The codebase now works without `cmbagent` installed
   - Features that require `cmbagent` will show helpful error messages when used
   - You can use the "fast" mode for idea and method generation, which uses LangGraph instead of cmbagent

## Features Requiring cmbagent

The following features require `cmbagent` to be installed:

- `get_idea(mode="cmbagent")` - Detailed idea generation with cmbagent backend
- `get_method(mode="cmbagent")` - Detailed method generation with cmbagent backend  
- `get_results()` - Research analysis and execution (always uses cmbagent)
- `enhance_data_description()` - Data description enhancement
- `get_keywords()` - Keyword extraction using cmbagent
- `get_paper(cmbagent_keywords=True)` - Paper generation with cmbagent keywords

## Verifying Installation

After installation, verify `cmbagent` is available:

```python
try:
    import cmbagent
    print("cmbagent is installed successfully")
except ImportError:
    print("cmbagent is not installed")
```

## Network/SSL Issues

If you encounter SSL errors during installation:

1. Try upgrading pip:
   ```bash
   python -m pip install --upgrade pip
   ```

2. Use trusted hosts:
   ```bash
   pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org cmbagent>=0.0.1post63
   ```

3. Use no cache:
   ```bash
   pip install --no-cache-dir cmbagent>=0.0.1post63
   ```

## Alternative: Install from Source

If pip installation fails, you can try installing from the GitHub repository:

```bash
pip install git+https://github.com/CMBAgents/cmbagent.git
```

Note: This may still fail if dependencies like `healpy` cannot be built.

