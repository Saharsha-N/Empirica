# Lightweight imports that don't require heavy dependencies
from .config import REPO_DIR
from .logger import get_logger, EmpiricaLogger
from .exceptions import (
    EmpiricaError,
    TaskResultError,
    ContentExtractionError,
    AgentExecutionError,
    DataValidationError,
    ConfigurationError
)

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("empirica")
except PackageNotFoundError:
    # fallback for editable installs, local runs, etc.
    __version__ = "0.0.0"

# Lazy imports for heavy dependencies - only import when actually accessed
# This allows CLI and other utilities to work without requiring all dependencies
def __getattr__(name):
    """Lazy import for heavy dependencies."""
    if name in ('Empirica', 'Research', 'Journal', 'LLM', 'models', 'KeyManager'):
        # Import the main module only when these are accessed
        from .empirica import Empirica, Research, Journal, LLM, models, KeyManager
        globals().update({
            'Empirica': Empirica,
            'Research': Research,
            'Journal': Journal,
            'LLM': LLM,
            'models': models,
            'KeyManager': KeyManager,
        })
        return globals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'Empirica', 'Research', 'Journal', 'REPO_DIR', 'LLM', "models", "KeyManager",
    "get_logger", "EmpiricaLogger",
    "EmpiricaError", "TaskResultError", "ContentExtractionError",
    "AgentExecutionError", "DataValidationError", "ConfigurationError"
]
