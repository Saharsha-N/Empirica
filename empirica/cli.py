import sys
import argparse
from importlib.metadata import version, PackageNotFoundError
from .logger import get_logger

logger = get_logger(__name__)

def get_version():
    """Get the version number without triggering heavy imports."""
    try:
        return version("empirica")
    except PackageNotFoundError:
        return "0.0.0"

def main():
    """
    Main entry point for the Empirica CLI.
    
    Provides command-line interface for running Empirica applications and utilities.
    """
    parser = argparse.ArgumentParser(
        prog="empirica",
        description="Empirica: Modular Multi-Agent System for Scientific Research Assistance",
        epilog=(
            "For more information, visit: "
            "https://empirica.readthedocs.io/en/latest/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"empirica {get_version()}",
        help="Show the version number and exit"
    )
    
    subparsers = parser.add_subparsers(
        dest="command",
        metavar="COMMAND",
        help="Available commands"
    )

    # `empirica run`
    run_parser = subparsers.add_parser(
        "run",
        help="Run the Empirica Streamlit app",
        description="Launch the Empirica graphical user interface (GUI) using Streamlit."
    )
    run_parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port number for the Streamlit app (default: 8501)"
    )
    run_parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host address for the Streamlit app (default: localhost)"
    )

    args = parser.parse_args()

    if args.command == "run":
        try:
            from empirica_app.cli import run
            # Pass port and host if the run function supports them
            # Otherwise, just call run() and let it use defaults
            try:
                run(port=args.port, host=args.host)
            except TypeError:
                # If run() doesn't accept these arguments, call without them
                run()
        except ImportError as e:
            logger.error(
                "EmpiricaApp is not installed. "
                "Install it with: pip install empirica-app "
                "or: pip install 'empirica[app]'"
            )
            logger.debug(f"Import error details: {e}", exc_info=True)
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to start Empirica app: {e}", exc_info=True)
            sys.exit(1)
    elif args.command is None:
        # No command provided, show help
        parser.print_help()
        sys.exit(0)
    else:
        # Unknown command (shouldn't happen with argparse, but handle gracefully)
        logger.error(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)
