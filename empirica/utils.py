import os
import re
from pathlib import Path
import warnings

from .llm import LLM, models
from .exceptions import DataValidationError

def validate_path_safety(file_path: str | Path, base_path: str | Path | None = None) -> Path:
    """
    Validate that a file path is safe and doesn't contain path traversal attempts.
    
    This function checks for path traversal patterns (e.g., ../, ..\\) and ensures
    that if a base_path is provided, the resolved path stays within that base.
    
    Args:
        file_path: The file path to validate.
        base_path: Optional base path to restrict the file path within. If provided,
                   the resolved path must be within this base path.
    
    Returns:
        A Path object representing the resolved, validated path.
    
    Raises:
        DataValidationError: If the path contains traversal patterns or escapes
                            the base_path when one is provided.
    """
    # Check for path traversal patterns in the original string before resolving
    path_str = str(file_path)
    
    # Check for path traversal patterns (.. or ..\ or ../)
    if '..' in path_str:
        # Normalize and check if any component is '..'
        normalized = os.path.normpath(path_str)
        path_parts = normalized.split(os.sep)
        if os.altsep:
            # Also check with alternate separator on Windows
            alt_parts = normalized.split(os.altsep)
            if '..' in alt_parts:
                raise DataValidationError(
                    f"Path traversal detected in path: {file_path}. "
                    "Paths containing '..' are not allowed for security reasons."
                )
        if '..' in path_parts:
            raise DataValidationError(
                f"Path traversal detected in path: {file_path}. "
                "Paths containing '..' are not allowed for security reasons."
            )
    
    # Resolve the path
    path = Path(file_path).resolve()
    
    # If base_path is provided, ensure the resolved path is within it
    if base_path is not None:
        base = Path(base_path).resolve()
        try:
            # Check if path is within base_path
            path.relative_to(base)
        except ValueError:
            raise DataValidationError(
                f"Path {file_path} resolves outside of base path {base_path}. "
                "This may indicate a path traversal attempt."
            )
    
    return path

def validate_file_path(file_path: str | Path, must_exist: bool = False, 
                       allowed_extensions: list[str] | None = None) -> Path:
    """
    Validate a file path for security and correctness.
    
    Args:
        file_path: The file path to validate.
        must_exist: If True, the file must exist. Defaults to False.
        allowed_extensions: Optional list of allowed file extensions (e.g., ['.md', '.txt']).
                           If provided, the file must have one of these extensions.
    
    Returns:
        A Path object representing the validated path.
    
    Raises:
        DataValidationError: If validation fails.
    """
    path = validate_path_safety(file_path)
    
    # Check file extension if specified
    if allowed_extensions is not None:
        if path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
            raise DataValidationError(
                f"File {file_path} has extension {path.suffix}, "
                f"which is not in allowed extensions: {allowed_extensions}"
            )
    
    # Check existence if required
    if must_exist:
        if not path.exists():
            raise DataValidationError(f"File does not exist: {file_path}")
        if not path.is_file():
            raise DataValidationError(f"Path is not a file: {file_path}")
    
    return path

def input_check(str_input: str) -> str:
    """
    Validate and process input string or markdown file path.
    
    Checks if the input is a string containing content directly, or a path to a markdown file.
    If it's a markdown file path (ending with .md), reads and returns the file content.
    Otherwise, returns the string as-is.
    
    Args:
        str_input: Input string containing content or a path to a markdown file.
    
    Returns:
        The content string, either from the input directly or read from the markdown file.
    
    Raises:
        ValueError: If the input is neither a string nor a valid markdown file path.
    """

    if str_input.endswith(".md"):
        # Validate the file path for security
        try:
            validated_path = validate_file_path(str_input, must_exist=True, allowed_extensions=['.md'])
        except DataValidationError as e:
            raise ValueError(f"Invalid file path: {e}") from e
        
        try:
            with open(validated_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except OSError as e:
            raise ValueError(f"Failed to read file {str_input}: {e}") from e
    elif isinstance(str_input, str):
        content = str_input
    else:
        raise ValueError("Input must be a string or a path to a markdown file.")
    return content

def llm_parser(llm: LLM | str) -> LLM:
    """
    Convert an LLM string identifier to an LLM instance.
    
    If the input is already an LLM instance, returns it unchanged.
    If it's a string, looks up the corresponding LLM instance from the models registry.
    
    Args:
        llm: Either an LLM instance or a string identifier for an LLM model.
    
    Returns:
        The LLM instance corresponding to the input.
    
    Raises:
        KeyError: If the string identifier is not found in the available models.
    """

    if isinstance(llm, str):
        try:
            llm = models[llm]
        except KeyError:
            raise KeyError(f"LLM '{llm}' not available. Please select from: {list(models.keys())}")
    return llm

def extract_file_paths(markdown_text: str) -> tuple[list[str], list[str]]:
    """
    Extract the bulleted file paths from markdown text 
    and check if they exist and are absolute paths.
    
    Args:
        markdown_text (str): The markdown text containing file paths
    
    Returns:
        tuple: (existing_paths, missing_paths)
    """
    
    # Pattern to match file paths in markdown bullet points
    pattern = r'-\s*([^\n]+\.(?:csv|txt|md|py|json|yaml|yml|xml|html|css|js|ts|tsx|jsx|java|cpp|c|h|hpp|go|rs|php|rb|pl|sh|bat|sql|log))'
    
    # Find all matches
    matches = re.findall(pattern, markdown_text, re.IGNORECASE)
    
    # Clean up paths and check existence
    existing_paths = []
    missing_paths = []
    invalid_paths = []
    
    for match in matches:
        path_str = match.strip()
        
        # Validate path safety
        try:
            validated_path = validate_path_safety(path_str)
        except DataValidationError:
            invalid_paths.append(path_str)
            continue
        
        # Check if path exists and is absolute
        if validated_path.exists() and validated_path.is_absolute():
            if validated_path.is_file():
                existing_paths.append(str(validated_path))
            else:
                missing_paths.append(path_str)  # Path exists but is not a file
        else:
            missing_paths.append(path_str)
    
    # Log warnings for invalid paths (path traversal attempts)
    if invalid_paths:
        warnings.warn(
            f"The following file paths contain path traversal patterns and were rejected:\n"
            f"{invalid_paths}\n"
            f"Please use absolute paths without '..' components for security reasons."
        )
    
    return existing_paths, missing_paths

def check_file_paths(content: str) -> None:
    """
    Validate file paths mentioned in markdown content.
    
    Extracts file paths from markdown bullet points and checks if they exist
    and are in absolute path format. Issues warnings for missing or invalid paths.
    
    Args:
        content: Markdown text content that may contain file path references.
    
    Note:
        Issues warnings (not exceptions) for:
        - File paths that don't exist or aren't absolute paths
        - Cases where no valid file paths are found in the content
    """

    existing_paths, missing_paths = extract_file_paths(content)

    if len(missing_paths) > 0:
        warnings.warn(
            f"The following data files paths in the data description are not in the right format or do not exist:\n"
            f"{missing_paths}\n"
            f"Please fix them according to the convention '- /absolute/path/to/file.ext'\n"
            f"otherwise this may cause hallucinations in the LLMs."
        )

    if len(existing_paths) == 0:
        warnings.warn(
            "No data files paths were found in the data description. If you want to provide input data, ensure that you indicate their path, otherwise this may cause hallucinations in the LLM in the get_results() workflow later on."
        )

def create_work_dir(work_dir: str | Path, name: str) -> Path:
    """
    Create a working directory for a specific task.
    
    Creates a subdirectory within the specified work directory with the format
    "{name}_generation_output". The directory is created if it doesn't exist.
    
    Args:
        work_dir: Base working directory path.
        name: Name identifier for the task (e.g., "idea", "method", "experiment").
              Must not contain path separators or traversal patterns.
    
    Returns:
        Path object pointing to the created working directory.
    
    Raises:
        DataValidationError: If the name contains invalid characters or path traversal patterns.
    """
    # Validate the name parameter to prevent path traversal
    if not name or not isinstance(name, str):
        raise DataValidationError(f"Invalid directory name: {name}. Must be a non-empty string.")
    
    # Check for path separators and traversal patterns in name
    if os.sep in name or (os.altsep and os.altsep in name) or '..' in name:
        raise DataValidationError(
            f"Directory name '{name}' contains invalid characters. "
            "Names must not contain path separators or '..' for security reasons."
        )
    
    # Validate base work directory
    try:
        base_path = validate_path_safety(work_dir)
        if not base_path.exists():
            # Create base directory if it doesn't exist
            base_path.mkdir(parents=True, exist_ok=True)
        if not base_path.is_dir():
            raise DataValidationError(f"Work directory path is not a directory: {work_dir}")
    except DataValidationError:
        raise
    except Exception as e:
        raise DataValidationError(f"Invalid work directory path: {work_dir}") from e
    
    # Construct the full path
    work_dir_path = base_path / f"{name}_generation_output"
    
    # Ensure the final path is within the base directory
    try:
        work_dir_path.resolve().relative_to(base_path.resolve())
    except ValueError:
        raise DataValidationError(
            f"Generated work directory path escapes base directory. "
            f"This should not happen with valid input."
        )
    
    # Create the directory
    try:
        work_dir_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise DataValidationError(f"Failed to create work directory: {e}") from e
    
    return work_dir_path

def get_task_result(chat_history: list[dict], name: str) -> str:
    """
    Get task result from chat history.
    
    Args:
        chat_history: List of chat history objects
        name: Name of the task to extract
        
    Returns:
        The content of the task result
        
    Raises:
        TaskResultError: If the task name is not found in chat history
        KeyError: If chat history objects don't have expected structure
    """
    from .exceptions import TaskResultError
    
    if not chat_history:
        raise TaskResultError(
            task_name=name,
            chat_history_length=0,
            message=f"Cannot extract task result for '{name}': chat history is empty."
        )
    
    result = None
    for obj in chat_history[::-1]:
        if isinstance(obj, dict) and obj.get('name') == name:
            result = obj.get('content')
            break
    
    if result is None:
        available_names = [obj.get('name') for obj in chat_history if isinstance(obj, dict) and 'name' in obj]
        raise TaskResultError(
            task_name=name,
            chat_history_length=len(chat_history),
            message=(
                f"Task '{name}' not found in chat history. "
                f"Available task names: {available_names[:10]}"  # Show first 10 to avoid huge messages
            )
        )
    
    return result

def in_notebook() -> bool:
    """Check whether the code is run from a Jupyter Notebook or not, to use different display options"""
    
    try:
        from IPython import get_ipython # type: ignore
        if 'IPKernelApp' not in get_ipython().config:  # type: ignore # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True
