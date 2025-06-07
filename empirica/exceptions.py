"""
Custom exception classes for Empirica.

Provides specific exception types for better error categorization and handling.
"""


class EmpiricaError(Exception):
    """Base exception class for all Empirica errors."""
    pass


class TaskResultError(EmpiricaError):
    """Raised when a task result cannot be extracted from chat history."""
    
    def __init__(self, task_name: str, chat_history_length: int = 0, message: str | None = None):
        self.task_name = task_name
        self.chat_history_length = chat_history_length
        if message is None:
            message = (
                f"Failed to extract task result for '{task_name}' from chat history. "
                f"Chat history contains {chat_history_length} entries. "
                f"The expected task name '{task_name}' was not found in the chat history."
            )
        super().__init__(message)


class ContentExtractionError(EmpiricaError):
    """Raised when content cannot be extracted from a formatted response."""
    
    def __init__(self, content_type: str, source: str | None = None, message: str | None = None):
        self.content_type = content_type
        self.source = source
        if message is None:
            source_info = f" from {source}" if source else ""
            message = f"Failed to extract {content_type}{source_info}. The expected format was not found in the response."
        super().__init__(message)


class AgentExecutionError(EmpiricaError):
    """Raised when an agent execution fails."""
    
    def __init__(self, agent_name: str, stage: str | None = None, message: str | None = None):
        self.agent_name = agent_name
        self.stage = stage
        if message is None:
            stage_info = f" at stage '{stage}'" if stage else ""
            message = f"Agent '{agent_name}' execution failed{stage_info}."
        super().__init__(message)


class DataValidationError(EmpiricaError):
    """Raised when data validation fails."""
    pass


class ConfigurationError(EmpiricaError):
    """Raised when there is a configuration error."""
    pass


