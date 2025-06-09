"""
Logging utility module for Empirica.

Provides structured logging with proper configuration, log levels, and file logging
capability while maintaining backward compatibility for interactive use.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


class EmpiricaLogger:
    """
    Centralized logging configuration for Empirica.
    
    Provides structured logging with file and console handlers.
    Maintains backward compatibility for interactive use.
    """
    
    _logger: Optional[logging.Logger] = None
    _configured: bool = False
    
    @classmethod
    def get_logger(cls, name: str = "empirica") -> logging.Logger:
        """
        Get or create the Empirica logger instance.
        
        Args:
            name: Logger name, defaults to "empirica"
            
        Returns:
            Configured logger instance
        """
        if cls._logger is None:
            cls._logger = logging.getLogger(name)
            if not cls._configured:
                cls._configure()
        return cls._logger
    
    @classmethod
    def _configure(cls, log_level: int = logging.INFO, log_file: Optional[Path] = None) -> None:
        """
        Configure the logger with console and optional file handlers.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            log_file: Optional path to log file. If None, no file logging.
        """
        if cls._configured:
            return
            
        logger = cls._logger
        logger.setLevel(log_level)
        
        # Remove existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # Console handler with formatted output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
        
        # File handler if log file specified
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)  # File gets all levels
            file_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)
        
        cls._configured = True
    
    @classmethod
    def configure(cls, log_level: str = "INFO", log_file: Optional[str] = None) -> None:
        """
        Configure logging from string parameters.
        
        Args:
            log_level: Log level as string ("DEBUG", "INFO", "WARNING", "ERROR")
            log_file: Optional path to log file as string
        """
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
        }
        level = level_map.get(log_level.upper(), logging.INFO)
        log_path = Path(log_file) if log_file else None
        cls._configure(level, log_path)


def get_logger(name: str = "empirica") -> logging.Logger:
    """
    Convenience function to get the Empirica logger.
    
    Args:
        name: Logger name, defaults to "empirica"
        
    Returns:
        Configured logger instance
    """
    return EmpiricaLogger.get_logger(name)


# Initialize default logger on import
_default_logger = get_logger()


