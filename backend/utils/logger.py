"""
Structured logging utility for the AI Travel Planner.
Provides consistent logging across all modules.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import yaml
from colorama import Fore, Back, Style, init

# Initialize colorama
init(autoreset=True)


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Back.WHITE + Style.BRIGHT,
    }
    
    def format(self, record):
        """Format log record with colors."""
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{Style.RESET_ALL}"
        return super().format(record)


class TravelPlannerLogger:
    """Centralized logger for the travel planner system."""
    
    _instance = None
    _loggers = {}
    
    def __new__(cls):
        """Singleton pattern to ensure one logger instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the logger system."""
        if self._initialized:
            return
        
        # Load configuration
        config_path = "backend/utils/config.yaml"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.log_config = config.get('logging', {})
        else:
            self.log_config = {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': 'logs/travel_planner.log'
            }
        
        # Create logs directory
        log_dir = Path(self.log_config['file']).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        self._initialized = True
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get or create a logger for a specific module.
        
        Args:
            name: Name of the module/logger
            
        Returns:
            Configured logger instance
        """
        if name in self._loggers:
            return self._loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, self.log_config['level']))
        
        # Remove existing handlers
        logger.handlers = []
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(self.log_config['file'])
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            self.log_config['format'],
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        self._loggers[name] = logger
        return logger
    
    def log_agent_start(self, logger: logging.Logger, agent_name: str, task: str):
        """Log the start of an agent task."""
        logger.info(f"🚀 {agent_name} started: {task}")
    
    def log_agent_complete(self, logger: logging.Logger, agent_name: str, duration: float):
        """Log the completion of an agent task."""
        logger.info(f"✅ {agent_name} completed in {duration:.2f}s")
    
    def log_agent_error(self, logger: logging.Logger, agent_name: str, error: str):
        """Log an agent error."""
        logger.error(f"❌ {agent_name} failed: {error}")
    
    def log_api_call(self, logger: logging.Logger, api_name: str, endpoint: str, status: str):
        """Log an API call."""
        logger.debug(f"🌐 API Call - {api_name} [{endpoint}]: {status}")
    
    def log_validation(self, logger: logging.Logger, validation_type: str, result: bool, message: str = ""):
        """Log a validation result."""
        emoji = "✓" if result else "✗"
        level = logging.INFO if result else logging.WARNING
        logger.log(level, f"{emoji} Validation [{validation_type}]: {message}")
    
    def log_pipeline_stage(self, logger: logging.Logger, stage: str, status: str):
        """Log a pipeline stage."""
        logger.info(f"📍 Pipeline Stage [{stage}]: {status}")


# Global logger instance
_logger_instance = TravelPlannerLogger()


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the specified module.
    
    Args:
        name: Name of the module
        
    Returns:
        Configured logger instance
    """
    return _logger_instance.get_logger(name)


def log_agent_activity(logger: logging.Logger, agent: str, action: str, status: str = "started"):
    """
    Log agent activity with consistent formatting.
    
    Args:
        logger: Logger instance
        agent: Agent name
        action: Action being performed
        status: Status of the action
    """
    status_emoji = {
        'started': '🚀',
        'completed': '✅',
        'failed': '❌',
        'processing': '⚙️',
        'waiting': '⏳'
    }
    emoji = status_emoji.get(status, '📌')
    logger.info(f"{emoji} [{agent}] {action} - {status}")


# Convenience functions
def log_info(logger: logging.Logger, message: str):
    """Log info message."""
    logger.info(message)


def log_error(logger: logging.Logger, message: str, exception: Optional[Exception] = None):
    """Log error message."""
    if exception:
        logger.error(f"{message}: {str(exception)}", exc_info=True)
    else:
        logger.error(message)


def log_debug(logger: logging.Logger, message: str):
    """Log debug message."""
    logger.debug(message)


def log_warning(logger: logging.Logger, message: str):
    """Log warning message."""
    logger.warning(message)
