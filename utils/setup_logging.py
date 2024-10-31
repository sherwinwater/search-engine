import logging
from typing import Optional

def setup_logging(name: Optional[str] = None, task_id: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration that uses the caller's module name and supports task_id tracking.

    Args:
        name (str, optional): Logger name. If None, caller should pass __name__
        task_id (str, optional): Task ID to be included in log messages
    Returns:
        logging.Logger: Configured logger instance
    """
    # Use the passed name or default to root logger
    logger = logging.getLogger(f"{name} - {task_id}" if task_id else name)
    logger.setLevel(logging.INFO)

    # Create a filter to inject task_id into LogRecord
    class TaskFilter(logging.Filter):
        def filter(self, record):
            record.task_id = task_id or 'NO_TASK'
            return True

    # Create and configure handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # Add filter to handler
    task_filter = TaskFilter()
    console_handler.addFilter(task_filter)

    # Only add handler if it hasn't been added before
    if not logger.handlers:
        logger.addHandler(console_handler)

    return logger