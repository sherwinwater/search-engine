# socket_logger.py
import logging
from datetime import datetime
from threading import Lock
from typing import Optional


class SocketIOLogHandler(logging.Handler):
    _instance = None
    _socketio = None

    def __init__(self):
        super().__init__()
        self.log_lock = Lock()

    @classmethod
    def init_handler(cls, socketio):
        """Initialize the handler with a SocketIO instance"""
        cls._socketio = socketio
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def get_handler(cls):
        """Get the initialized handler instance"""
        if cls._instance is None:
            raise RuntimeError("SocketIOLogHandler not initialized. Call init_handler first.")
        return cls._instance

    def emit(self, record):
        if self._socketio is None:
            return

        try:
            with self.log_lock:
                log_entry = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'level': record.levelname,
                    'message': record.getMessage(),
                    'task_id': getattr(record, 'task_id', 'NO_TASK')
                }
                self._socketio.emit('log_message', log_entry)
        except Exception as e:
            print(f"Error in log handler: {e}")


def setup_logging(name: Optional[str] = None, task_id: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration with both console and SocketIO handlers.

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

    # Only add handlers if they haven't been added before
    if not logger.handlers:
        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        # Add filter to handlers
        task_filter = TaskFilter()
        console_handler.addFilter(task_filter)
        logger.addHandler(console_handler)

        # SocketIO Handler (if initialized)
        try:
            socket_handler = SocketIOLogHandler.get_handler()
            socket_handler.setLevel(logging.INFO)
            socket_handler.setFormatter(formatter)
            socket_handler.addFilter(task_filter)
            logger.addHandler(socket_handler)
        except RuntimeError:
            # SocketIO handler not initialized yet, skip it
            pass

    return logger