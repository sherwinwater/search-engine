import logging


def setup_logging(name=None):
    """
    Setup logging configuration that uses the caller's module name.

    Args:
        name (str, optional): Logger name. If None, caller should pass __name__
    Returns:
        logging.Logger: Configured logger instance
    """
    # Use the passed name or default to root logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(console_handler)

    return logger