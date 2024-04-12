import logging
from typing import Optional


def setup_logger(
    identifier: str,
    console: bool = True,
    file: Optional[str] = None,
    default_level=logging.DEBUG,
) -> logging.Logger:
    """Sets up a logger with the given identifier.

    Args:
        identifier (str): The name of the logger.
        console (bool, optional): Whether to log to the console. Defaults to True.
        file (Optional[str], optional): The file to log to. Defaults to None.
        default_level ([type], optional): The default logging level.
            Defaults to logging.DEBUG.

    Returns:
        logging.Logger: The logger object.
    """
    # create logger named `identifier`
    logger = logging.getLogger(identifier)
    logger.setLevel(default_level)

    if file:
        file_handler = logging.FileHandler(file)
        file_handler.setLevel(default_level)
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(default_level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if file:
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    if console:
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


logger = setup_logger(
    "fullerenetool", console=True, file=None, default_level=logging.INFO
)
