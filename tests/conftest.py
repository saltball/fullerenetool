"""
    Dummy conftest.py for fullerenetool.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    - https://docs.pytest.org/en/stable/fixture.html
    - https://docs.pytest.org/en/stable/writing_plugins.html
"""

import logging

# import pytest
import os
from pathlib import Path

from fullerenetool.logger import _LoggerLevelController

_LoggerLevelController._global_level = logging.DEBUG
os.chdir(Path(__file__).parent)
