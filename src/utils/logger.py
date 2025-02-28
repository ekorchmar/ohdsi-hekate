import logging
import sys

FORMATTER = logging.Formatter(
    "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] "
    "%(message)s"
)
_stdout_handler = logging.StreamHandler(sys.stdout)
_stdout_handler.setFormatter(FORMATTER)
logging.basicConfig(
    level=logging.INFO,
    handlers=[_stdout_handler],
)
LOGGER = logging.getLogger("Hekate")
LOGGER.info("Logging started")
