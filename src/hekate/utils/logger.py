import logging
import sys

FORMATTER = logging.Formatter(
    "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s"
)
STDOUT_HANDLER = logging.StreamHandler(sys.stdout)
STDOUT_HANDLER.setFormatter(FORMATTER)
STDOUT_HANDLER.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[STDOUT_HANDLER],
)
LOGGER = logging.getLogger("Hekate")
