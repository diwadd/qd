import logging

# LOGGING_LEVEL = logging.CRITICAL
# LOGGING_LEVEL = logging.ERROR
# LOGGING_LEVEL = logging.WARNING
LOGGING_LEVEL = logging.INFO
# LOGGING_LEVEL = logging.DEBUG
# LOGGING_LEVEL = logging.NOTSET

logger = logging.getLogger(__name__)

file_handler = logging.FileHandler("logs.log")
file_handler.setLevel(LOGGING_LEVEL)

handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")

handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(handler)
logger.addHandler(file_handler)
logger.setLevel(LOGGING_LEVEL)