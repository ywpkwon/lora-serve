import logging
import sys

def configure_logging(level: str = "INFO"):
    """Configure root logger for LoRAServe."""
    root = logging.getLogger()
    root.setLevel(level.upper())

    handler = logging.StreamHandler(sys.stdout)
    fmt = "%(asctime)s [%(levelname)s] %(name)-40s: %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    root.handlers.clear()
    root.addHandler(handler)

# A convenience logger for imports
logger = logging.getLogger("lora_serve")
