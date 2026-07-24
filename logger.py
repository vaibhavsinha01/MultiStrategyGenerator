import logging
import os
from colorama import Fore, Style, init

# Initialize colorama (Windows support)
init(autoreset=True)


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA + Style.BRIGHT,
    }

    def format(self, record):
        color = self.COLORS.get(record.levelno, Fore.WHITE)

        message = super().format(record)

        return f"{color}{message}{Style.RESET_ALL}"


def get_logger(name: str = "MultiStrategyGenerator"):
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    os.makedirs("logs", exist_ok=True)

    # Common format
    log_format = "%(asctime)s | %(levelname)-8s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Console Handler (Colored)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        ColorFormatter(log_format, datefmt=date_format)
    )

    # File Handler (No Colors)
    file_handler = logging.FileHandler(
        "logs/application.log",
        encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(log_format, datefmt=date_format)
    )

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.propagate = False

    return logger

if __name__ == "__main__":
    logger = get_logger()
    logger.info("Application started.")
    logger.warning("API quota is getting low.")
    logger.error("Gemini API failed.")
    logger.critical("Database connection lost.")