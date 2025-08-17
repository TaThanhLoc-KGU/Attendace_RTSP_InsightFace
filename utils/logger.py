"""
Enhanced logging utilities with Unicode support for Camera RTSP Application
"""
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from config.config import config


class UnicodeFormatter(logging.Formatter):
    """Unicode-safe formatter that handles emoji and special characters"""

    def format(self, record):
        # Get the original formatted message
        formatted = super().format(record)

        # Handle encoding issues
        try:
            # Try to encode to the console's encoding
            if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding:
                formatted.encode(sys.stdout.encoding)
            return formatted
        except UnicodeEncodeError:
            # If encoding fails, remove non-ASCII characters
            return formatted.encode('ascii', errors='ignore').decode('ascii')


class ColoredFormatter(UnicodeFormatter):
    """Custom formatter with colors for console output - Unicode safe"""

    COLORS = {
        'DEBUG': '\033[36m',  # Cyan
        'INFO': '\033[32m',  # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',  # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'  # Reset
    }

    def format(self, record):
        # Apply colors only if terminal supports it
        if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            if record.levelname in self.COLORS:
                record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"

        return super().format(record)


class UnicodeFileHandler(logging.handlers.RotatingFileHandler):
    """File handler that properly handles Unicode content"""

    def __init__(self, filename, mode='a', maxBytes=0, backupCount=0, encoding='utf-8', delay=False):
        # Force UTF-8 encoding for file output
        super().__init__(filename, mode, maxBytes, backupCount, encoding, delay)


def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    """Setup logger with both file and console handlers - Unicode safe"""

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.LOG_LEVEL.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatters
    file_formatter = UnicodeFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler with proper encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # Set encoding for console if possible
    if hasattr(console_handler.stream, 'reconfigure'):
        try:
            console_handler.stream.reconfigure(encoding='utf-8', errors='ignore')
        except:
            pass

    logger.addHandler(console_handler)

    # File handler with UTF-8 encoding
    if log_file:
        log_path = Path(config.LOG_DIR) / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = UnicodeFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get logger instance for module"""
    return setup_logger(name, f"{name}.log")


def safe_log(logger, level, message, *args, **kwargs):
    """Safe logging function that handles Unicode issues"""
    try:
        # Convert message to string and handle Unicode
        safe_message = str(message)

        # Try to log normally first
        getattr(logger, level)(safe_message, *args, **kwargs)

    except UnicodeEncodeError:
        # Fallback: remove non-ASCII characters
        ascii_message = safe_message.encode('ascii', errors='ignore').decode('ascii')
        getattr(logger, level)(f"[Unicode Error] {ascii_message}", *args, **kwargs)
    except Exception as e:
        # Last resort: basic logging
        print(f"Logging error: {e}")
        print(f"Original message: {message}")


# Enhanced logger instances with safe logging methods
class SafeLogger:
    """Wrapper for logger with safe Unicode handling"""

    def __init__(self, logger):
        self._logger = logger

    def debug(self, message, *args, **kwargs):
        safe_log(self._logger, 'debug', message, *args, **kwargs)

    def info(self, message, *args, **kwargs):
        safe_log(self._logger, 'info', message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        safe_log(self._logger, 'warning', message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        safe_log(self._logger, 'error', message, *args, **kwargs)

    def critical(self, message, *args, **kwargs):
        safe_log(self._logger, 'critical', message, *args, **kwargs)


# Create safe logger instances
_app_logger = get_logger("camera_app")
_camera_logger = get_logger("camera_service")
_face_logger = get_logger("face_service")
_api_logger = get_logger("backend_api")
_ui_logger = get_logger("ui")

# Export safe logger instances
app_logger = SafeLogger(_app_logger)
camera_logger = SafeLogger(_camera_logger)
face_logger = SafeLogger(_face_logger)
api_logger = SafeLogger(_api_logger)
ui_logger = SafeLogger(_ui_logger)


# Set UTF-8 encoding for stdout/stderr if on Windows
if sys.platform.startswith('win'):
    try:
        # Try to set UTF-8 encoding for Windows console
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    except:
        # If that fails, just continue
        pass