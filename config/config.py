"""
Configuration settings for Camera RTSP Application
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration"""

    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    LOGS_DIR = DATA_DIR / "logs"
    EMBEDDINGS_DIR = DATA_DIR / "embeddings"

    # Backend API
    BACKEND_URL = os.getenv('SPRING_BOOT_API_URL', 'http://localhost:8080/api')
    WEBSOCKET_URL = os.getenv('WEBSOCKET_URL', 'ws://localhost:8080/ws')

    # Recognition Settings
    RECOGNITION_THRESHOLD = float(os.getenv('RECOGNITION_THRESHOLD', '0.6'))
    DETECTION_THRESHOLD = float(os.getenv('DETECTION_THRESHOLD', '0.5'))
    FRAME_PROCESSING_INTERVAL = int(os.getenv('FRAME_PROCESSING_INTERVAL', '3'))
    MAX_CONCURRENT_CAMERAS = int(os.getenv('MAX_CONCURRENT_CAMERAS', '4'))

    # Cache Settings
    EMBEDDING_CACHE_DIR = os.getenv('EMBEDDING_CACHE_DIR', str(EMBEDDINGS_DIR))
    FACE_DB_PATH = os.getenv('FACE_DB_PATH', str(DATA_DIR / 'faces_db.json'))
    CACHE_EXPIRE_HOURS = int(os.getenv('CACHE_EXPIRE_HOURS', '24'))

    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_DIR = os.getenv('LOG_DIR', str(LOGS_DIR))

    # RTSP Settings
    RTSP_TIMEOUT = int(os.getenv('RTSP_TIMEOUT', '30'))
    HLS_SEGMENT_DURATION = int(os.getenv('HLS_SEGMENT_DURATION', '2'))

    # UI Settings
    WINDOW_WIDTH = int(os.getenv('WINDOW_WIDTH', '1200'))
    WINDOW_HEIGHT = int(os.getenv('WINDOW_HEIGHT', '800'))

    # InsightFace Settings
    INSIGHTFACE_MODEL = os.getenv('INSIGHTFACE_MODEL', 'buffalo_l')
    INSIGHTFACE_PROVIDERS = os.getenv('INSIGHTFACE_PROVIDERS', 'CPUExecutionProvider').split(',')

    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.LOGS_DIR.mkdir(exist_ok=True)
        cls.EMBEDDINGS_DIR.mkdir(exist_ok=True)

    @classmethod
    def get_config_dict(cls):
        """Get configuration as dictionary"""
        return {
            'backend_url': cls.BACKEND_URL,
            'recognition_threshold': cls.RECOGNITION_THRESHOLD,
            'detection_threshold': cls.DETECTION_THRESHOLD,
            'frame_processing_interval': cls.FRAME_PROCESSING_INTERVAL,
            'max_concurrent_cameras': cls.MAX_CONCURRENT_CAMERAS,
            'rtsp_timeout': cls.RTSP_TIMEOUT,
            'window_width': cls.WINDOW_WIDTH,
            'window_height': cls.WINDOW_HEIGHT,
            'log_level': cls.LOG_LEVEL
        }


# Global config instance
config = Config()