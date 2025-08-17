"""
Configuration settings for Camera RTSP Application - HIGH PERFORMANCE VERSION
Optimized for 15-30 FPS with webcam support
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, List

# Load environment variables
load_dotenv()


class Config:
    """Application configuration - OPTIMIZED FOR HIGH FPS"""
    # Authentication Settings
    BACKEND_USERNAME = os.getenv('BACKEND_USERNAME', 'admin')
    BACKEND_PASSWORD = os.getenv('BACKEND_PASSWORD', 'admin@123')
    AUTO_LOGIN = bool(os.getenv('AUTO_LOGIN', 'true').lower() == 'true')

    # Login retry settings
    LOGIN_RETRY_ATTEMPTS = int(os.getenv('LOGIN_RETRY_ATTEMPTS', '3'))
    LOGIN_RETRY_DELAY = int(os.getenv('LOGIN_RETRY_DELAY', '2'))  # seconds

    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    LOGS_DIR = DATA_DIR / "logs"
    EMBEDDINGS_DIR = DATA_DIR / "embeddings"

    # Backend API
    BACKEND_URL = os.getenv('SPRING_BOOT_API_URL', 'http://localhost:8080/api')
    WEBSOCKET_URL = os.getenv('WEBSOCKET_URL', 'ws://localhost:8080/ws')

    # Recognition Settings - OPTIMIZED FOR SPEED
    RECOGNITION_THRESHOLD = float(os.getenv('RECOGNITION_THRESHOLD', '0.40'))  # Lower for faster processing
    DETECTION_THRESHOLD = float(os.getenv('DETECTION_THRESHOLD', '0.35'))      # Lower threshold
    FRAME_PROCESSING_INTERVAL = int(os.getenv('FRAME_PROCESSING_INTERVAL', '3'))  # Skip 2 frames for speed
    MAX_CONCURRENT_CAMERAS = int(os.getenv('MAX_CONCURRENT_CAMERAS', '6'))

    # Cache Settings
    EMBEDDING_CACHE_DIR = os.getenv('EMBEDDING_CACHE_DIR', str(EMBEDDINGS_DIR))
    FACE_DB_PATH = os.getenv('FACE_DB_PATH', str(DATA_DIR / 'faces_db.json'))
    CACHE_EXPIRE_HOURS = int(os.getenv('CACHE_EXPIRE_HOURS', '24'))

    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_DIR = os.getenv('LOG_DIR', str(LOGS_DIR))

    # RTSP Settings - OPTIMIZED FOR LOW LATENCY
    RTSP_TIMEOUT = int(os.getenv('RTSP_TIMEOUT', '3000'))  # Shorter timeout
    HLS_SEGMENT_DURATION = int(os.getenv('HLS_SEGMENT_DURATION', '1'))

    # Camera Performance Settings - HIGH FPS OPTIMIZATION
    CAMERA_BUFFER_SIZE = int(os.getenv('CAMERA_BUFFER_SIZE', '1'))  # Minimal buffer for low latency
    CAMERA_FPS = int(os.getenv('CAMERA_FPS', '30'))

    # CRITICAL: Reduced processing resolution for speed
    PROCESSING_WIDTH = int(os.getenv('PROCESSING_WIDTH', '416'))    # Reduced from 640
    PROCESSING_HEIGHT = int(os.getenv('PROCESSING_HEIGHT', '312'))  # Reduced from 480

    # Display can remain higher for good viewing experience
    DISPLAY_WIDTH = int(os.getenv('DISPLAY_WIDTH', '640'))
    DISPLAY_HEIGHT = int(os.getenv('DISPLAY_HEIGHT', '480'))

    # UI Settings
    WINDOW_WIDTH = int(os.getenv('WINDOW_WIDTH', '1400'))
    WINDOW_HEIGHT = int(os.getenv('WINDOW_HEIGHT', '900'))

    # InsightFace Settings - OPTIMIZED FOR SPEED
    INSIGHTFACE_MODEL = os.getenv('INSIGHTFACE_MODEL', 'buffalo_l')  # Smaller, faster model

    # Performance Optimization - NEW SETTINGS
    MAX_FACES_PER_FRAME = int(os.getenv('MAX_FACES_PER_FRAME', '8'))  # Reduced from 15
    ENABLE_PARALLEL_PROCESSING = bool(os.getenv('ENABLE_PARALLEL_PROCESSING', 'True').lower() == 'true')
    MAX_WORKER_THREADS = int(os.getenv('MAX_WORKER_THREADS', '4'))

    # Quality vs Speed trade-off
    MIN_QUALITY_SCORE = float(os.getenv('MIN_QUALITY_SCORE', '0.15'))  # Lower quality threshold for speed
    MIN_BLUR_SCORE = float(os.getenv('MIN_BLUR_SCORE', '10.0'))       # Lower blur threshold
    ENABLE_QUALITY_FILTER = bool(os.getenv('ENABLE_QUALITY_FILTER', 'False').lower() == 'true')  # Disable for speed

    # Tracking Settings - OPTIMIZED
    TRACK_BUFFER_SIZE = int(os.getenv('TRACK_BUFFER_SIZE', '15'))     # Reduced buffer size
    MIN_TRACK_LENGTH = int(os.getenv('MIN_TRACK_LENGTH', '3'))        # Shorter minimum track
    TRACK_MATCH_THRESHOLD = float(os.getenv('TRACK_MATCH_THRESHOLD', '0.5'))
    STABLE_TRACK_THRESHOLD = float(os.getenv('STABLE_TRACK_THRESHOLD', '0.75'))  # Lower for faster recognition

    # Brightness/contrast filters - relaxed for speed
    MIN_BRIGHTNESS = float(os.getenv('MIN_BRIGHTNESS', '20'))
    MAX_BRIGHTNESS = float(os.getenv('MAX_BRIGHTNESS', '250'))

    # NEW: Webcam Support Settings
    ENABLE_WEBCAM_OPTION = bool(os.getenv('ENABLE_WEBCAM_OPTION', 'True').lower() == 'true')
    WEBCAM_PROCESSING_WIDTH = int(os.getenv('WEBCAM_PROCESSING_WIDTH', '320'))   # Even smaller for webcams
    WEBCAM_PROCESSING_HEIGHT = int(os.getenv('WEBCAM_PROCESSING_HEIGHT', '240'))
    WEBCAM_FPS_TARGET = int(os.getenv('WEBCAM_FPS_TARGET', '15'))  # Target FPS for webcams

    # GPU/CPU Provider Configuration
    @classmethod
    def get_insightface_providers(cls):
        """Get optimized providers for high FPS"""
        providers = []

        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()

            # Prioritize CUDA for speed
            if 'CUDAExecutionProvider' in available_providers:
                providers.append(('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kSameAsRequested',  # Faster memory allocation
                    'gpu_mem_limit': 1 * 1024 * 1024 * 1024,  # Reduced to 1GB for speed
                    'cudnn_conv_algo_search': 'HEURISTIC',     # Faster algorithm search
                    'do_copy_in_default_stream': True,
                }))
                print("🚀 CUDA GPU optimized for high FPS")

            # CPU fallback with optimizations
            if 'CPUExecutionProvider' in available_providers:
                providers.append(('CPUExecutionProvider', {
                    'intra_op_num_threads': 4,  # Optimize thread count
                    'inter_op_num_threads': 2,
                    'omp_num_threads': 4,
                }))

            return providers

        except Exception as e:
            print(f"⚠️ Provider setup error: {e}")
            return ['CPUExecutionProvider']

    @classmethod
    def get_all_settings(cls) -> Dict:
        """Get all configuration settings"""
        return {
            'backend_url': cls.BACKEND_URL,
            'recognition_threshold': cls.RECOGNITION_THRESHOLD,
            'detection_threshold': cls.DETECTION_THRESHOLD,
            'frame_processing_interval': cls.FRAME_PROCESSING_INTERVAL,
            'max_concurrent_cameras': cls.MAX_CONCURRENT_CAMERAS,
            'rtsp_timeout': cls.RTSP_TIMEOUT,
            'window_width': cls.WINDOW_WIDTH,
            'window_height': cls.WINDOW_HEIGHT,
            'log_level': cls.LOG_LEVEL,

            # Performance settings
            'processing_width': cls.PROCESSING_WIDTH,
            'processing_height': cls.PROCESSING_HEIGHT,
            'camera_fps': cls.CAMERA_FPS,
            'max_faces': cls.MAX_FACES_PER_FRAME,
            'enable_parallel': cls.ENABLE_PARALLEL_PROCESSING,
            'max_workers': cls.MAX_WORKER_THREADS,

            # InsightFace settings
            'insightface_model': cls.INSIGHTFACE_MODEL,
            'insightface_providers': cls.get_insightface_providers(),

            # Tracking settings
            'track_buffer': cls.TRACK_BUFFER_SIZE,
            'min_track_length': cls.MIN_TRACK_LENGTH,
            'track_threshold': cls.TRACK_MATCH_THRESHOLD,
            'stable_threshold': cls.STABLE_TRACK_THRESHOLD,

            # Quality settings
            'enable_quality_filter': cls.ENABLE_QUALITY_FILTER,
            'min_quality_score': cls.MIN_QUALITY_SCORE,
            'min_blur_score': cls.MIN_BLUR_SCORE,
            'brightness_range': (cls.MIN_BRIGHTNESS, cls.MAX_BRIGHTNESS),

            # Webcam settings
            'enable_webcam': cls.ENABLE_WEBCAM_OPTION,
            'webcam_width': cls.WEBCAM_PROCESSING_WIDTH,
            'webcam_height': cls.WEBCAM_PROCESSING_HEIGHT,
            'webcam_fps': cls.WEBCAM_FPS_TARGET
        }

    @classmethod
    def print_system_info(cls):
        """Print optimized system information"""
        print("🚀 HIGH PERFORMANCE Configuration:")
        print(f"  - InsightFace Model: {cls.INSIGHTFACE_MODEL} (fast)")
        print(f"  - Processing Resolution: {cls.PROCESSING_WIDTH}x{cls.PROCESSING_HEIGHT} (optimized)")
        print(f"  - Frame Skip: Every {cls.FRAME_PROCESSING_INTERVAL} frames")
        print(f"  - Max Faces: {cls.MAX_FACES_PER_FRAME}")
        print(f"  - Recognition Threshold: {cls.RECOGNITION_THRESHOLD}")
        print(f"  - Detection Threshold: {cls.DETECTION_THRESHOLD}")
        print(f"  - Parallel Processing: {cls.ENABLE_PARALLEL_PROCESSING}")
        print(f"  - Worker Threads: {cls.MAX_WORKER_THREADS}")
        print(f"  - Webcam Support: {cls.ENABLE_WEBCAM_OPTION}")

        # Check GPU availability
        providers = cls.get_insightface_providers()
        print(f"  - Available Providers: {len(providers)}")
        for provider in providers:
            if isinstance(provider, tuple):
                print(f"    • {provider[0]}")
            else:
                print(f"    • {provider}")

    @classmethod
    def apply_high_fps_mode(cls):
        """Apply aggressive optimizations for highest FPS"""
        print("🚀 Applying HIGH FPS MODE...")

        # Ultra-low resolution for maximum speed
        cls.PROCESSING_WIDTH = 320
        cls.PROCESSING_HEIGHT = 240
        cls.WEBCAM_PROCESSING_WIDTH = 256
        cls.WEBCAM_PROCESSING_HEIGHT = 192

        # Skip more frames
        cls.FRAME_PROCESSING_INTERVAL = 4

        # Fewer faces per frame
        cls.MAX_FACES_PER_FRAME = 5

        # Lower quality thresholds
        cls.MIN_QUALITY_SCORE = 0.1
        cls.RECOGNITION_THRESHOLD = 0.35
        cls.DETECTION_THRESHOLD = 0.3

        # Disable quality filtering completely
        cls.ENABLE_QUALITY_FILTER = False

        print("✅ HIGH FPS MODE enabled - expect 15-25 FPS")

    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        try:
            # Create data directories
            cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
            cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)
            cls.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

            print(f"✅ Created directories: {cls.DATA_DIR}")
            return True
        except Exception as e:
            print(f"❌ Failed to create directories: {e}")
            return False

    @classmethod
    def get_webcam_devices(cls):
        """Get available webcam devices"""
        import cv2
        devices = []

        for i in range(10):  # Check first 10 indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    devices.append({
                        'id': i,
                        'name': f'Webcam {i}',
                        'url': i,
                        'type': 'webcam'
                    })
                cap.release()

        return devices


# Global config instance
config = Config()

# Check for high FPS mode
if os.getenv('HIGH_FPS_MODE', '0') == '1':
    config.apply_high_fps_mode()

# Print configuration on import
if os.getenv('DEBUG_CONFIG', '0') == '1':
    config.print_system_info()