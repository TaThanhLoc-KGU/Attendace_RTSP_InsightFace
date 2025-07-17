"""
Configuration settings for Camera RTSP Application - OPTIMIZED VERSION
High performance configuration với GPU support
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration - OPTIMIZED FOR PERFORMANCE"""

    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    LOGS_DIR = DATA_DIR / "logs"
    EMBEDDINGS_DIR = DATA_DIR / "embeddings"

    # Backend API
    BACKEND_URL = os.getenv('SPRING_BOOT_API_URL', 'http://localhost:8080/api')
    WEBSOCKET_URL = os.getenv('WEBSOCKET_URL', 'ws://localhost:8080/ws')

    # Recognition Settings - OPTIMIZED FOR PERFORMANCE
    RECOGNITION_THRESHOLD = float(os.getenv('RECOGNITION_THRESHOLD', '0.45'))  # Slightly lower for better recognition
    DETECTION_THRESHOLD = float(os.getenv('DETECTION_THRESHOLD', '0.4'))      # Lower for better detection
    FRAME_PROCESSING_INTERVAL = int(os.getenv('FRAME_PROCESSING_INTERVAL', '1'))  # Process every frame for better tracking
    MAX_CONCURRENT_CAMERAS = int(os.getenv('MAX_CONCURRENT_CAMERAS', '6'))    # More cameras

    # Cache Settings
    EMBEDDING_CACHE_DIR = os.getenv('EMBEDDING_CACHE_DIR', str(EMBEDDINGS_DIR))
    FACE_DB_PATH = os.getenv('FACE_DB_PATH', str(DATA_DIR / 'faces_db.json'))
    CACHE_EXPIRE_HOURS = int(os.getenv('CACHE_EXPIRE_HOURS', '24'))

    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_DIR = os.getenv('LOG_DIR', str(LOGS_DIR))

    # RTSP Settings - OPTIMIZED
    RTSP_TIMEOUT = int(os.getenv('RTSP_TIMEOUT', '5000'))  # Longer timeout
    HLS_SEGMENT_DURATION = int(os.getenv('HLS_SEGMENT_DURATION', '2'))

    # Camera Performance Settings
    CAMERA_BUFFER_SIZE = int(os.getenv('CAMERA_BUFFER_SIZE', '1'))
    CAMERA_FPS = int(os.getenv('CAMERA_FPS', '30'))
    PROCESSING_WIDTH = int(os.getenv('PROCESSING_WIDTH', '640'))    # Processing resolution
    PROCESSING_HEIGHT = int(os.getenv('PROCESSING_HEIGHT', '480'))  # Processing resolution
    DISPLAY_WIDTH = int(os.getenv('DISPLAY_WIDTH', '640'))          # Display resolution
    DISPLAY_HEIGHT = int(os.getenv('DISPLAY_HEIGHT', '480'))        # Display resolution

    # UI Settings
    WINDOW_WIDTH = int(os.getenv('WINDOW_WIDTH', '1400'))
    WINDOW_HEIGHT = int(os.getenv('WINDOW_HEIGHT', '900'))

    # InsightFace Settings - OPTIMIZED FOR GPU/CPU
    INSIGHTFACE_MODEL = os.getenv('INSIGHTFACE_MODEL', 'buffalo_l')  # Best model

    # GPU/CPU Provider Configuration
    @classmethod
    def get_insightface_providers(cls):
        """Get optimized providers based on system capabilities"""
        providers = []

        # Try to detect CUDA availability
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()

            # Prioritize CUDA if available
            if 'CUDAExecutionProvider' in available_providers:
                providers.append(('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }))
                print("🚀 CUDA GPU detected and enabled")

            # Add CPU as fallback
            if 'CPUExecutionProvider' in available_providers:
                providers.append(('CPUExecutionProvider', {
                    'intra_op_num_threads': 4,  # Use 4 threads
                    'inter_op_num_threads': 4,
                }))

        except ImportError:
            print("⚠️ ONNXRuntime not available, using default providers")
            providers = ['CPUExecutionProvider']
        except Exception as e:
            print(f"⚠️ Error detecting providers: {e}")
            providers = ['CPUExecutionProvider']

        if not providers:
            providers = ['CPUExecutionProvider']

        return providers

    # Performance Optimization Settings
    ENABLE_PARALLEL_PROCESSING = bool(int(os.getenv('ENABLE_PARALLEL_PROCESSING', '1')))
    MAX_WORKER_THREADS = int(os.getenv('MAX_WORKER_THREADS', '4'))

    # Face Recognition Optimization
    MAX_FACES_PER_FRAME = int(os.getenv('MAX_FACES_PER_FRAME', '15'))
    MIN_FACE_SIZE = int(os.getenv('MIN_FACE_SIZE', '40'))
    MAX_FACE_SIZE = int(os.getenv('MAX_FACE_SIZE', '500'))

    # Tracking Optimization
    TRACK_BUFFER_SIZE = int(os.getenv('TRACK_BUFFER_SIZE', '50'))
    MIN_TRACK_LENGTH = int(os.getenv('MIN_TRACK_LENGTH', '5'))
    TRACK_MATCH_THRESHOLD = float(os.getenv('TRACK_MATCH_THRESHOLD', '0.7'))
    STABLE_TRACK_THRESHOLD = float(os.getenv('STABLE_TRACK_THRESHOLD', '0.8'))

    # Quality Assessment
    ENABLE_QUALITY_FILTER = bool(int(os.getenv('ENABLE_QUALITY_FILTER', '1')))
    MIN_QUALITY_SCORE = float(os.getenv('MIN_QUALITY_SCORE', '0.3'))
    MIN_BLUR_SCORE = float(os.getenv('MIN_BLUR_SCORE', '20.0'))
    MIN_BRIGHTNESS = int(os.getenv('MIN_BRIGHTNESS', '30'))
    MAX_BRIGHTNESS = int(os.getenv('MAX_BRIGHTNESS', '220'))

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
            'brightness_range': (cls.MIN_BRIGHTNESS, cls.MAX_BRIGHTNESS)
        }

    @classmethod
    def print_system_info(cls):
        """Print system information for debugging"""
        print("🖥️ System Configuration:")
        print(f"  - InsightFace Model: {cls.INSIGHTFACE_MODEL}")
        print(f"  - Processing Resolution: {cls.PROCESSING_WIDTH}x{cls.PROCESSING_HEIGHT}")
        print(f"  - Max Faces: {cls.MAX_FACES_PER_FRAME}")
        print(f"  - Recognition Threshold: {cls.RECOGNITION_THRESHOLD}")
        print(f"  - Detection Threshold: {cls.DETECTION_THRESHOLD}")
        print(f"  - Parallel Processing: {cls.ENABLE_PARALLEL_PROCESSING}")
        print(f"  - Worker Threads: {cls.MAX_WORKER_THREADS}")

        # Check GPU availability
        providers = cls.get_insightface_providers()
        print(f"  - Available Providers: {len(providers)}")
        for provider in providers:
            if isinstance(provider, tuple):
                print(f"    • {provider[0]}")
            else:
                print(f"    • {provider}")

    @classmethod
    def optimize_for_performance(cls):
        """Apply performance optimizations"""
        # Reduce processing resolution for speed
        cls.PROCESSING_WIDTH = 416
        cls.PROCESSING_HEIGHT = 312

        # Reduce quality requirements
        cls.MIN_QUALITY_SCORE = 0.2
        cls.MIN_BLUR_SCORE = 15.0

        # Increase parallel processing
        cls.MAX_WORKER_THREADS = 6

        # Process every frame
        cls.FRAME_PROCESSING_INTERVAL = 1

        print("🚀 Performance optimizations applied")

    @classmethod
    def optimize_for_accuracy(cls):
        """Apply accuracy optimizations"""
        # Higher processing resolution
        cls.PROCESSING_WIDTH = 640
        cls.PROCESSING_HEIGHT = 480

        # Stricter quality requirements
        cls.MIN_QUALITY_SCORE = 0.4
        cls.MIN_BLUR_SCORE = 30.0

        # Higher thresholds
        cls.RECOGNITION_THRESHOLD = 0.6
        cls.STABLE_TRACK_THRESHOLD = 0.85

        print("🎯 Accuracy optimizations applied")


# Global config instance
config = Config()

# Check if we should optimize for performance vs accuracy
performance_mode = os.getenv('PERFORMANCE_MODE', 'balanced').lower()
if performance_mode == 'speed':
    config.optimize_for_performance()
elif performance_mode == 'accuracy':
    config.optimize_for_accuracy()

# Print configuration on import
if os.getenv('DEBUG_CONFIG', '0') == '1':
    config.print_system_info()