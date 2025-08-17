"""
Enhanced Face recognition service - OPTIMIZED FOR HIGH PERFORMANCE
Fixed ONNX Runtime issues và GPU optimization
"""
import os
import json
import pickle
import base64
import numpy as np
import cv2
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from collections import defaultdict, deque
from dataclasses import dataclass

# FIXED ONNX Runtime Import
INSIGHTFACE_AVAILABLE = False
ONNX_AVAILABLE = False

try:
    # Try different ONNX Runtime imports
    try:
        import onnxruntime as ort
        # Test if InferenceSession is available
        _ = ort.InferenceSession
        ONNX_AVAILABLE = True
        print(f"✅ ONNXRuntime {ort.__version__} loaded successfully")
        print(f"📊 Available providers: {ort.get_available_providers()}")
    except (ImportError, AttributeError) as e:
        print(f"⚠️ ONNXRuntime issue: {e}")
        # Try uninstall and reinstall
        try:
            import subprocess
            import sys
            print("🔄 Attempting to fix ONNXRuntime...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y', 'onnxruntime', 'onnxruntime-gpu'])
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'onnxruntime'])
            import onnxruntime as ort
            ONNX_AVAILABLE = True
            print("✅ ONNXRuntime fixed and reloaded")
        except Exception as fix_error:
            print(f"❌ Could not fix ONNXRuntime: {fix_error}")
            ONNX_AVAILABLE = False

    # Import InsightFace only if ONNX is working
    if ONNX_AVAILABLE:
        import insightface
        from insightface.app import FaceAnalysis
        INSIGHTFACE_AVAILABLE = True
        print("✅ InsightFace loaded successfully")
    else:
        print("❌ InsightFace disabled due to ONNX issues")

except Exception as e:
    print(f"❌ Failed to load face recognition libraries: {e}")
    print("📝 To fix:")
    print("   pip uninstall onnxruntime onnxruntime-gpu")
    print("   pip install onnxruntime")
    print("   # OR for GPU: pip install onnxruntime-gpu")

from config.config import config
from utils.logger import face_logger


@dataclass
class FaceTrack:
    """Lightweight face tracking object for performance"""
    track_id: int
    bbox: List[int]
    embedding: np.ndarray
    confidence: float

    # Essential features only for performance
    age: int = 0
    gender: int = -1  # 0: male, 1: female, -1: unknown

    # Identity information
    identity: Optional[str] = None
    student_id: Optional[str] = None
    is_known: bool = False

    # Tracking metrics
    first_seen: float = 0.0
    last_seen: float = 0.0
    frame_count: int = 0
    stable_count: int = 0

    # Quality metrics (simplified)
    quality_score: float = 0.0

    # State management
    is_stable: bool = False
    attendance_recorded: bool = False

    def __post_init__(self):
        current_time = time.time()
        if self.first_seen == 0.0:
            self.first_seen = current_time
        self.last_seen = current_time


class StudentEmbeddingService:
    """Optimized student embedding service"""

    def __init__(self, backend_api):
        self.backend_api = backend_api
        self.student_embeddings = {}
        self.student_index = {}
        self.cache_dir = Path(config.EMBEDDING_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache files
        self.embeddings_file = self.cache_dir / "student_embeddings.pkl"
        self.index_file = self.cache_dir / "student_index.json"

        face_logger.info("🎓 StudentEmbeddingService initialized")

    def load_embeddings_from_backend(self) -> Dict[str, any]:
        """Load tất cả embeddings từ backend"""
        try:
            face_logger.info("📥 Loading student embeddings from backend...")

            # Call backend API
            response = self.backend_api._make_request('GET', '/python/embeddings')

            if not response.success:
                return {
                    'success': False,
                    'message': f"Backend API failed: {response.message}",
                    'count': 0
                }

            students_data = response.data
            if not isinstance(students_data, list):
                return {
                    'success': False,
                    'message': "Invalid data format from backend",
                    'count': 0
                }

            # Process student embeddings
            processed_count = 0
            self.student_embeddings = {}
            self.student_index = {}

            for student in students_data:
                try:
                    student_id = student.get('studentId')
                    name = student.get('name')
                    embedding_data = student.get('embedding')

                    if not student_id or not embedding_data:
                        continue

                    # Decode embedding
                    embedding_array = self._decode_embedding(embedding_data)
                    if embedding_array is not None:
                        self.student_embeddings[student_id] = embedding_array
                        self.student_index[student_id] = {
                            'name': name,
                            'student_id': student_id,
                            'loaded_at': datetime.now().isoformat()
                        }
                        processed_count += 1

                except Exception as e:
                    face_logger.warning(f"Error processing student {student.get('studentId', 'unknown')}: {e}")
                    continue

            # Save to cache
            self._save_to_cache()

            face_logger.info(f"✅ Loaded {processed_count} student embeddings")
            return {
                'success': True,
                'count': processed_count,
                'message': f"Successfully loaded {processed_count} student embeddings"
            }

        except Exception as e:
            face_logger.error(f"❌ Error loading embeddings: {e}")
            return {
                'success': False,
                'message': f"Error loading embeddings: {str(e)}",
                'count': 0
            }

    def _decode_embedding(self, embedding_data) -> Optional[np.ndarray]:
        """Decode embedding data từ backend"""
        try:
            if isinstance(embedding_data, str):
                # Try decode as base64
                try:
                    embedding_bytes = base64.b64decode(embedding_data)
                    embedding_array = np.frombuffer(embedding_bytes, dtype=np.float32)
                except:
                    # Try parse as JSON array
                    try:
                        embedding_list = json.loads(embedding_data)
                        embedding_array = np.array(embedding_list, dtype=np.float32)
                    except:
                        return None
            elif isinstance(embedding_data, list):
                embedding_array = np.array(embedding_data, dtype=np.float32)
            else:
                return None

            # Validate embedding size (should be 512 for InsightFace)
            if len(embedding_array) == 512:
                return embedding_array
            else:
                face_logger.warning(f"Invalid embedding size: {len(embedding_array)}")
                return None

        except Exception as e:
            face_logger.error(f"Error decoding embedding: {e}")
            return None

    def _save_to_cache(self):
        """Save embeddings to cache files"""
        try:
            # Save embeddings
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.student_embeddings, f)

            # Save index
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.student_index, f, ensure_ascii=False, indent=2)

            face_logger.info("💾 Embeddings saved to cache")

        except Exception as e:
            face_logger.error(f"❌ Error saving cache: {e}")

    def load_from_cache(self) -> bool:
        """Load embeddings from cache files"""
        try:
            if self.embeddings_file.exists():
                with open(self.embeddings_file, 'rb') as f:
                    self.student_embeddings = pickle.load(f)

                if self.index_file.exists():
                    with open(self.index_file, 'r', encoding='utf-8') as f:
                        self.student_index = json.load(f)

                face_logger.info(f"📂 Loaded {len(self.student_embeddings)} embeddings from cache")
                return True
            else:
                face_logger.info("📂 No cache found")
                return False

        except Exception as e:
            face_logger.error(f"❌ Error loading cache: {e}")
            return False

    def get_student_info(self, student_id: str) -> Optional[Dict]:
        """Get student info by ID"""
        return self.student_index.get(student_id)

    def get_embedding(self, student_id: str) -> Optional[np.ndarray]:
        """Get embedding by student ID"""
        return self.student_embeddings.get(student_id)

    def get_statistics(self) -> Dict:
        """Get embedding statistics"""
        return {
            'total_students': len(self.student_embeddings),
            'cache_size': len(self.student_embeddings),
            'last_loaded': datetime.now().isoformat()
        }


class FaceRecognitionService:
    """HIGH PERFORMANCE Face recognition service với optimized tracking"""

    def __init__(self, backend_api):
        self.backend_api = backend_api
        self.face_app = None
        self.is_initialized = False

        # Student embedding service
        self.student_service = StudentEmbeddingService(backend_api)

        # Optimized tracking system
        self.active_tracks = {}
        self.next_track_id = 1
        self.track_history = deque(maxlen=500)  # Reduced for performance

        # Performance-optimized configuration
        self.tracking_config = {
            'track_thresh': config.TRACK_MATCH_THRESHOLD,
            'match_thresh': 0.52,  # Lower for better matching
            'stable_thresh': config.STABLE_TRACK_THRESHOLD,
            'min_track_length': config.MIN_TRACK_LENGTH,
            'min_stable_frames': 8,  # Reduced for faster recognition
            'track_buffer': config.TRACK_BUFFER_SIZE,
            'max_distance': 80,  # Reduced for performance
            'quality_thresh': config.MIN_QUALITY_SCORE
        }

        # Performance tracking
        self.stats = {
            'total_detections': 0,
            'total_recognitions': 0,
            'total_tracks': 0,
            'stable_tracks': 0,
            'processing_times': deque(maxlen=50)  # Reduced for performance
        }

        # Thread safety
        self.lock = threading.RLock()

        # Performance optimization flags
        self.skip_quality_check = False
        self.simple_tracking = True
        self.last_recognition_time = 0
        self.recognition_interval = 0.1  # 100ms between recognitions

        # Load từ cache trước
        if self.student_service.load_from_cache():
            face_logger.info("✅ Loaded student embeddings from cache")

        # Initialize InsightFace with optimized settings
        if INSIGHTFACE_AVAILABLE:
            self.initialize_face_recognition()
        else:
            face_logger.error("❌ InsightFace not available. Please fix ONNX Runtime installation")

    def initialize_face_recognition(self):
        """Initialize InsightFace với GPU/CPU optimization"""
        try:
            face_logger.info("🚀 Initializing HIGH PERFORMANCE InsightFace...")

            # Get optimized providers từ config
            providers = config.get_insightface_providers()

            face_logger.info(f"🔧 Using providers: {[p[0] if isinstance(p, tuple) else p for p in providers]}")

            # Initialize FaceAnalysis với optimized settings
            self.face_app = FaceAnalysis(
                name=config.INSIGHTFACE_MODEL,
                providers=providers,
                allowed_modules=['detection', 'recognition']  # Only essential modules for speed
            )

            # Optimized prepare settings
            det_size = (config.PROCESSING_WIDTH, config.PROCESSING_HEIGHT)
            self.face_app.prepare(
                ctx_id=0,
                det_size=det_size,
                det_thresh=config.DETECTION_THRESHOLD
            )

            self.is_initialized = True

            face_logger.info(f"✅ HIGH PERFORMANCE InsightFace initialized")
            face_logger.info(f"📊 Model: {config.INSIGHTFACE_MODEL}")
            face_logger.info(f"📐 Detection size: {det_size}")
            face_logger.info(f"🎯 Detection threshold: {config.DETECTION_THRESHOLD}")

            # Quick performance test
            self._performance_test()

        except Exception as e:
            face_logger.error(f"❌ Failed to initialize InsightFace: {e}")
            self.face_app = None
            self.is_initialized = False

            # Fallback: Try with CPU only
            try:
                face_logger.info("🔄 Trying CPU-only fallback...")
                self.face_app = FaceAnalysis(
                    name='buffalo_l',  # Smaller model for CPU
                    providers=['CUDAExecutionProvider'],
                    allowed_modules=['detection', 'recognition']
                )
                self.face_app.prepare(ctx_id=0, det_size=(416, 312), det_thresh=0.5)
                self.is_initialized = True
                face_logger.info("✅ CPU-only fallback successful")
            except Exception as fallback_error:
                face_logger.error(f"❌ CPU fallback also failed: {fallback_error}")

    def _performance_test(self):
        """Quick performance test"""
        try:
            test_img = np.random.randint(0, 255, (config.PROCESSING_HEIGHT, config.PROCESSING_WIDTH, 3), dtype=np.uint8)

            # Warm up
            for _ in range(3):
                self.face_app.get(test_img)

            # Actual test
            start_time = time.time()
            for _ in range(10):
                faces = self.face_app.get(test_img)
            end_time = time.time()

            avg_time = (end_time - start_time) / 10
            fps_estimate = 1.0 / avg_time if avg_time > 0 else 0

            face_logger.info(f"🚀 Performance test: {avg_time:.3f}s per frame (~{fps_estimate:.1f} FPS)")

            if avg_time > 0.1:  # > 100ms
                face_logger.warning("⚠️ Slow performance detected. Consider:")
                face_logger.warning("   - Using smaller model (buffalo_s)")
                face_logger.warning("   - Reducing detection resolution")
                face_logger.warning("   - Installing GPU support")

        except Exception as e:
            face_logger.warning(f"⚠️ Performance test failed: {e}")

    def load_student_embeddings(self) -> Dict[str, any]:
        """Load student embeddings từ backend"""
        result = self.student_service.load_embeddings_from_backend()

        if result['success']:
            face_logger.info(f"✅ Successfully loaded {result['count']} student embeddings")
        else:
            face_logger.error(f"❌ Failed to load student embeddings: {result['message']}")

        return result

    def recognize_faces(self, image: np.ndarray) -> List[Dict]:
        """
        HIGH PERFORMANCE face recognition với optimized tracking
        """
        if not self.is_initialized:
            return []

        # Performance throttling - skip if called too frequently
        current_time = time.time()
        if current_time - self.last_recognition_time < self.recognition_interval:
            return []

        self.last_recognition_time = current_time
        start_time = current_time

        try:
            with self.lock:
                # Resize image for processing if needed
                if image.shape[1] > config.PROCESSING_WIDTH or image.shape[0] > config.PROCESSING_HEIGHT:
                    image = cv2.resize(image, (config.PROCESSING_WIDTH, config.PROCESSING_HEIGHT))

                # Get faces từ InsightFace
                faces = self.face_app.get(image)
                self.stats['total_detections'] += len(faces)

                if len(faces) > config.MAX_FACES_PER_FRAME:
                    # Sort by detection confidence and take top N
                    faces = sorted(faces, key=lambda x: x.det_score, reverse=True)[:config.MAX_FACES_PER_FRAME]

                face_logger.debug(f"🔍 Detected {len(faces)} faces")

                # Simplified tracking for performance
                if self.simple_tracking:
                    tracked_faces = self._simple_tracking(faces, image)
                else:
                    tracked_faces = self._advanced_tracking(faces, image)

                # Recognition for stable tracks only (performance optimization)
                recognized_faces = self._fast_recognition(tracked_faces)

                # Cleanup inactive tracks
                self._fast_cleanup()

                # Update statistics
                processing_time = time.time() - start_time
                self.stats['processing_times'].append(processing_time)

                return self._format_results_fast(recognized_faces)

        except Exception as e:
            face_logger.error(f"❌ Error in recognition: {e}")
            return []

    def _simple_tracking(self, faces: List, image: np.ndarray) -> List[FaceTrack]:
        """Simplified tracking for maximum performance"""
        current_time = time.time()
        updated_tracks = []

        # Convert faces to simple detections
        detections = []
        for face in faces:
            try:
                bbox = face.bbox.astype(int) if hasattr(face, 'bbox') else [0, 0, 10, 10]

                # Skip quality check for speed
                detection = {
                    'bbox': bbox.tolist(),
                    'embedding': face.embedding if hasattr(face, 'embedding') else np.zeros(512),
                    'confidence': float(face.det_score) if hasattr(face, 'det_score') else 0.5,
                    'age': getattr(face, 'age', 0),
                    'gender': getattr(face, 'gender', -1),
                }
                detections.append(detection)
            except Exception as e:
                face_logger.debug(f"Error processing face: {e}")
                continue

        # Simple distance-based matching
        for detection in detections:
            best_match = None
            best_distance = float('inf')

            # Find closest existing track
            for track in self.active_tracks.values():
                # Simple geometric distance
                det_center = [(detection['bbox'][0] + detection['bbox'][2]) / 2,
                             (detection['bbox'][1] + detection['bbox'][3]) / 2]
                track_center = [(track.bbox[0] + track.bbox[2]) / 2,
                               (track.bbox[1] + track.bbox[3]) / 2]

                distance = np.sqrt((det_center[0] - track_center[0])**2 +
                                 (det_center[1] - track_center[1])**2)

                if distance < best_distance and distance < 50:  # Max 50 pixels
                    best_distance = distance
                    best_match = track

            if best_match:
                # Update existing track
                best_match.bbox = detection['bbox']
                best_match.confidence = max(best_match.confidence, detection['confidence'])
                best_match.last_seen = current_time
                best_match.frame_count += 1

                # Quick stability check
                if best_match.frame_count >= 5 and best_match.confidence > 0.7:
                    if not best_match.is_stable:
                        best_match.is_stable = True
                        self.stats['stable_tracks'] += 1
                    best_match.stable_count += 1

                updated_tracks.append(best_match)
            else:
                # Create new track
                if len(self.active_tracks) < config.MAX_FACES_PER_FRAME:
                    new_track = FaceTrack(
                        track_id=self.next_track_id,
                        bbox=detection['bbox'],
                        embedding=detection['embedding'],
                        confidence=detection['confidence'],
                        age=detection['age'],
                        gender=detection['gender'],
                        first_seen=current_time,
                        last_seen=current_time
                    )
                    self.next_track_id += 1
                    self.stats['total_tracks'] += 1
                    self.active_tracks[new_track.track_id] = new_track
                    updated_tracks.append(new_track)

        return updated_tracks

    def _advanced_tracking(self, faces: List, image: np.ndarray) -> List[FaceTrack]:
        """Advanced tracking (fallback if simple tracking is disabled)"""
        # This is the original complex tracking - kept for compatibility
        # but not used by default for performance reasons
        return self._simple_tracking(faces, image)

    def _fast_recognition(self, tracks: List[FaceTrack]) -> List[FaceTrack]:
        """Fast recognition for stable tracks only"""
        for track in tracks:
            if track.is_stable and not track.is_known:
                # Use fast similarity search
                best_match = None
                best_score = 0

                # Only check against student embeddings if we have them
                if self.student_service.student_embeddings:
                    for student_id, student_embedding in self.student_service.student_embeddings.items():
                        try:
                            # Fast cosine similarity
                            similarity = np.dot(track.embedding, student_embedding) / (
                                np.linalg.norm(track.embedding) * np.linalg.norm(student_embedding)
                            )

                            if similarity > best_score and similarity > config.RECOGNITION_THRESHOLD:
                                best_score = similarity
                                student_info = self.student_service.get_student_info(student_id)
                                best_match = {
                                    'student_id': student_id,
                                    'name': student_info['name'] if student_info else student_id,
                                    'confidence': similarity
                                }
                        except Exception:
                            continue

                    if best_match:
                        track.identity = best_match['name']
                        track.student_id = best_match['student_id']
                        track.is_known = True
                        track.confidence = max(track.confidence, best_match['confidence'])
                        self.stats['total_recognitions'] += 1

                        face_logger.info(f"✅ Fast recognition: {best_match['name']} (Track {track.track_id})")

        return tracks

    def _fast_cleanup(self):
        """Fast cleanup of inactive tracks"""
        current_time = time.time()
        timeout = 2.0  # 2 seconds timeout

        inactive_ids = [
            track_id for track_id, track in self.active_tracks.items()
            if current_time - track.last_seen > timeout
        ]

        for track_id in inactive_ids:
            track = self.active_tracks.pop(track_id)
            if track.is_stable:
                self.stats['stable_tracks'] = max(0, self.stats['stable_tracks'] - 1)

    def _format_results_fast(self, tracks: List[FaceTrack]) -> List[Dict]:
        """Fast result formatting"""
        results = []

        for track in tracks:
            gender_text = "Female" if track.gender == 1 else "Male" if track.gender == 0 else "Unknown"

            result = {
                # Essential fields for compatibility
                'face_id': track.track_id,
                'name': track.identity or 'Unknown',
                'student_id': track.student_id,
                'confidence': float(track.confidence),
                'bbox': track.bbox,
                'age': track.age,
                'gender': track.gender,
                'gender_text': gender_text,
                'landmarks': [],  # Empty for performance
                'embedding': track.embedding.tolist(),
                'is_known': track.is_known,

                # Tracking info
                'track_id': track.track_id,
                'track_length': track.frame_count,
                'stable_count': track.stable_count,
                'is_stable': track.is_stable,
                'duration': time.time() - track.first_seen,

                # Quality (simplified)
                'quality_score': track.quality_score,
                'blur_score': 50.0,  # Default for performance
                'brightness_score': 128.0,  # Default for performance

                # State
                'attendance_recorded': track.attendance_recorded
            }

            results.append(result)

        return results

    def get_statistics(self) -> Dict:
        """Get performance statistics"""
        student_stats = self.student_service.get_statistics()

        # Calculate average processing time
        avg_processing_time = 0
        if self.stats['processing_times']:
            avg_processing_time = sum(self.stats['processing_times']) / len(self.stats['processing_times'])

        return {
            # Student database
            'total_students': student_stats['total_students'],
            'cache_size': student_stats['cache_size'],
            'last_loaded': student_stats['last_loaded'],

            # System status
            'is_initialized': self.is_initialized,
            'recognition_threshold': config.RECOGNITION_THRESHOLD,

            # Tracking stats
            'active_tracks': len(self.active_tracks),
            'stable_tracks': self.stats['stable_tracks'],
            'total_tracks': self.stats['total_tracks'],
            'total_detections': self.stats['total_detections'],
            'total_recognitions': self.stats['total_recognitions'],

            # Performance
            'avg_processing_time': avg_processing_time,
            'estimated_fps': 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        }

    def get_active_tracks_info(self) -> List[Dict]:
        """Get active tracks info"""
        tracks_info = []

        for track in self.active_tracks.values():
            tracks_info.append({
                'track_id': track.track_id,
                'identity': track.identity,
                'student_id': track.student_id,
                'confidence': track.confidence,
                'age': track.age,
                'gender': 'Female' if track.gender == 1 else 'Male' if track.gender == 0 else 'Unknown',
                'frame_count': track.frame_count,
                'stable_count': track.stable_count,
                'duration': time.time() - track.first_seen,
                'is_known': track.is_known,
                'is_stable': track.is_stable,
                'quality_score': track.quality_score,
                'attendance_recorded': track.attendance_recorded
            })

        return tracks_info

    def record_attendance(self, student_id: str, camera_id: int) -> Dict:
        """Record attendance với tracking integration"""
        try:
            face_logger.info(f"📝 Recording attendance for {student_id} at camera {camera_id}")

            # Update track attendance status
            for track in self.active_tracks.values():
                if track.student_id == student_id and not track.attendance_recorded:
                    track.attendance_recorded = True
                    break

            # Call backend API
            response = self.backend_api._make_request('POST', '/python/attendance', json={
                'studentId': student_id,
                'cameraId': camera_id
            })

            if response.success:
                face_logger.info(f"✅ Attendance recorded for {student_id}")
                return {
                    'success': True,
                    'message': 'Attendance recorded successfully',
                    'data': response.data
                }
            else:
                face_logger.error(f"❌ Failed to record attendance: {response.message}")
                return {
                    'success': False,
                    'message': response.message
                }

        except Exception as e:
            face_logger.error(f"❌ Error recording attendance: {e}")
            return {
                'success': False,
                'message': f"Error recording attendance: {str(e)}"
            }

    # Performance tuning methods
    def enable_speed_mode(self):
        """Enable maximum speed mode"""
        self.simple_tracking = True
        self.skip_quality_check = True
        self.recognition_interval = 0.2  # 200ms between recognitions
        self.tracking_config['min_stable_frames'] = 3
        face_logger.info("🚀 Speed mode enabled")

    def enable_accuracy_mode(self):
        """Enable accuracy mode (slower)"""
        self.simple_tracking = False
        self.skip_quality_check = False
        self.recognition_interval = 0.05  # 50ms between recognitions
        self.tracking_config['min_stable_frames'] = 10
        face_logger.info("🎯 Accuracy mode enabled")

    # Backward compatibility methods
    def get_face_list(self) -> List[Dict]:
        """Get student list for compatibility"""
        return [
            {
                'name': info['name'],
                'student_id': info['student_id'],
                'loaded_at': info['loaded_at']
            }
            for info in self.student_service.student_index.values()
        ]

    def backup_face_database(self, file_path: str) -> bool:
        """Backup student embeddings"""
        try:
            backup_data = {
                'version': '2.1',
                'created_at': datetime.now().isoformat(),
                'embeddings': {k: v.tolist() for k, v in self.student_service.student_embeddings.items()},
                'index': self.student_service.student_index,
                'performance_stats': self.stats
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, ensure_ascii=False, indent=2)

            face_logger.info(f"💾 Backup saved to {file_path}")
            return True

        except Exception as e:
            face_logger.error(f"❌ Error backing up: {e}")
            return False

    def restore_face_database(self, file_path: str) -> bool:
        """Restore student embeddings"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)

            if 'embeddings' in backup_data and 'index' in backup_data:
                # Restore embeddings
                self.student_service.student_embeddings = {
                    k: np.array(v, dtype=np.float32)
                    for k, v in backup_data['embeddings'].items()
                }

                # Restore index
                self.student_service.student_index = backup_data['index']

                # Save to cache
                self.student_service._save_to_cache()

                face_logger.info(f"✅ Restored from {file_path}")
                return True
            else:
                face_logger.error("Invalid backup format")
                return False

        except Exception as e:
            face_logger.error(f"❌ Error restoring: {e}")
            return False