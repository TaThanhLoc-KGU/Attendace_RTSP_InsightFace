"""
High Performance Face Recognition Service
Optimized for 15-30 FPS with frame skipping and efficient processing
"""
import cv2
import numpy as np
import time
import threading
from collections import deque
from typing import Dict, List, Optional
from datetime import datetime
import queue

try:
    INSIGHTFACE_AVAILABLE = True
    from insightface.app import FaceAnalysis
except ImportError:
    INSIGHTFACE_AVAILABLE = False

from services.backend_api import BackendAPI
from config.config import config
from utils.logger import face_logger


class StudentEmbeddingService:
    """Optimized student embedding service"""

    def __init__(self, backend_api: BackendAPI):
        self.backend_api = backend_api
        self.student_embeddings: Dict[int, Dict] = {}
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.last_loaded = None
        self.lock = threading.RLock()

    def load_embeddings_from_backend(self) -> Dict[str, any]:
        """Load student embeddings from backend - FIXED API endpoint"""
        try:
            with self.lock:
                face_logger.info("🔄 Loading student embeddings from backend...")

                # FIXED: Use correct API endpoint
                response = self.backend_api.get_all_embeddings()  # This calls /python/embeddings

                if not response.success:
                    face_logger.error(f"❌ Backend API failed: {response.message}")
                    return {
                        'success': False,
                        'message': response.message,
                        'count': 0
                    }

                embeddings_data = response.data or []
                face_logger.info(f"📥 Received {len(embeddings_data)} students from backend")

                self.student_embeddings.clear()
                self.embedding_cache.clear()

                # Process embeddings with optimization
                successful_count = 0
                for student_data in embeddings_data:
                    try:
                        # Handle different response formats
                        if isinstance(student_data, dict):
                            # Check multiple possible field names for ID
                            student_id = (
                                    student_data.get('id') or
                                    student_data.get('studentId') or
                                    student_data.get('maSv') or
                                    student_data.get('sinhVienId')
                            )

                            # Check multiple possible field names for embedding
                            embedding_str = (
                                    student_data.get('embedding') or
                                    student_data.get('faceEmbedding') or
                                    student_data.get('embeddings')
                            )

                            # Check multiple possible field names for name
                            name = (
                                    student_data.get('name') or
                                    student_data.get('ten') or
                                    student_data.get('hoTen') or
                                    student_data.get('fullName') or
                                    f'Student {student_id}'
                            )

                            # Check for student code
                            student_code = (
                                    student_data.get('studentCode') or
                                    student_data.get('maSv') or
                                    student_data.get('code') or
                                    ''
                            )

                            face_logger.debug(
                                f"🔍 Processing student: ID={student_id}, Name={name}, HasEmbedding={bool(embedding_str)}")

                            if student_id and embedding_str:
                                try:
                                    # Parse embedding based on format
                                    if isinstance(embedding_str, str):
                                        if embedding_str.startswith('[') and embedding_str.endswith(']'):
                                            # JSON array format: "[1.0, 2.0, 3.0]"
                                            import json
                                            embedding_list = json.loads(embedding_str)
                                            embedding = np.array(embedding_list, dtype=np.float32)
                                        else:
                                            # Comma-separated format: "1.0,2.0,3.0"
                                            embedding = np.fromstring(embedding_str, sep=',', dtype=np.float32)
                                    elif isinstance(embedding_str, list):
                                        # Already a list
                                        embedding = np.array(embedding_str, dtype=np.float32)
                                    else:
                                        face_logger.warning(
                                            f"⚠️ Unknown embedding format for student {student_id}: {type(embedding_str)}")
                                        continue

                                    # Validate embedding
                                    if len(embedding) == 0:
                                        face_logger.warning(f"⚠️ Empty embedding for student {student_id}")
                                        continue

                                    # Normalize embedding for faster comparison
                                    if np.linalg.norm(embedding) > 0:
                                        embedding = embedding / np.linalg.norm(embedding)
                                    else:
                                        face_logger.warning(f"⚠️ Zero embedding for student {student_id}")
                                        continue

                                    # Store student data
                                    self.student_embeddings[student_id] = {
                                        'name': name,
                                        'student_code': student_code,
                                        'embedding': embedding,
                                        'class_name': student_data.get('className', student_data.get('lopHoc', '')),
                                        'email': student_data.get('email', ''),
                                        'phone': student_data.get('phone', student_data.get('sdt', ''))
                                    }

                                    # Cache for faster lookup
                                    cache_key = f"student_{student_id}"
                                    self.embedding_cache[cache_key] = embedding

                                    successful_count += 1
                                    face_logger.debug(f"✅ Processed student {student_id}: {name}")

                                except Exception as e:
                                    face_logger.error(f"❌ Error processing embedding for student {student_id}: {e}")
                                    continue
                            else:
                                if not student_id:
                                    face_logger.warning(f"⚠️ Student missing ID: {student_data}")
                                if not embedding_str:
                                    face_logger.warning(f"⚠️ Student {student_id} missing embedding")
                        else:
                            face_logger.warning(f"⚠️ Invalid student data format: {type(student_data)}")

                    except Exception as e:
                        face_logger.error(f"❌ Error processing student: {e}")

                self.last_loaded = datetime.now()
                face_logger.info(f"✅ Successfully loaded {successful_count} student embeddings")

                return {
                    'success': True,
                    'count': successful_count,
                    'message': f'Loaded {successful_count} students'
                }

        except Exception as e:
            face_logger.error(f"❌ Error loading embeddings: {e}")
            import traceback
            face_logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return {
                'success': False,
                'message': str(e),
                'count': 0
            }



    def find_student_by_embedding(self, face_embedding: np.ndarray, threshold: float = None) -> Optional[Dict]:
        """Find student by face embedding - OPTIMIZED with vectorized operations"""
        if not self.student_embeddings:
            face_logger.debug("🔍 No student embeddings available for recognition")
            return None

        threshold = threshold or config.RECOGNITION_THRESHOLD

        try:
            # Normalize input embedding
            if np.linalg.norm(face_embedding) > 0:
                face_embedding = face_embedding / np.linalg.norm(face_embedding)
            else:
                face_logger.warning("⚠️ Zero face embedding received")
                return None

            # Vectorized similarity computation for speed
            best_match = None
            best_similarity = threshold

            # Use optimized comparison
            for student_id, student_data in self.student_embeddings.items():
                stored_embedding = student_data['embedding']

                # Fast dot product for normalized embeddings
                similarity = float(np.dot(face_embedding, stored_embedding))

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = {
                        'student_id': student_id,
                        'name': student_data['name'],
                        'student_code': student_data['student_code'],
                        'similarity': similarity,
                        'class_name': student_data['class_name']
                    }
                    face_logger.debug(f"🎯 New best match: {student_data['name']} (similarity: {similarity:.3f})")

            if best_match:
                face_logger.info(
                    f"✅ Student recognized: {best_match['name']} (similarity: {best_match['similarity']:.3f})")
            else:
                face_logger.debug(f"❌ No student match found (threshold: {threshold})")

            return best_match

        except Exception as e:
            face_logger.error(f"❌ Error in recognition: {e}")
            return None

    def get_statistics(self) -> Dict:
        """Get embedding statistics"""
        return {
            'total_students': len(self.student_embeddings),
            'cache_size': len(self.embedding_cache),
            'last_loaded': self.last_loaded.isoformat() if self.last_loaded else None
        }

class FaceRecognitionService:
    """HIGH PERFORMANCE Face recognition service"""

    def __init__(self, backend_api):
        self.backend_api = backend_api
        self.face_app = None
        self.is_initialized = False

        self.face_app = FaceAnalysis(name=config.INSIGHTFACE_MODEL, providers=config.get_insightface_providers())
        self.face_app.prepare(ctx_id=0, det_size=(config.PROCESSING_WIDTH, config.PROCESSING_HEIGHT))
        self.is_initialized = True


        # Student embedding service
        self.student_service = StudentEmbeddingService(backend_api)

        # HIGH PERFORMANCE tracking system
        self.active_tracks = {}
        self.next_track_id = 1
        self.track_history = deque(maxlen=100)  # Reduced for performance

        # OPTIMIZED configuration for speed
        self.tracking_config = {
            'track_thresh': config.TRACK_MATCH_THRESHOLD,
            'match_thresh': 0.45,  # Lower for faster matching
            'stable_thresh': config.STABLE_TRACK_THRESHOLD,
            'min_track_length': config.MIN_TRACK_LENGTH,
            'min_stable_frames': 5,  # Reduced for faster recognition
            'track_buffer': config.TRACK_BUFFER_SIZE,
            'max_distance': 60,  # Reduced for performance
            'quality_thresh': config.MIN_QUALITY_SCORE
        }

        # Performance tracking
        self.stats = {
            'total_detections': 0,
            'total_recognitions': 0,
            'total_tracks': 0,
            'stable_tracks': 0,
            'processing_times': deque(maxlen=30)  # Reduced for performance
        }

        # Thread safety
        self.lock = threading.RLock()

        # HIGH PERFORMANCE optimization flags
        self.skip_quality_check = not config.ENABLE_QUALITY_FILTER
        self.simple_tracking = True
        self.frame_skip_counter = 0

        # Frame processing optimization
        self.last_recognition_time = 0
        self.recognition_interval = 0.1  # 100ms between recognitions
        self.frame_queue = queue.Queue(maxsize=2)  # Small queue for low latency

        # Load embeddings
        if self.student_service.load_embeddings_from_backend():
            face_logger.info("✅ Student embeddings loaded")

        # Initialize InsightFace
        if INSIGHTFACE_AVAILABLE:
            self.initialize_face_recognition()
        else:
            face_logger.error("❌ InsightFace not available")

    def initialize_face_recognition(self):
        """Initialize InsightFace with HIGH PERFORMANCE settings"""
        try:
            face_logger.info("🚀 Initializing HIGH PERFORMANCE InsightFace...")

            # Get optimized providers
            providers = config.get_insightface_providers()
            face_logger.info(f"🔧 Using providers: {[p[0] if isinstance(p, tuple) else p for p in providers]}")

            # Initialize with minimal modules for speed
            self.face_app = FaceAnalysis(
                name=config.INSIGHTFACE_MODEL,
                providers=providers,
                allowed_modules=['detection', 'recognition']  # Only essential modules
            )

            # CRITICAL: Use optimized detection size for speed
            det_size = (config.PROCESSING_WIDTH, config.PROCESSING_HEIGHT)
            self.face_app.prepare(
                ctx_id=0,
                det_size=det_size,
                det_thresh=config.DETECTION_THRESHOLD
            )

            self.is_initialized = True
            face_logger.info(f"✅ HIGH PERFORMANCE InsightFace ready - {config.INSIGHTFACE_MODEL}")
            face_logger.info(f"📐 Detection size: {det_size}")
            face_logger.info(f"🎯 Detection threshold: {config.DETECTION_THRESHOLD}")

        except Exception as e:
            face_logger.error(f"❌ Failed to initialize InsightFace: {e}")
            self.face_app = None
            self.is_initialized = False

    def process_frame(self, frame):
        """Detect faces, draw bounding boxes and recognize students"""
        if not self.is_initialized or frame is None:
            return frame

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.face_app.get(rgb_frame)

            for face in faces:
                box = face.bbox.astype(int)

                # Vẽ khung xanh quanh mặt
                cv2.rectangle(
                    frame,
                    (box[0], box[1]),
                    (box[2], box[3]),
                    (0, 255, 0),
                    2
                )

                # Nhận diện student nếu có embeddings
                if hasattr(face, "embedding") and face.embedding is not None:
                    student = self.student_service.find_student_by_embedding(face.embedding)
                    if student:
                        cv2.putText(
                            frame,
                            f"{student['name']} ({student['student_code']})",
                            (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )
                    else:
                        cv2.putText(
                            frame,
                            "Unknown",
                            (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2
                        )

        except Exception as e:
            face_logger.error(f"❌ Error processing frame: {e}")

        return frame

    def recognize_faces_optimized(self, image: np.ndarray) -> List[Dict]:
        """
        ULTRA HIGH PERFORMANCE face recognition with aggressive optimizations
        """
        if not self.is_initialized:
            return []

        # CRITICAL: Frame skipping for performance
        self.frame_skip_counter += 1
        if self.frame_skip_counter % config.FRAME_PROCESSING_INTERVAL != 0:
            return self._get_cached_results()

        # Throttle recognition calls
        current_time = time.time()
        if current_time - self.last_recognition_time < self.recognition_interval:
            return self._get_cached_results()

        self.last_recognition_time = current_time
        start_time = current_time

        try:
            with self.lock:
                # Backup original image
                original_image = image.copy()

                # Resize for processing if needed
                if image.shape[1] > config.PROCESSING_WIDTH:
                    scale_x = image.shape[1] / config.PROCESSING_WIDTH
                    scale_y = image.shape[0] / config.PROCESSING_HEIGHT
                    image = cv2.resize(image, (config.PROCESSING_WIDTH, config.PROCESSING_HEIGHT))
                else:
                    scale_x = scale_y = 1.0

                # Convert to RGB for InsightFace
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Get faces from InsightFace
                faces = self.face_app.get(rgb_image)
                self.stats['total_detections'] += len(faces)

                # OPTIMIZATION: Limit faces processed
                if len(faces) > config.MAX_FACES_PER_FRAME:
                    faces = sorted(faces, key=lambda x: x.det_score, reverse=True)[:config.MAX_FACES_PER_FRAME]

                # OPTIMIZATION: Skip quality checks for speed
                if not self.skip_quality_check:
                    faces = self._filter_quality(faces, image)

                # Scale bounding boxes back to original size
                for face in faces:
                    if scale_x != 1.0 or scale_y != 1.0:
                        bbox = face.bbox
                        face.bbox = np.array([
                            bbox[0] * scale_x,
                            bbox[1] * scale_y,
                            bbox[2] * scale_x,
                            bbox[3] * scale_y
                        ])

                # HIGH PERFORMANCE tracking
                tracked_faces = self._ultra_fast_tracking(faces, original_image)

                # Recognition for stable tracks
                recognized_faces = []
                for face in tracked_faces:
                    if hasattr(face, "embedding") and face.embedding is not None:
                        emb = face.embedding.astype(np.float32)

                        # Normalize embedding
                        if np.linalg.norm(emb) > 0:
                            emb = emb / np.linalg.norm(emb)

                        best_match = None
                        best_similarity = config.RECOGNITION_THRESHOLD

                        for student_id, student_data in self.student_service.student_embeddings.items():
                            stored_emb = student_data['embedding']

                            # 🚑 FIX: skip nếu dimension mismatch
                            if stored_emb.shape != emb.shape:
                                continue

                            sim = float(np.dot(emb, stored_emb))
                            if sim > best_similarity:
                                best_similarity = sim
                                best_match = {
                                    'student_id': student_id,
                                    'name': student_data['name'],
                                    'student_code': student_data['student_code'],
                                    'similarity': sim,
                                    'class_name': student_data['class_name']
                                }

                        if best_match:
                            face.recognition = best_match
                        else:
                            face.recognition = None
                    else:
                        face.recognition = None

                    recognized_faces.append(face)

                # Cleanup old tracks
                self._cleanup_old_tracks()

                # Update performance stats
                processing_time = time.time() - start_time
                self.stats['processing_times'].append(processing_time)

                # Cache results
                self._cache_results(recognized_faces)

                face_logger.debug(f"🎯 Processed {len(recognized_faces)} faces in {processing_time:.3f}s")

                return recognized_faces

        except Exception as e:
            face_logger.error(f"❌ Recognition error: {e}")
            return []

    def _ultra_fast_tracking(self, faces: List, image: np.ndarray) -> List[Dict]:
        """Ultra-fast tracking with minimal overhead"""
        tracked_faces = []
        current_time = time.time()

        for face in faces:
            bbox = face.bbox
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

            # Find closest existing track
            best_track_id = None
            min_distance = float('inf')

            for track_id, track_data in self.active_tracks.items():
                if current_time - track_data['last_seen'] > 2.0:  # Skip old tracks
                    continue

                track_center = track_data['center']
                distance = np.sqrt((center[0] - track_center[0])**2 + (center[1] - track_center[1])**2)

                if distance < self.tracking_config['max_distance'] and distance < min_distance:
                    min_distance = distance
                    best_track_id = track_id

            # Update or create track
            if best_track_id is not None:
                # Update existing track
                track_data = self.active_tracks[best_track_id]
                track_data['center'] = center
                track_data['bbox'] = bbox
                track_data['last_seen'] = current_time
                track_data['frames_tracked'] += 1
                track_data['face'] = face

                is_stable = track_data['frames_tracked'] >= self.tracking_config['min_stable_frames']

            else:
                # Create new track
                track_id = self.next_track_id
                self.next_track_id += 1

                self.active_tracks[track_id] = {
                    'id': track_id,
                    'center': center,
                    'bbox': bbox,
                    'last_seen': current_time,
                    'frames_tracked': 1,
                    'face': face,
                    'last_recognition': 0,
                    'recognized_student': None
                }

                is_stable = False

            track_data = self.active_tracks.get(best_track_id or track_id)
            if track_data:
                tracked_faces.append({
                    'track_id': track_data['id'],
                    'bbox': bbox,
                    'face': face,
                    'is_stable': is_stable,
                    'frames_tracked': track_data['frames_tracked'],
                    'last_recognition': track_data['last_recognition'],
                    'recognized_student': track_data.get('recognized_student')
                })

        return tracked_faces

    def _fast_recognition(self, tracked_faces: List[Dict]) -> List[Dict]:
        """Fast recognition with caching"""
        recognized_faces = []
        current_time = time.time()

        for track_data in tracked_faces:
            # Only recognize stable tracks and limit frequency
            if (track_data['is_stable'] and
                current_time - track_data['last_recognition'] > 1.0):  # Recognize every 1 second max

                face = track_data['face']

                # Get embedding
                embedding = face.embedding
                if embedding is not None:
                    # Find student
                    student = self.student_service.find_student_by_embedding(embedding)

                    if student:
                        # Update track with recognition
                        track_id = track_data['track_id']
                        if track_id in self.active_tracks:
                            self.active_tracks[track_id]['last_recognition'] = current_time
                            self.active_tracks[track_id]['recognized_student'] = student

                        # Create recognition result
                        recognized_faces.append({
                            'track_id': track_id,
                            'bbox': track_data['bbox'],
                            'student_id': student['student_id'],
                            'name': student['name'],
                            'student_code': student['student_code'],
                            'similarity': student['similarity'],
                            'class_name': student.get('class_name', ''),
                            'detection_time': current_time
                        })

                        self.stats['total_recognitions'] += 1

            elif track_data['recognized_student']:
                # Use cached recognition
                student = track_data['recognized_student']
                recognized_faces.append({
                    'track_id': track_data['track_id'],
                    'bbox': track_data['bbox'],
                    'student_id': student['student_id'],
                    'name': student['name'],
                    'student_code': student['student_code'],
                    'similarity': student['similarity'],
                    'class_name': student.get('class_name', ''),
                    'detection_time': track_data['last_recognition'],
                    'cached': True
                })

        return recognized_faces

    def _filter_quality(self, faces: List, image: np.ndarray) -> List:
        """Fast quality filtering"""
        if self.skip_quality_check:
            return faces

        filtered_faces = []
        for face in faces:
            # Simple quality checks
            if (hasattr(face, 'det_score') and face.det_score > config.DETECTION_THRESHOLD):
                filtered_faces.append(face)

        return filtered_faces

    def _cleanup_old_tracks(self):
        """Cleanup old tracks for performance"""
        current_time = time.time()
        old_tracks = []

        for track_id, track_data in self.active_tracks.items():
            if current_time - track_data['last_seen'] > 3.0:  # Remove tracks older than 3 seconds
                old_tracks.append(track_id)

        for track_id in old_tracks:
            del self.active_tracks[track_id]

    def _cache_results(self, results: List[Dict]):
        """Cache results for frame skipping"""
        self.cached_results = results
        self.cache_time = time.time()

    def _get_cached_results(self) -> List[Dict]:
        """Get cached results for skipped frames"""
        if hasattr(self, 'cached_results') and time.time() - self.cache_time < 0.5:
            return self.cached_results
        return []

    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if self.stats['processing_times']:
            avg_time = sum(self.stats['processing_times']) / len(self.stats['processing_times'])
            fps = 1.0 / avg_time if avg_time > 0 else 0
        else:
            avg_time = 0
            fps = 0

        return {
            'average_processing_time': avg_time,
            'estimated_fps': fps,
            'total_detections': self.stats['total_detections'],
            'total_recognitions': self.stats['total_recognitions'],
            'active_tracks': len(self.active_tracks),
            'students_loaded': len(self.student_service.student_embeddings)
        }

    def load_student_embeddings(self) -> Dict[str, any]:
        """Load student embeddings"""
        return self.student_service.load_embeddings_from_backend()