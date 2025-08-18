"""
Face Recognition Service - COMPLETE WORKING VERSION
Both process_frame() and recognize_faces_optimized() work properly
"""
import cv2
import numpy as np
import time
import threading
from collections import deque
from typing import Dict, List, Optional
from datetime import datetime
import queue
import requests
import json
import base64

try:
    INSIGHTFACE_AVAILABLE = True
    from insightface.app import FaceAnalysis
except ImportError:
    INSIGHTFACE_AVAILABLE = False

from services.backend_api import BackendAPI
from config.config import config
from utils.logger import face_logger


class StudentEmbeddingService:
    """Direct API embedding service"""

    def __init__(self, backend_api: BackendAPI):
        self.backend_api = backend_api
        self.student_embeddings: Dict[str, Dict] = {}
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.last_loaded = None
        self.lock = threading.RLock()

    def load_embeddings_from_backend(self) -> Dict[str, any]:
        """Load student embeddings từ /python/embeddings endpoint"""
        try:
            with self.lock:
                face_logger.info("🔄 Loading student embeddings from backend API...")

                # API URL
                api_url = f"{config.BACKEND_URL.rstrip('/')}/python/embeddings"
                face_logger.info(f"📡 API URL: {api_url}")

                headers = {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                    'User-Agent': 'FaceRecognitionService/1.0'
                }

                try:
                    response = requests.get(api_url, headers=headers, timeout=30)
                    face_logger.info(f"📡 Response: {response.status_code}")

                    if response.status_code != 200:
                        face_logger.error(f"❌ API failed: {response.status_code}")
                        return {'success': False, 'message': f'API failed: {response.status_code}', 'count': 0}

                    # Check content type
                    content_type = response.headers.get('content-type', '')
                    if 'application/json' not in content_type:
                        face_logger.error(f"❌ Wrong content type: {content_type}")
                        return {'success': False, 'message': f'Wrong content type: {content_type}', 'count': 0}

                    # Parse JSON
                    embeddings_data = response.json()
                    face_logger.info(f"✅ Parsed JSON: {len(embeddings_data)} students")

                except requests.exceptions.RequestException as e:
                    face_logger.error(f"❌ Request failed: {e}")
                    return {'success': False, 'message': f'Request failed: {e}', 'count': 0}
                except json.JSONDecodeError as e:
                    face_logger.error(f"❌ JSON decode error: {e}")
                    return {'success': False, 'message': f'JSON error: {e}', 'count': 0}

                # Validate data
                if not isinstance(embeddings_data, list):
                    face_logger.error(f"❌ Expected list, got {type(embeddings_data)}")
                    return {'success': False, 'message': 'Invalid data format', 'count': 0}

                # Clear existing data
                self.student_embeddings.clear()
                self.embedding_cache.clear()

                # Process embeddings
                successful_count = 0
                skipped_count = 0

                for i, student_data in enumerate(embeddings_data):
                    try:
                        if not isinstance(student_data, dict):
                            skipped_count += 1
                            continue

                        # Extract data
                        student_id = student_data.get('studentId')
                        name = student_data.get('name', 'Unknown')
                        embedding_str = student_data.get('embedding')

                        if not student_id or not embedding_str:
                            skipped_count += 1
                            continue

                        # Parse embedding
                        embedding = self._parse_embedding(embedding_str, student_id)
                        if embedding is None:
                            skipped_count += 1
                            continue

                        # Store student data
                        self.student_embeddings[student_id] = {
                            'name': name,
                            'student_code': student_id,
                            'embedding': embedding,
                            'class_name': '',
                            'email': '',
                            'phone': ''
                        }

                        # Cache for faster lookup
                        cache_key = f"student_{student_id}"
                        self.embedding_cache[cache_key] = embedding

                        successful_count += 1

                    except Exception as e:
                        face_logger.error(f"❌ Error processing student {i}: {e}")
                        skipped_count += 1

                self.last_loaded = datetime.now()
                face_logger.info(f"✅ Load completed: {successful_count} loaded, {skipped_count} skipped")

                return {
                    'success': True,
                    'count': successful_count,
                    'skipped': skipped_count,
                    'message': f'Loaded {successful_count} students'
                }

        except Exception as e:
            face_logger.error(f"❌ Load error: {e}")
            return {'success': False, 'message': str(e), 'count': 0}

    def _parse_embedding(self, embedding_str: str, student_id: str) -> Optional[np.ndarray]:
        """Parse embedding từ various formats"""
        try:
            embedding = None

            if isinstance(embedding_str, str):
                if embedding_str.startswith("[") and embedding_str.endswith("]"):
                    # JSON array format
                    embedding_list = json.loads(embedding_str)
                    embedding = np.array(embedding_list, dtype=np.float32)
                elif "," in embedding_str and not embedding_str.startswith("["):
                    # Comma-separated format
                    embedding = np.fromstring(embedding_str, sep=',', dtype=np.float32)
                else:
                    # Base64 format
                    try:
                        embedding_bytes = base64.b64decode(embedding_str)
                        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    except Exception:
                        face_logger.error(f"❌ {student_id}: Base64 decode failed")
                        return None
            elif isinstance(embedding_str, list):
                embedding = np.array(embedding_str, dtype=np.float32)
            else:
                face_logger.warning(f"❌ {student_id}: Unknown format: {type(embedding_str)}")
                return None

            # Validate dimension
            if len(embedding) != 512:
                face_logger.warning(f"❌ {student_id}: Wrong dimension {len(embedding)}")
                return None

            # Check for invalid values
            if not np.isfinite(embedding).all():
                face_logger.warning(f"❌ {student_id}: Contains NaN/Inf")
                return None

            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                return embedding / norm
            else:
                face_logger.warning(f"❌ {student_id}: Zero norm")
                return None

        except Exception as e:
            face_logger.error(f"❌ {student_id}: Parse error: {e}")
            return None

    def find_student_by_embedding(self, face_embedding: np.ndarray, threshold: float = None) -> Optional[Dict]:
        """Find student by embedding"""
        if not self.student_embeddings:
            return None

        threshold = threshold or config.RECOGNITION_THRESHOLD

        try:
            # Validate input
            if face_embedding.shape[0] != 512:
                face_logger.error(f"❌ Input wrong dimension: {face_embedding.shape[0]}")
                return None

            # Normalize input
            norm = np.linalg.norm(face_embedding)
            if norm > 0:
                face_embedding = face_embedding / norm
            else:
                return None

            # Find best match
            best_match = None
            best_similarity = threshold

            for student_id, student_data in self.student_embeddings.items():
                stored_embedding = student_data['embedding']

                # Skip dimension mismatches
                if stored_embedding.shape != face_embedding.shape:
                    continue

                try:
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

                except Exception as e:
                    face_logger.error(f"❌ Similarity error for {student_id}: {e}")
                    continue

            if best_match:
                face_logger.info(f"✅ RECOGNIZED: {best_match['name']} ({best_match['similarity']:.3f})")

            return best_match

        except Exception as e:
            face_logger.error(f"❌ Recognition error: {e}")
            return None

    def get_statistics(self) -> Dict:
        """Get statistics"""
        return {
            'total_students': len(self.student_embeddings),
            'cache_size': len(self.embedding_cache),
            'last_loaded': self.last_loaded.isoformat() if self.last_loaded else None,
            'data_source': 'direct_api'
        }


class FaceRecognitionService:
    """COMPLETE Face recognition service with both methods working"""

    def __init__(self, backend_api):
        self.backend_api = backend_api
        self.face_app = None
        self.is_initialized = False

        # Student embedding service
        self.student_service = StudentEmbeddingService(backend_api)

        # Tracking system for recognize_faces_optimized
        self.active_tracks = {}
        self.next_track_id = 1
        self.track_history = deque(maxlen=100)

        # ENHANCED: Recognition history for temporal smoothing
        self.recognition_history = deque(maxlen=20)  # Store recent recognitions

        # Configuration - ENHANCED for pose robustness
        self.tracking_config = {
            'track_thresh': config.TRACK_MATCH_THRESHOLD * 0.9,  # Lower for pose variations
            'match_thresh': 0.4,  # More lenient matching
            'stable_thresh': config.STABLE_TRACK_THRESHOLD * 0.85,  # Easier to become stable
            'min_track_length': max(config.MIN_TRACK_LENGTH - 1, 1),  # Shorter minimum track
            'min_stable_frames': 3,  # Faster recognition
            'track_buffer': config.TRACK_BUFFER_SIZE,
            'max_distance': 80,  # Allow more movement
            'quality_thresh': 0.3,  # Accept lower quality for pose variations
            'pose_tolerance': {
                'max_yaw': 60,    # Allow larger head turns
                'max_pitch': 45,  # Allow head up/down
                'max_roll': 30    # Allow head tilt
            }
        }

        # Performance tracking
        self.stats = {
            'total_detections': 0,
            'total_recognitions': 0,
            'processing_times': deque(maxlen=30)
        }

        # Thread safety
        self.lock = threading.RLock()

        # Optimization flags
        self.skip_quality_check = not config.ENABLE_QUALITY_FILTER
        self.frame_skip_counter = 0
        self.last_recognition_time = 0
        self.recognition_interval = 0.1

        # Cache for results
        self.cached_results = []
        self.cache_time = 0

        # Load embeddings
        face_logger.info("🚀 Initializing FaceRecognitionService...")
        try:
            result = self.student_service.load_embeddings_from_backend()
            if result and result.get('success'):
                face_logger.info(f"✅ Embeddings loaded: {result.get('count')} students")
            else:
                face_logger.warning(f"⚠️ Load failed: {result.get('message') if result else 'Unknown'}")
        except Exception as e:
            face_logger.error(f"❌ Load error: {e}")

        # Initialize InsightFace
        if INSIGHTFACE_AVAILABLE:
            self.initialize_face_recognition()
        else:
            face_logger.error("❌ InsightFace not available")

    def _assess_face_quality(self, face) -> Dict:
        """Assess face quality for robust tracking"""
        quality_score = 1.0
        quality_info = {
            'overall_score': 1.0,
            'detection_score': 1.0,
            'pose_score': 1.0,
            'size_score': 1.0,
            'blur_score': 1.0,
            'pose_angles': {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0},
            'is_good_quality': True
        }

        try:
            # Detection confidence
            if hasattr(face, 'det_score'):
                quality_info['detection_score'] = float(face.det_score)
                quality_score *= face.det_score

            # Face size assessment
            if hasattr(face, 'bbox'):
                bbox = face.bbox
                face_width = bbox[2] - bbox[0]
                face_height = bbox[3] - bbox[1]
                face_area = face_width * face_height

                # Prefer faces with good size (not too small/large)
                ideal_area = 10000  # Adjust based on your resolution
                size_ratio = min(face_area / ideal_area, ideal_area / face_area)
                quality_info['size_score'] = min(size_ratio, 1.0)
                quality_score *= quality_info['size_score']

            # Pose estimation (key improvement!)
            if hasattr(face, 'pose'):
                pose = face.pose
                if pose is not None:
                    # Extract yaw, pitch, roll angles
                    yaw = abs(pose[0]) if len(pose) > 0 else 0
                    pitch = abs(pose[1]) if len(pose) > 1 else 0
                    roll = abs(pose[2]) if len(pose) > 2 else 0

                    quality_info['pose_angles'] = {
                        'yaw': float(yaw),
                        'pitch': float(pitch),
                        'roll': float(roll)
                    }

                    # CRITICAL: Pose score - penalize extreme angles
                    max_angle = max(yaw, pitch, roll)
                    if max_angle < 15:  # Very good pose
                        pose_score = 1.0
                    elif max_angle < 30:  # Good pose
                        pose_score = 0.8
                    elif max_angle < 45:  # Acceptable pose
                        pose_score = 0.6
                    else:  # Poor pose
                        pose_score = 0.3

                    quality_info['pose_score'] = pose_score
                    quality_score *= pose_score

            # Landmark quality (if available)
            if hasattr(face, 'landmark_2d_106') or hasattr(face, 'kps'):
                # Use landmarks to assess face quality
                landmarks = face.landmark_2d_106 if hasattr(face, 'landmark_2d_106') else face.kps
                if landmarks is not None:
                    # Simple landmark spread assessment
                    landmark_spread = np.std(landmarks) if len(landmarks) > 0 else 0
                    landmark_score = min(landmark_spread / 50.0, 1.0)  # Normalize
                    quality_score *= landmark_score

            # Overall quality assessment
            quality_info['overall_score'] = quality_score
            quality_info['is_good_quality'] = quality_score > 0.4  # Threshold for good quality

            face_logger.debug(f"Face quality: {quality_score:.3f} (pose: {quality_info['pose_score']:.3f})")

        except Exception as e:
            face_logger.error(f"❌ Quality assessment error: {e}")

        return quality_info

    def _calculate_pose_similarity(self, face1_quality, face2_quality) -> float:
        """Calculate pose similarity between two faces"""
        try:
            angles1 = face1_quality['pose_angles']
            angles2 = face2_quality['pose_angles']

            # Calculate angle differences
            yaw_diff = abs(angles1['yaw'] - angles2['yaw'])
            pitch_diff = abs(angles1['pitch'] - angles2['pitch'])
            roll_diff = abs(angles1['roll'] - angles2['roll'])

            # Convert to similarity (0-1)
            max_diff = max(yaw_diff, pitch_diff, roll_diff)
            if max_diff < 10:
                return 1.0
            elif max_diff < 20:
                return 0.8
            elif max_diff < 30:
                return 0.6
            else:
                return 0.3

        except Exception:
            return 0.5  # Default similarity

    def _temporal_smooth_recognition(self, current_result: Dict, history: List[Dict]) -> Dict:
        """Apply temporal smoothing to reduce recognition flickering"""
        try:
            if not history or len(history) < 2:
                return current_result

            # Count votes for each student in recent history
            student_votes = {}
            total_weight = 0

            for i, hist_result in enumerate(history[-5:]):  # Last 5 frames
                if hist_result.get('student_id'):
                    student_id = hist_result['student_id']
                    similarity = hist_result.get('similarity', 0)
                    weight = (i + 1) * similarity  # Recent frames have more weight

                    if student_id not in student_votes:
                        student_votes[student_id] = {'weight': 0, 'name': hist_result.get('name', '')}

                    student_votes[student_id]['weight'] += weight
                    total_weight += weight

            # Check if current recognition conflicts with history
            current_student = current_result.get('student_id')
            current_similarity = current_result.get('similarity', 0)

            if current_student and current_student in student_votes:
                # Consistent with history - boost confidence
                history_weight = student_votes[current_student]['weight'] / total_weight if total_weight > 0 else 0
                if history_weight > 0.6:  # Strong historical support
                    current_result['similarity'] = min(current_result['similarity'] * 1.1, 1.0)
                    current_result['temporal_confidence'] = 'high'
                return current_result

            elif student_votes and total_weight > 0:
                # Check if history strongly suggests a different student
                best_historical = max(student_votes.items(), key=lambda x: x[1]['weight'])
                best_student_id, best_data = best_historical
                best_weight = best_data['weight'] / total_weight

                if best_weight > 0.7 and current_similarity < 0.6:
                    # Strong historical evidence, weak current evidence
                    face_logger.debug(f"🔄 Temporal smoothing: using historical {best_data['name']} (weight: {best_weight:.3f})")
                    return {
                        'student_id': best_student_id,
                        'name': best_data['name'],
                        'similarity': best_weight * 0.8,  # Reduced confidence for historical match
                        'temporal_confidence': 'historical',
                        'class_name': current_result.get('class_name', '')
                    }

            # No strong historical evidence, use current
            current_result['temporal_confidence'] = 'current'
            return current_result

        except Exception as e:
            face_logger.error(f"❌ Temporal smoothing error: {e}")
            return current_result

    def initialize_face_recognition(self):
        """Initialize InsightFace with FULL features for robust tracking"""
        try:
            face_logger.info("🚀 Initializing ENHANCED InsightFace...")

            # Use buffalo_l for better pose estimation
            model_name = 'buffalo_l'  # Better model for pose robustness
            providers = config.get_insightface_providers()

            # ENHANCED: Enable ALL modules for robust tracking
            self.face_app = FaceAnalysis(
                name=model_name,
                providers=providers,
                allowed_modules=['detection', 'recognition', 'genderage']  # All modules
            )

            # Optimized detection size for pose variation
            det_size = (640, 640)
            self.face_app.prepare(
                ctx_id=0,
                det_size=det_size,
                det_thresh=0.3  # Lower threshold for pose variations
            )

            self.is_initialized = True
            face_logger.info(f"✅ ENHANCED InsightFace ready - {model_name}")
            face_logger.info("📐 Enabled: detection, recognition, pose estimation, age/gender")

        except Exception as e:
            face_logger.error(f"❌ Enhanced InsightFace init failed: {e}")
            # Fallback to basic setup
            try:
                face_logger.info("🔄 Trying basic InsightFace setup...")
                self.face_app = FaceAnalysis(name='buffalo_s')
                self.face_app.prepare(ctx_id=0, det_size=(320, 320), det_thresh=0.3)
                self.is_initialized = True
                face_logger.info("✅ Basic InsightFace ready")
            except Exception:
                self.face_app = None
                self.is_initialized = False

    def process_frame(self, frame):
        """METHOD 1: ENHANCED Process frame with pose-aware recognition"""
        if not self.is_initialized or frame is None:
            return frame

        try:
            # Validate frame
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                return frame

            # Resize if too large
            height, width = frame.shape[:2]
            if width > 1920 or height > 1080:
                scale = min(1920/width, 1080/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))

            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces with ENHANCED features
            try:
                faces = self.face_app.get(rgb_frame)
                self.stats['total_detections'] += len(faces)
                face_logger.debug(f"🔍 Detected {len(faces)} faces")
            except Exception as e:
                face_logger.error(f"❌ Detection failed: {e}")
                return frame

            # ENHANCED: Process each face with quality assessment
            for face in faces:
                try:
                    box = face.bbox.astype(int)

                    # CRITICAL: Assess face quality and pose
                    face_quality = self._assess_face_quality(face)

                    # Choose box color based on quality
                    if face_quality['is_good_quality']:
                        box_color = (0, 255, 0)  # Green for good quality
                        thickness = 2
                    else:
                        box_color = (0, 165, 255)  # Orange for poor quality
                        thickness = 1

                    # Draw bounding box with quality indication
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), box_color, thickness)

                    # Enhanced recognition with pose tolerance
                    if hasattr(face, "embedding") and face.embedding is not None:
                        if face.embedding.shape[0] == 512:
                            # ENHANCED: Use pose-aware recognition
                            student = self._pose_aware_recognition(face, face_quality)

                            if student:
                                # Enhanced display with quality info
                                confidence_text = f"{student['name']}"
                                similarity_text = f"({student['similarity']:.2f})"

                                # Add pose info for debugging
                                pose_angles = face_quality['pose_angles']
                                if max(pose_angles.values()) > 20:  # Show pose for large angles
                                    pose_text = f"Y:{pose_angles['yaw']:.0f}°"
                                    similarity_text += f" {pose_text}"

                                # Draw name with enhanced info
                                cv2.putText(frame, confidence_text, (box[0], box[1] - 25),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
                                cv2.putText(frame, similarity_text, (box[0], box[1] - 5),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

                                self.stats['total_recognitions'] += 1
                            else:
                                # Enhanced unknown display
                                quality_text = f"Unknown (Q:{face_quality['overall_score']:.2f})"
                                cv2.putText(frame, quality_text, (box[0], box[1] - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        else:
                            cv2.putText(frame, f"BadEmb({face.embedding.shape[0]})", (box[0], box[1] - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    else:
                        # Show quality info for faces without embeddings
                        quality_text = f"NoEmb (Q:{face_quality['overall_score']:.2f})"
                        cv2.putText(frame, quality_text, (box[0], box[1] - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)

                except Exception as e:
                    face_logger.error(f"❌ Face processing error: {e}")
                    continue

        except Exception as e:
            face_logger.error(f"❌ Frame processing error: {e}")

        return frame

    def _pose_aware_recognition(self, face, face_quality) -> Optional[Dict]:
        """ENHANCED recognition with pose awareness"""
        try:
            if face.embedding.shape[0] != 512:
                return None

            # Get normalized embedding
            face_embedding = face.embedding / np.linalg.norm(face.embedding)

            # ENHANCED: Adaptive threshold based on pose quality
            base_threshold = config.RECOGNITION_THRESHOLD
            pose_score = face_quality['pose_score']

            # Lower threshold for good poses, higher tolerance for poor poses
            if pose_score > 0.8:
                adaptive_threshold = base_threshold  # Standard threshold
            elif pose_score > 0.6:
                adaptive_threshold = base_threshold * 0.9  # Slightly lower
            else:
                adaptive_threshold = base_threshold * 0.8  # Much lower for poor poses

            best_match = None
            best_similarity = adaptive_threshold

            for student_id, student_data in self.student_service.student_embeddings.items():
                stored_embedding = student_data['embedding']

                if stored_embedding.shape != face_embedding.shape:
                    continue

                try:
                    # Basic similarity
                    similarity = float(np.dot(face_embedding, stored_embedding))

                    # ENHANCED: Apply pose-based confidence boost
                    # Give bonus for good poses, maintain similarity for poor poses
                    if pose_score > 0.8:
                        adjusted_similarity = similarity * 1.02  # Small boost for good poses
                    elif pose_score > 0.6:
                        adjusted_similarity = similarity  # No change
                    else:
                        adjusted_similarity = similarity * 0.98  # Small penalty for poor poses

                    if adjusted_similarity > best_similarity:
                        best_similarity = adjusted_similarity
                        best_match = {
                            'student_id': student_id,
                            'name': student_data['name'],
                            'student_code': student_data['student_code'],
                            'similarity': similarity,  # Keep original similarity for display
                            'adjusted_similarity': adjusted_similarity,
                            'pose_score': pose_score,
                            'class_name': student_data['class_name']
                        }

                except Exception as e:
                    face_logger.error(f"❌ Similarity error for {student_id}: {e}")
                    continue

            # ENHANCED: Apply temporal smoothing
            if best_match:
                best_match = self._temporal_smooth_recognition(best_match, list(self.recognition_history))

                # Add to recognition history
                self.recognition_history.append({
                    'student_id': best_match['student_id'],
                    'name': best_match['name'],
                    'similarity': best_match['similarity'],
                    'timestamp': time.time()
                })

                face_logger.info(f"✅ POSE-AWARE RECOGNITION: {best_match['name']} "
                               f"(sim: {best_match['similarity']:.3f}, pose: {pose_score:.3f}, "
                               f"conf: {best_match.get('temporal_confidence', 'current')})")

            return best_match

        except Exception as e:
            face_logger.error(f"❌ Pose-aware recognition error: {e}")
            return None

    def recognize_faces_optimized(self, image: np.ndarray) -> List[Dict]:
        """METHOD 2: ENHANCED Return face data with pose awareness (for stream_widget)"""
        if not self.is_initialized:
            return []

        # Frame skipping for performance
        self.frame_skip_counter += 1
        if self.frame_skip_counter % config.FRAME_PROCESSING_INTERVAL != 0:
            return self._get_cached_results()

        # Throttle recognition
        current_time = time.time()
        if current_time - self.last_recognition_time < self.recognition_interval:
            return self._get_cached_results()

        self.last_recognition_time = current_time
        start_time = current_time

        try:
            with self.lock:
                # Backup original
                original_image = image.copy()

                # Resize for processing
                if image.shape[1] > config.PROCESSING_WIDTH:
                    scale_x = image.shape[1] / config.PROCESSING_WIDTH
                    scale_y = image.shape[0] / config.PROCESSING_HEIGHT
                    image = cv2.resize(image, (config.PROCESSING_WIDTH, config.PROCESSING_HEIGHT))
                else:
                    scale_x = scale_y = 1.0

                # Convert to RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Get faces with enhanced detection
                try:
                    faces = self.face_app.get(rgb_image)
                    self.stats['total_detections'] += len(faces)
                except Exception as e:
                    face_logger.error(f"❌ Detection error: {e}")
                    return []

                # ENHANCED: Sort by quality score instead of just detection score
                if len(faces) > config.MAX_FACES_PER_FRAME:
                    # Assess quality for all faces first
                    face_qualities = []
                    for face in faces:
                        quality = self._assess_face_quality(face)
                        face_qualities.append((face, quality['overall_score']))

                    # Sort by quality score
                    face_qualities.sort(key=lambda x: x[1], reverse=True)
                    faces = [fq[0] for fq in face_qualities[:config.MAX_FACES_PER_FRAME]]

                # Scale bounding boxes back
                for face in faces:
                    if scale_x != 1.0 or scale_y != 1.0:
                        bbox = face.bbox
                        face.bbox = np.array([
                            bbox[0] * scale_x,
                            bbox[1] * scale_y,
                            bbox[2] * scale_x,
                            bbox[3] * scale_y
                        ])

                # ENHANCED: Convert to result format with quality info
                results = []
                for face in faces:
                    try:
                        # Assess face quality
                        face_quality = self._assess_face_quality(face)

                        face_result = {
                            'bbox': face.bbox.tolist(),
                            'det_score': float(face.det_score) if hasattr(face, 'det_score') else 1.0,
                            'quality_score': face_quality['overall_score'],
                            'pose_score': face_quality['pose_score'],
                            'pose_angles': face_quality['pose_angles'],
                            'is_good_quality': face_quality['is_good_quality'],
                            'name': 'Unknown',
                            'student_id': None,
                            'similarity': 0.0,
                            'class_name': ''
                        }

                        # ENHANCED Recognition with pose awareness
                        if hasattr(face, "embedding") and face.embedding is not None:
                            if face.embedding.shape[0] == 512:
                                student = self._pose_aware_recognition(face, face_quality)
                                if student:
                                    face_result.update({
                                        'name': student['name'],
                                        'student_id': student['student_id'],
                                        'similarity': student['similarity'],
                                        'adjusted_similarity': student['adjusted_similarity'],
                                        'class_name': student['class_name']
                                    })
                                    self.stats['total_recognitions'] += 1

                        results.append(face_result)

                    except Exception as e:
                        face_logger.error(f"❌ Face result error: {e}")
                        continue

                # Update performance
                processing_time = time.time() - start_time
                self.stats['processing_times'].append(processing_time)

                # Cache results
                self._cache_results(results)

                face_logger.debug(f"🎯 ENHANCED recognition: {len(results)} faces in {processing_time:.3f}s")
                return results

        except Exception as e:
            face_logger.error(f"❌ Enhanced recognition error: {e}")
            return []

    def _cache_results(self, results: List[Dict]):
        """Cache results for frame skipping"""
        self.cached_results = results
        self.cache_time = time.time()

    def _get_cached_results(self) -> List[Dict]:
        """Get cached results"""
        if hasattr(self, 'cached_results') and time.time() - self.cache_time < 0.5:
            return self.cached_results
        return []

    def get_performance_stats(self) -> Dict:
        """Get ENHANCED performance stats with pose information"""
        avg_time = 0
        if self.stats['processing_times']:
            avg_time = sum(self.stats['processing_times']) / len(self.stats['processing_times'])

        # Calculate pose statistics from recent history
        pose_stats = {'good_poses': 0, 'poor_poses': 0, 'avg_pose_score': 0.0}
        if hasattr(self, 'recognition_history') and self.recognition_history:
            recent_poses = []
            for hist in list(self.recognition_history)[-10:]:  # Last 10 recognitions
                if 'pose_score' in hist:
                    recent_poses.append(hist['pose_score'])

            if recent_poses:
                pose_stats['avg_pose_score'] = sum(recent_poses) / len(recent_poses)
                pose_stats['good_poses'] = sum(1 for p in recent_poses if p > 0.7)
                pose_stats['poor_poses'] = sum(1 for p in recent_poses if p < 0.5)

        return {
            'average_processing_time': avg_time,
            'estimated_fps': 1.0 / avg_time if avg_time > 0 else 0,
            'total_detections': self.stats['total_detections'],
            'total_recognitions': self.stats['total_recognitions'],
            'students_loaded': len(self.student_service.student_embeddings),
            'is_initialized': self.is_initialized,
            'pose_tolerance_enabled': True,
            'pose_statistics': pose_stats,
            'recognition_history_size': len(getattr(self, 'recognition_history', [])),
            'enhanced_features': ['pose_estimation', 'quality_assessment', 'temporal_smoothing']
        }

    def get_statistics(self) -> Dict:
        """Get statistics for UI"""
        stats = self.student_service.get_statistics()
        stats.update({
            'recognition_threshold': config.RECOGNITION_THRESHOLD,
            'is_initialized': self.is_initialized
        })
        return stats

    def load_student_embeddings(self) -> Dict[str, any]:
        """Load/refresh embeddings"""
        try:
            return self.student_service.load_embeddings_from_backend()
        except Exception as e:
            face_logger.error(f"❌ Load error: {e}")
            return {'success': False, 'message': str(e), 'count': 0}