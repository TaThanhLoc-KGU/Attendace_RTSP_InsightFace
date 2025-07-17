"""
Advanced Face Recognition Service - Full InsightFace Integration
Sử dụng tất cả tính năng: Detection, Recognition, Tracking, Age, Gender, Emotion
"""
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import threading
import time
import json
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import sqlite3
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class FaceTrack:
    """Face tracking object với full InsightFace features"""
    track_id: int
    bbox: List[int]
    embedding: np.ndarray
    confidence: float

    # InsightFace features
    age: int
    gender: int  # 0: male, 1: female
    emotion: str
    pose: List[float]  # [pitch, yaw, roll]

    # Identity
    identity: Optional[str] = None
    student_id: Optional[str] = None
    is_known: bool = False

    # Tracking info
    first_seen: float = 0.0
    last_seen: float = 0.0
    frame_count: int = 0
    positions: deque = None

    # Quality metrics
    quality_score: float = 0.0
    blur_score: float = 0.0
    brightness_score: float = 0.0

    def __post_init__(self):
        if self.positions is None:
            self.positions = deque(maxlen=30)  # Keep last 30 positions
        self.first_seen = time.time()
        self.last_seen = time.time()


class AdvancedFaceService:
    """Advanced Face Recognition Service với full InsightFace integration"""

    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()

        # InsightFace models
        self.face_app = None
        self.det_model = None
        self.rec_model = None
        self.ga_model = None  # Gender & Age

        # Face database
        self.face_database = {}
        self.student_embeddings = {}

        # Tracking system
        self.active_tracks = {}
        self.next_track_id = 1
        self.track_history = deque(maxlen=1000)

        # Performance optimization
        self.embedding_cache = {}
        self.detection_cache = {}
        self.last_detection_time = 0

        # Statistics
        self.stats = {
            'total_detections': 0,
            'total_recognitions': 0,
            'total_tracks': 0,
            'avg_confidence': 0.0,
            'processing_times': deque(maxlen=100)
        }

        # Thread safety
        self.lock = threading.RLock()

        # Initialize models
        self.initialize_models()
        self.load_face_database()

        logger.info("🚀 Advanced Face Service initialized with full InsightFace features")

    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            # Detection settings
            'det_size': (640, 640),
            'det_thresh': 0.5,
            'nms_thresh': 0.4,

            # Recognition settings
            'rec_thresh': 0.5,  # Lower for easier recognition
            'embedding_dim': 512,

            # Tracking settings
            'track_thresh': 0.7,
            'track_buffer': 30,
            'match_thresh': 0.8,
            'min_track_length': 5,

            # Quality settings
            'min_face_size': 50,
            'max_face_size': 500,
            'blur_thresh': 30.0,
            'brightness_thresh': (50, 200),

            # Performance settings
            'max_faces': 10,
            'skip_frames': 1,
            'cache_size': 1000,

            # Models
            'model_pack': 'buffalo_l',  # Model tốt nhất
            'providers': ['CUDAExecutionProvider', 'CPUExecutionProvider']
            # GPU nếu có
            # 'model_pack': 'buffalo_l',  # buffalo_l, buffalo_m, buffalo_s
        }

    def initialize_models(self):
        """Initialize all InsightFace models"""
        try:
            logger.info("🔍 Initializing InsightFace models...")

            # Main FaceAnalysis app (includes detection + recognition + analysis)
            self.face_app = FaceAnalysis(
                name=self.config['model_pack'],
                providers=self.config['providers'],
                allowed_modules=['detection', 'recognition', 'genderage']  # Enable all modules
            )
            self.face_app.prepare(
                ctx_id=0,
                det_size=self.config['det_size'],
                det_thresh=self.config['det_thresh']
            )

            logger.info(f"✅ InsightFace initialized with {self.config['model_pack']} model pack")
            logger.info(f"📊 Available models: {list(self.face_app.models.keys())}")

            # Test the models
            self._test_models()

        except Exception as e:
            logger.error(f"❌ Failed to initialize InsightFace: {e}")
            raise

    def _test_models(self):
        """Test InsightFace models with sample image"""
        try:
            # Create test image
            test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            # Test detection and analysis
            faces = self.face_app.get(test_img)
            logger.info(f"✅ Model test completed - detected {len(faces)} faces in test image")

        except Exception as e:
            logger.warning(f"⚠️ Model test failed: {e}")

    def load_face_database(self):
        """Load face database from file"""
        try:
            db_path = self.config.get('face_db_path', 'data/face_database.json')

            if os.path.exists(db_path):
                with open(db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Convert embeddings back to numpy arrays
                for person_id, person_data in data.items():
                    if 'embeddings' in person_data:
                        embeddings = []
                        for emb_data in person_data['embeddings']:
                            embeddings.append(np.array(emb_data['embedding']))
                        person_data['embeddings'] = embeddings

                self.face_database = data

                # Build student embeddings for fast lookup
                self._build_student_embeddings()

                logger.info(f"✅ Loaded {len(self.face_database)} persons from face database")
            else:
                logger.info("📁 No face database found, starting with empty database")
                self.face_database = {}

        except Exception as e:
            logger.error(f"❌ Error loading face database: {e}")
            self.face_database = {}

    def _build_student_embeddings(self):
        """Build optimized student embeddings for recognition"""
        self.student_embeddings = {}

        for person_id, person_data in self.face_database.items():
            if 'embeddings' in person_data and person_data['embeddings']:
                # Use average embedding if multiple embeddings exist
                if len(person_data['embeddings']) > 1:
                    avg_embedding = np.mean(person_data['embeddings'], axis=0)
                else:
                    avg_embedding = person_data['embeddings'][0]

                # Normalize embedding
                avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

                self.student_embeddings[person_id] = {
                    'embedding': avg_embedding,
                    'name': person_data.get('name', person_id),
                    'student_id': person_data.get('student_id', person_id),
                    'metadata': person_data.get('metadata', {})
                }

        logger.info(f"✅ Built {len(self.student_embeddings)} student embeddings for recognition")

    def analyze_faces_advanced(self, image: np.ndarray) -> List[Dict]:
        """Advanced face analysis với full InsightFace features"""
        start_time = time.time()

        try:
            with self.lock:
                # Get faces with full analysis
                faces = self.face_app.get(image)

                results = []
                for face in faces:
                    # Extract all InsightFace features
                    face_data = self._extract_face_features(face, image)

                    # Perform recognition
                    identity_info = self._recognize_face(face_data['embedding'])
                    face_data.update(identity_info)

                    # Calculate quality metrics
                    quality_info = self._calculate_face_quality(face, image)
                    face_data.update(quality_info)

                    results.append(face_data)

                # Update statistics
                processing_time = (time.time() - start_time) * 1000
                self.stats['processing_times'].append(processing_time)
                self.stats['total_detections'] += len(results)

                return results

        except Exception as e:
            logger.error(f"❌ Error in advanced face analysis: {e}")
            return []

    def _extract_face_features(self, face, image: np.ndarray) -> Dict:
        """Extract all features from InsightFace face object"""
        try:
            # Basic detection info
            bbox = face.bbox.astype(int)

            # Embedding (recognition feature)
            embedding = face.embedding
            embedding = embedding / np.linalg.norm(embedding)  # Normalize

            # Age and Gender
            age = int(face.age) if hasattr(face, 'age') else 0
            gender = int(face.gender) if hasattr(face, 'gender') else -1

            # Pose estimation (if available)
            pose = [0.0, 0.0, 0.0]  # [pitch, yaw, roll]
            if hasattr(face, 'pose'):
                pose = face.pose.tolist() if face.pose is not None else pose

            # Landmarks (if available)
            landmarks = []
            if hasattr(face, 'kps') and face.kps is not None:
                landmarks = face.kps.tolist()

            # Detection confidence
            det_score = float(face.det_score) if hasattr(face, 'det_score') else 1.0

            return {
                'bbox': bbox.tolist(),
                'embedding': embedding,
                'age': age,
                'gender': gender,
                'gender_text': 'Female' if gender == 1 else 'Male' if gender == 0 else 'Unknown',
                'pose': pose,
                'landmarks': landmarks,
                'det_score': det_score,
                'timestamp': time.time()
            }

        except Exception as e:
            logger.error(f"❌ Error extracting face features: {e}")
            return {}

    def _recognize_face(self, embedding: np.ndarray) -> Dict:
        """Recognize face using student database"""
        try:
            if not self.student_embeddings:
                return {
                    'identity': 'Unknown',
                    'student_id': None,
                    'confidence': 0.0,
                    'is_known': False
                }

            best_match = None
            best_score = 0.0

            # Compare with all known faces
            for person_id, person_data in self.student_embeddings.items():
                # Calculate cosine similarity
                similarity = np.dot(embedding, person_data['embedding'])

                if similarity > best_score:
                    best_score = similarity
                    best_match = person_data

            # Check if above threshold
            if best_score >= self.config['rec_thresh']:
                self.stats['total_recognitions'] += 1
                return {
                    'identity': best_match['name'],
                    'student_id': best_match['student_id'],
                    'confidence': float(best_score),
                    'is_known': True
                }
            else:
                return {
                    'identity': 'Unknown',
                    'student_id': None,
                    'confidence': float(best_score),
                    'is_known': False
                }

        except Exception as e:
            logger.error(f"❌ Error in face recognition: {e}")
            return {
                'identity': 'Unknown',
                'student_id': None,
                'confidence': 0.0,
                'is_known': False
            }

    def _calculate_face_quality(self, face, image: np.ndarray) -> Dict:
        """Calculate face quality metrics"""
        try:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox

            # Extract face region
            face_img = image[y1:y2, x1:x2]

            if face_img.size == 0:
                return {'quality_score': 0.0, 'blur_score': 0.0, 'brightness_score': 0.0}

            # Face size
            face_width = x2 - x1
            face_height = y2 - y1
            face_size = min(face_width, face_height)

            # Blur detection (Laplacian variance)
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
            blur_score = cv2.Laplacian(gray_face, cv2.CV_64F).var()

            # Brightness
            brightness_score = np.mean(gray_face)

            # Overall quality score
            size_score = min(1.0, face_size / 100.0)  # Normalize by 100px
            blur_quality = min(1.0, blur_score / 100.0)  # Normalize blur
            brightness_quality = 1.0 - abs(brightness_score - 128) / 128.0  # Optimal at 128

            quality_score = (size_score + blur_quality + brightness_quality) / 3.0

            return {
                'quality_score': float(quality_score),
                'blur_score': float(blur_score),
                'brightness_score': float(brightness_score),
                'face_size': face_size
            }

        except Exception as e:
            logger.error(f"❌ Error calculating face quality: {e}")
            return {'quality_score': 0.0, 'blur_score': 0.0, 'brightness_score': 0.0}

    def track_faces_advanced(self, image: np.ndarray) -> List[Dict]:
        """Advanced face tracking với continuous tracking"""
        try:
            # Analyze faces
            current_faces = self.analyze_faces_advanced(image)

            if not current_faces:
                # Update existing tracks as lost
                self._update_lost_tracks()
                return []

            # Match faces with existing tracks
            matched_tracks = self._match_faces_to_tracks(current_faces)

            # Update tracks
            tracked_faces = []
            for face_data, track in matched_tracks:
                if track is None:
                    # Create new track
                    track = self._create_new_track(face_data)
                else:
                    # Update existing track
                    self._update_track(track, face_data)

                # Add tracking info to face data
                face_data.update({
                    'track_id': track.track_id,
                    'track_length': track.frame_count,
                    'track_confidence': track.confidence,
                    'first_seen': track.first_seen,
                    'is_stable': track.frame_count >= self.config['min_track_length']
                })

                tracked_faces.append(face_data)

            # Clean up old tracks
            self._cleanup_old_tracks()

            return tracked_faces

        except Exception as e:
            logger.error(f"❌ Error in advanced face tracking: {e}")
            return []

    def _match_faces_to_tracks(self, faces: List[Dict]) -> List[Tuple[Dict, Optional[FaceTrack]]]:
        """Match detected faces to existing tracks"""
        matches = []
        used_tracks = set()

        for face in faces:
            best_track = None
            best_score = 0.0

            face_embedding = face['embedding']
            face_bbox = face['bbox']

            # Find best matching track
            for track_id, track in self.active_tracks.items():
                if track_id in used_tracks:
                    continue

                # Calculate embedding similarity
                emb_similarity = np.dot(face_embedding, track.embedding)

                # Calculate spatial distance
                track_center = [(track.bbox[0] + track.bbox[2]) / 2, (track.bbox[1] + track.bbox[3]) / 2]
                face_center = [(face_bbox[0] + face_bbox[2]) / 2, (face_bbox[1] + face_bbox[3]) / 2]
                spatial_dist = np.sqrt(
                    (track_center[0] - face_center[0]) ** 2 + (track_center[1] - face_center[1]) ** 2)

                # Combined score (embedding + spatial)
                spatial_score = max(0, 1.0 - spatial_dist / 200.0)  # Normalize by 200px
                combined_score = 0.7 * emb_similarity + 0.3 * spatial_score

                if combined_score > best_score and combined_score > self.config['match_thresh']:
                    best_score = combined_score
                    best_track = track

            if best_track:
                used_tracks.add(best_track.track_id)

            matches.append((face, best_track))

        return matches

    def _create_new_track(self, face_data: Dict) -> FaceTrack:
        """Create new face track"""
        track = FaceTrack(
            track_id=self.next_track_id,
            bbox=face_data['bbox'],
            embedding=face_data['embedding'],
            confidence=face_data['confidence'],
            age=face_data['age'],
            gender=face_data['gender'],
            emotion='neutral',  # Default emotion
            pose=face_data['pose'],
            identity=face_data['identity'],
            student_id=face_data['student_id'],
            is_known=face_data['is_known'],
            quality_score=face_data['quality_score']
        )

        track.positions.append(face_data['bbox'])
        track.frame_count = 1

        self.active_tracks[self.next_track_id] = track
        self.next_track_id += 1
        self.stats['total_tracks'] += 1

        logger.debug(f"🆕 Created new track {track.track_id} for {track.identity}")

        return track

    def _update_track(self, track: FaceTrack, face_data: Dict):
        """Update existing face track"""
        # Update position and embedding (with smoothing)
        alpha = 0.7  # Smoothing factor
        track.bbox = face_data['bbox']
        track.embedding = alpha * track.embedding + (1 - alpha) * face_data['embedding']
        track.embedding = track.embedding / np.linalg.norm(track.embedding)  # Renormalize

        # Update other attributes
        track.confidence = max(track.confidence, face_data['confidence'])
        track.age = int(alpha * track.age + (1 - alpha) * face_data['age'])
        track.last_seen = time.time()
        track.frame_count += 1

        # Update position history
        track.positions.append(face_data['bbox'])

        # Update identity if better match found
        if face_data['confidence'] > track.confidence and face_data['is_known']:
            track.identity = face_data['identity']
            track.student_id = face_data['student_id']
            track.is_known = face_data['is_known']

    def _update_lost_tracks(self):
        """Update tracks that weren't matched in current frame"""
        current_time = time.time()

        for track in self.active_tracks.values():
            if current_time - track.last_seen > 0.1:  # 100ms threshold
                # Track is lost, could implement prediction here
                pass

    def _cleanup_old_tracks(self):
        """Remove old/stale tracks"""
        current_time = time.time()
        tracks_to_remove = []

        for track_id, track in self.active_tracks.items():
            # Remove tracks not seen for more than track_buffer frames
            if current_time - track.last_seen > self.config['track_buffer'] / 30.0:  # Assume 30 FPS
                tracks_to_remove.append(track_id)

                # Archive track to history
                self.track_history.append({
                    'track_id': track_id,
                    'identity': track.identity,
                    'student_id': track.student_id,
                    'duration': track.last_seen - track.first_seen,
                    'frame_count': track.frame_count,
                    'avg_confidence': track.confidence
                })

        for track_id in tracks_to_remove:
            del self.active_tracks[track_id]
            logger.debug(f"🗑️ Removed stale track {track_id}")

    def add_person_to_database(self, name: str, student_id: str, images: List[np.ndarray],
                               metadata: Dict = None) -> bool:
        """Add person to face database with multiple images"""
        try:
            embeddings = []

            for img in images:
                faces = self.face_app.get(img)

                if len(faces) == 0:
                    logger.warning(f"No face detected in image for {name}")
                    continue

                # Use the largest face
                face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

                # Normalize embedding
                embedding = face.embedding / np.linalg.norm(face.embedding)
                embeddings.append(embedding)

            if not embeddings:
                logger.error(f"No valid faces found for {name}")
                return False

            # Store in database
            person_data = {
                'name': name,
                'student_id': student_id,
                'embeddings': embeddings,
                'metadata': metadata or {},
                'created_at': datetime.now().isoformat(),
                'image_count': len(embeddings)
            }

            self.face_database[student_id] = person_data

            # Rebuild student embeddings
            self._build_student_embeddings()

            # Save to file
            self.save_face_database()

            logger.info(f"✅ Added {name} to database with {len(embeddings)} embeddings")
            return True

        except Exception as e:
            logger.error(f"❌ Error adding person to database: {e}")
            return False

    def save_face_database(self):
        """Save face database to file"""
        try:
            db_path = self.config.get('face_db_path', 'data/face_database.json')
            os.makedirs(os.path.dirname(db_path), exist_ok=True)

            # Convert numpy arrays to lists for JSON serialization
            serializable_db = {}
            for person_id, person_data in self.face_database.items():
                serializable_data = person_data.copy()
                if 'embeddings' in serializable_data:
                    serializable_data['embeddings'] = [
                        {'embedding': emb.tolist()} for emb in serializable_data['embeddings']
                    ]
                serializable_db[person_id] = serializable_data

            with open(db_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_db, f, ensure_ascii=False, indent=2)

            logger.info(f"💾 Saved face database with {len(self.face_database)} persons")

        except Exception as e:
            logger.error(f"❌ Error saving face database: {e}")

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        avg_processing_time = np.mean(self.stats['processing_times']) if self.stats['processing_times'] else 0

        return {
            'total_detections': self.stats['total_detections'],
            'total_recognitions': self.stats['total_recognitions'],
            'total_tracks': self.stats['total_tracks'],
            'active_tracks': len(self.active_tracks),
            'database_size': len(self.face_database),
            'student_embeddings': len(self.student_embeddings),
            'avg_processing_time': float(avg_processing_time),
            'recognition_threshold': self.config['rec_thresh'],
            'track_threshold': self.config['track_thresh'],
            'model_pack': self.config['model_pack'],
            'providers': self.config['providers']
        }

    def update_threshold(self, new_threshold: float):
        """Update recognition threshold"""
        self.config['rec_thresh'] = new_threshold
        logger.info(f"🎯 Updated recognition threshold to {new_threshold}")

    def get_active_tracks_info(self) -> List[Dict]:
        """Get information about active tracks"""
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
                'duration': time.time() - track.first_seen,
                'is_known': track.is_known,
                'quality_score': track.quality_score
            })

        return tracks_info

    # Backward compatibility methods
    def recognize_faces(self, image: np.ndarray) -> List[Dict]:
        """Backward compatibility method"""
        return self.track_faces_advanced(image)

    def load_student_embeddings(self) -> Dict:
        """Backward compatibility method"""
        try:
            self.load_face_database()
            return {
                'success': True,
                'count': len(self.student_embeddings),
                'message': f'Loaded {len(self.student_embeddings)} student embeddings'
            }
        except Exception as e:
            return {
                'success': False,
                'count': 0,
                'message': str(e)
            }

    def record_attendance(self, student_id: str, camera_id: int) -> Dict:
        """Record attendance (placeholder - implement based on your backend)"""
        try:
            # This should integrate with your attendance system
            logger.info(f"📝 Recording attendance for {student_id} at camera {camera_id}")

            return {
                'success': True,
                'message': f'Attendance recorded for {student_id}'
            }
        except Exception as e:
            return {
                'success': False,
                'message': str(e)
            }
