"""
Enhanced Face recognition service with student embeddings from backend
"""
import os
import json
import pickle
import base64
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import cv2

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

from config.config import config
from utils.logger import face_logger

class StudentEmbeddingService:
    """Service để quản lý student embeddings từ backend"""

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
    """Enhanced Face recognition service với InsightFace và student embeddings"""

    def __init__(self, backend_api):
        self.backend_api = backend_api
        self.face_app = None
        self.is_initialized = False

        # Student embedding service
        self.student_service = StudentEmbeddingService(backend_api)

        # Load từ cache trước
        if self.student_service.load_from_cache():
            face_logger.info("✅ Loaded student embeddings from cache")

        # Initialize InsightFace
        if INSIGHTFACE_AVAILABLE:
            self.initialize_face_recognition()
        else:
            face_logger.error("❌ InsightFace not available. Install with: pip install insightface")

    def initialize_face_recognition(self):
        """Khởi tạo InsightFace model"""
        try:
            face_logger.info("🔍 Initializing InsightFace...")

            self.face_app = FaceAnalysis(
                name=config.INSIGHTFACE_MODEL,
                providers=config.INSIGHTFACE_PROVIDERS,
                allowed_modules=['detection', 'recognition']
            )

            # Prepare with context
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))

            self.is_initialized = True
            face_logger.info("✅ InsightFace initialized successfully")

        except Exception as e:
            face_logger.error(f"❌ Failed to initialize InsightFace: {e}")
            self.face_app = None
            self.is_initialized = False

    def load_student_embeddings(self) -> Dict[str, any]:
        """Load student embeddings từ backend"""
        result = self.student_service.load_embeddings_from_backend()

        if result['success']:
            face_logger.info(f"✅ Successfully loaded {result['count']} student embeddings")
        else:
            face_logger.error(f"❌ Failed to load student embeddings: {result['message']}")

        return result

    def recognize_faces(self, image: np.ndarray) -> List[Dict]:
        """Nhận diện khuôn mặt trong ảnh với student database"""
        if not self.is_initialized:
            return []

        try:
            faces = self.face_app.get(image)
            results = []

            face_logger.debug(f"🔍 Detected {len(faces)} faces in image")

            for i, face in enumerate(faces):
                # Tính toán similarity với student embeddings
                best_match = None
                best_score = 0
                best_student_id = None

                for student_id, student_embedding in self.student_service.student_embeddings.items():
                    try:
                        similarity = self._calculate_similarity(face.embedding, student_embedding)

                        if similarity > best_score and similarity > config.RECOGNITION_THRESHOLD:
                            best_score = similarity
                            best_student_id = student_id
                            student_info = self.student_service.get_student_info(student_id)
                            best_match = student_info['name'] if student_info else student_id
                    except Exception as e:
                        face_logger.warning(f"Error calculating similarity for {student_id}: {e}")
                        continue

                # Tạo bounding box - FIX LỖII NONETYP
                bbox = [0, 0, 0, 0]
                if hasattr(face, 'bbox') and face.bbox is not None:
                    bbox = face.bbox.astype(int).tolist()

                result = {
                    'face_id': i,
                    'name': best_match if best_match else 'Unknown',
                    'student_id': best_student_id,
                    'confidence': float(best_score),
                    'bbox': bbox,
                    'age': getattr(face, 'age', 0),
                    'gender': getattr(face, 'gender', 0),
                    'landmarks': face.kps.tolist() if hasattr(face, 'kps') and face.kps is not None else [],
                    'embedding': face.embedding.tolist(),
                    'is_known': best_match is not None
                }

                results.append(result)

                # Log result
                status = "✅ Known" if best_match else "❓ Unknown"
                face_logger.debug(f"{status} face {i+1}: {result['name']} ({result['confidence']:.3f})")

            return results

        except Exception as e:
            face_logger.error(f"❌ Error recognizing faces: {e}")
            return []

    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Tính toán similarity giữa hai embeddings"""
        try:
            # Cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return float(similarity)

        except Exception as e:
            face_logger.error(f"Error calculating similarity: {e}")
            return 0.0

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        student_stats = self.student_service.get_statistics()

        return {
            'total_students': student_stats['total_students'],
            'cache_size': student_stats['cache_size'],
            'is_initialized': self.is_initialized,
            'recognition_threshold': config.RECOGNITION_THRESHOLD,
            'last_loaded': student_stats['last_loaded']
        }

    def record_attendance(self, student_id: str, camera_id: int) -> Dict:
        """Ghi nhận điểm danh"""
        try:
            face_logger.info(f"📝 Recording attendance for {student_id} at camera {camera_id}")

            # Call backend API để ghi điểm danh
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
                'version': '2.0',
                'created_at': datetime.now().isoformat(),
                'embeddings': {k: v.tolist() for k, v in self.student_service.student_embeddings.items()},
                'index': self.student_service.student_index
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