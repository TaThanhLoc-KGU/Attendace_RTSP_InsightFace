"""
Face recognition service using InsightFace
"""
import os
import json
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


class FaceRecognitionService:
    """Service để nhận diện khuôn mặt với InsightFace"""

    def __init__(self):
        self.face_app = None
        self.face_db = {}
        self.embeddings_cache = {}
        self.is_initialized = False

        if INSIGHTFACE_AVAILABLE:
            self.initialize_face_recognition()
        else:
            face_logger.error("❌ InsightFace not available. Install with: pip install insightface")

        self.load_face_database()

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

    def load_face_database(self):
        """Load face database từ file"""
        try:
            face_db_path = Path(config.FACE_DB_PATH)

            if face_db_path.exists():
                with open(face_db_path, 'r', encoding='utf-8') as f:
                    self.face_db = json.load(f)
                face_logger.info(f"✅ Loaded {len(self.face_db)} faces from database")
            else:
                face_logger.info("📁 No face database found, creating new one")
                self.face_db = {}
                self.save_face_database()

        except Exception as e:
            face_logger.error(f"❌ Error loading face database: {e}")
            self.face_db = {}

    def save_face_database(self):
        """Lưu face database vào file"""
        try:
            face_db_path = Path(config.FACE_DB_PATH)
            face_db_path.parent.mkdir(parents=True, exist_ok=True)

            with open(face_db_path, 'w', encoding='utf-8') as f:
                json.dump(self.face_db, f, ensure_ascii=False, indent=2)

            face_logger.info("💾 Face database saved successfully")

        except Exception as e:
            face_logger.error(f"❌ Error saving face database: {e}")

    def add_face_to_database(self, name: str, image: np.ndarray) -> bool:
        """Thêm khuôn mặt vào database"""
        if not self.is_initialized:
            face_logger.error("Face recognition not initialized")
            return False

        try:
            faces = self.face_app.get(image)
            if len(faces) == 0:
                face_logger.warning("No face detected in image")
                return False

            if len(faces) > 1:
                face_logger.warning(f"Multiple faces detected ({len(faces)}), using the first one")

            # Lấy face đầu tiên
            face = faces[0]
            embedding = face.embedding.tolist()

            # Kiểm tra duplicate
            if name in self.face_db:
                face_logger.warning(f"Face '{name}' already exists, updating...")

            # Lưu vào database
            self.face_db[name] = {
                'embedding': embedding,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'face_count': len(faces),
                'bbox': face.bbox.tolist() if hasattr(face, 'bbox') else [],
                'landmarks': face.kps.tolist() if hasattr(face, 'kps') else [],
                'age': getattr(face, 'age', 0),
                'gender': getattr(face, 'gender', 0),
                'embedding_size': len(embedding)
            }

            self.save_face_database()
            face_logger.info(f"✅ Added face '{name}' to database")
            return True

        except Exception as e:
            face_logger.error(f"❌ Error adding face to database: {e}")
            return False

    def recognize_faces(self, image: np.ndarray) -> List[Dict]:
        """Nhận diện khuôn mặt trong ảnh"""
        if not self.is_initialized:
            return []

        try:
            faces = self.face_app.get(image)
            results = []

            face_logger.debug(f"🔍 Detected {len(faces)} faces in image")

            for i, face in enumerate(faces):
                # Tính toán similarity với faces trong database
                best_match = None
                best_score = 0

                for name, face_data in self.face_db.items():
                    try:
                        db_embedding = np.array(face_data['embedding'])
                        similarity = self._calculate_similarity(face.embedding, db_embedding)

                        if similarity > best_score and similarity > config.RECOGNITION_THRESHOLD:
                            best_score = similarity
                            best_match = name
                    except Exception as e:
                        face_logger.warning(f"Error calculating similarity for {name}: {e}")
                        continue

                # Tạo bounding box
                bbox = face.bbox.astype(int) if hasattr(face, 'bbox') else [0, 0, 0, 0]

                result = {
                    'face_id': i,
                    'name': best_match if best_match else 'Unknown',
                    'confidence': float(best_score),
                    'bbox': bbox.tolist(),
                    'age': getattr(face, 'age', 0),
                    'gender': getattr(face, 'gender', 0),  # 0: male, 1: female
                    'landmarks': face.kps.tolist() if hasattr(face, 'kps') else [],
                    'embedding': face.embedding.tolist(),
                    'is_known': best_match is not None
                }

                results.append(result)

                # Log result
                status = "✅ Known" if best_match else "❓ Unknown"
                face_logger.debug(f"{status} face {i + 1}: {result['name']} ({result['confidence']:.3f})")

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

    def remove_face_from_database(self, name: str) -> bool:
        """Xóa khuôn mặt khỏi database"""
        try:
            if name in self.face_db:
                del self.face_db[name]
                self.save_face_database()
                face_logger.info(f"🗑️ Removed face '{name}' from database")
                return True
            else:
                face_logger.warning(f"Face '{name}' not found in database")
                return False

        except Exception as e:
            face_logger.error(f"❌ Error removing face from database: {e}")
            return False

    def get_face_list(self) -> List[Dict]:
        """Lấy danh sách tất cả faces trong database"""
        faces = []
        for name, face_data in self.face_db.items():
            faces.append({
                'name': name,
                'created_at': face_data.get('created_at', ''),
                'updated_at': face_data.get('updated_at', ''),
                'age': face_data.get('age', 0),
                'gender': face_data.get('gender', 0),
                'embedding_size': face_data.get('embedding_size', 0)
            })
        return faces

    def get_face_info(self, name: str) -> Optional[Dict]:
        """Lấy thông tin chi tiết của một face"""
        if name in self.face_db:
            return self.face_db[name].copy()
        return None

    def update_face_info(self, name: str, info: Dict) -> bool:
        """Cập nhật thông tin face"""
        try:
            if name in self.face_db:
                self.face_db[name].update(info)
                self.face_db[name]['updated_at'] = datetime.now().isoformat()
                self.save_face_database()
                face_logger.info(f"✅ Updated face '{name}' info")
                return True
            else:
                face_logger.warning(f"Face '{name}' not found")
                return False

        except Exception as e:
            face_logger.error(f"❌ Error updating face info: {e}")
            return False

    def backup_face_database(self, backup_path: str) -> bool:
        """Backup face database"""
        try:
            backup_data = {
                'version': '1.0',
                'created_at': datetime.now().isoformat(),
                'faces': self.face_db
            }

            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, ensure_ascii=False, indent=2)

            face_logger.info(f"💾 Face database backed up to {backup_path}")
            return True

        except Exception as e:
            face_logger.error(f"❌ Error backing up face database: {e}")
            return False

    def restore_face_database(self, backup_path: str) -> bool:
        """Restore face database from backup"""
        try:
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)

            if 'faces' in backup_data:
                self.face_db = backup_data['faces']
                self.save_face_database()
                face_logger.info(f"✅ Face database restored from {backup_path}")
                return True
            else:
                face_logger.error("Invalid backup format")
                return False

        except Exception as e:
            face_logger.error(f"❌ Error restoring face database: {e}")
            return False

    def get_statistics(self) -> Dict:
        """Lấy thống kê face database"""
        total_faces = len(self.face_db)
        male_count = sum(1 for face in self.face_db.values() if face.get('gender', 0) == 0)
        female_count = sum(1 for face in self.face_db.values() if face.get('gender', 0) == 1)

        ages = [face.get('age', 0) for face in self.face_db.values() if face.get('age', 0) > 0]
        avg_age = sum(ages) / len(ages) if ages else 0

        stats = {
            'total_faces': total_faces,
            'male_count': male_count,
            'female_count': female_count,
            'unknown_gender': total_faces - male_count - female_count,
            'average_age': round(avg_age, 1),
            'is_initialized': self.is_initialized,
            'recognition_threshold': config.RECOGNITION_THRESHOLD,
            'database_size': len(json.dumps(self.face_db))
        }

        face_logger.info(f"📊 Face database stats: {stats}")
        return stats

    def validate_face_image(self, image: np.ndarray) -> Dict:
        """Validate face image quality"""
        if not self.is_initialized:
            return {'valid': False, 'message': 'Face recognition not initialized'}

        try:
            faces = self.face_app.get(image)

            if len(faces) == 0:
                return {'valid': False, 'message': 'No face detected'}

            if len(faces) > 1:
                return {'valid': False,
                        'message': f'Multiple faces detected ({len(faces)}), please use image with single face'}

            face = faces[0]
            bbox = face.bbox

            # Check face size
            face_width = bbox[2] - bbox[0]
            face_height = bbox[3] - bbox[1]

            if face_width < 100 or face_height < 100:
                return {'valid': False, 'message': 'Face too small, minimum size is 100x100 pixels'}

            # Check face position (not too close to edges)
            img_height, img_width = image.shape[:2]
            margin_x = img_width * 0.1
            margin_y = img_height * 0.1

            if (bbox[0] < margin_x or bbox[1] < margin_y or
                    bbox[2] > img_width - margin_x or bbox[3] > img_height - margin_y):
                return {'valid': False, 'message': 'Face too close to image edges'}

            return {
                'valid': True,
                'message': 'Face image is valid',
                'face_size': (int(face_width), int(face_height)),
                'face_position': bbox.tolist(),
                'age': getattr(face, 'age', 0),
                'gender': getattr(face, 'gender', 0)
            }

        except Exception as e:
            face_logger.error(f"Error validating face image: {e}")
            return {'valid': False, 'message': f'Error validating image: {str(e)}'}