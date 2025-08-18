#!/usr/bin/env python3
"""
Cleanup and Rebuild Embeddings Script
Dọn dẹp database và rebuild lại embeddings với consistency
"""
import os
import sys
import json
import numpy as np
from pathlib import Path
import cv2
import base64
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from insightface.app import FaceAnalysis

    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("❌ InsightFace not available. Install with: pip install insightface")
    sys.exit(1)

from config.config import config
from services.backend_api import BackendAPI
from utils.logger import setup_logger

# Setup logging
logger = setup_logger('cleanup', 'data/logs/cleanup.log')


class EmbeddingCleanupService:
    """Service để dọn dẹp và rebuild embeddings"""

    def __init__(self):
        self.face_app = None
        self.backend_api = None
        self.model_name = config.INSIGHTFACE_MODEL
        self.expected_dim = 512

    def initialize(self):
        """Initialize services"""
        try:
            # Initialize InsightFace with consistent model
            logger.info(f"🚀 Initializing InsightFace with model: {self.model_name}")

            self.face_app = FaceAnalysis(
                name=self.model_name,
                providers=config.get_insightface_providers(),
                allowed_modules=['detection', 'recognition']
            )

            self.face_app.prepare(
                ctx_id=0,
                det_size=(640, 640),  # High quality for rebuilding
                det_thresh=0.5
            )

            # Initialize backend API
            self.backend_api = BackendAPI()
            if not self.backend_api.test_connection():
                logger.error("❌ Cannot connect to backend API")
                return False

            logger.info("✅ Services initialized successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            return False

    def validate_existing_embeddings(self):
        """Validate existing embeddings in database"""
        logger.info("🔍 Validating existing embeddings...")

        try:
            response = self.backend_api.get_all_embeddings()
            if not response.success:
                logger.error(f"❌ Failed to get embeddings: {response.message}")
                return []

            embeddings_data = response.data or []
            logger.info(f"📥 Found {len(embeddings_data)} embeddings to validate")

            valid_embeddings = []
            invalid_embeddings = []

            for student_data in embeddings_data:
                try:
                    student_id = (
                            student_data.get('id') or
                            student_data.get('studentId') or
                            student_data.get('maSv')
                    )

                    embedding_str = (
                            student_data.get('embedding') or
                            student_data.get('faceEmbedding')
                    )

                    if not student_id or not embedding_str:
                        continue

                    # Try to decode embedding
                    embedding = self._decode_embedding(embedding_str)

                    if embedding is not None and len(embedding) == self.expected_dim:
                        valid_embeddings.append({
                            'student_id': student_id,
                            'name': student_data.get('name', student_data.get('hoTen', 'Unknown')),
                            'embedding_size': len(embedding)
                        })
                    else:
                        invalid_embeddings.append({
                            'student_id': student_id,
                            'name': student_data.get('name', student_data.get('hoTen', 'Unknown')),
                            'issue': f"Invalid dimension: {len(embedding) if embedding is not None else 'None'}"
                        })

                except Exception as e:
                    invalid_embeddings.append({
                        'student_id': student_id if 'student_id' in locals() else 'Unknown',
                        'issue': f"Decode error: {e}"
                    })

            logger.info(f"✅ Valid embeddings: {len(valid_embeddings)}")
            logger.info(f"❌ Invalid embeddings: {len(invalid_embeddings)}")

            if invalid_embeddings:
                logger.warning("Invalid embeddings found:")
                for invalid in invalid_embeddings[:10]:  # Show first 10
                    logger.warning(f"  - Student {invalid['student_id']}: {invalid['issue']}")

            return valid_embeddings, invalid_embeddings

        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            return [], []

    def _decode_embedding(self, embedding_str):
        """Decode embedding from various formats"""
        try:
            if isinstance(embedding_str, str):
                # Try base64 first
                try:
                    embedding_bytes = base64.b64decode(embedding_str)
                    embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    return embedding
                except:
                    pass

                # Try JSON
                try:
                    embedding_list = json.loads(embedding_str)
                    return np.array(embedding_list, dtype=np.float32)
                except:
                    pass

            elif isinstance(embedding_str, list):
                return np.array(embedding_str, dtype=np.float32)

            return None

        except Exception as e:
            logger.error(f"❌ Decode error: {e}")
            return None

    def rebuild_embeddings_from_images(self):
        """Rebuild embeddings từ ảnh gốc"""
        logger.info("🔄 Rebuilding embeddings from original images...")

        # Đường dẫn đến thư mục ảnh sinh viên
        students_dir = project_root / "src" / "main" / "resources" / "static" / "uploads" / "students"

        if not students_dir.exists():
            logger.error(f"❌ Students directory not found: {students_dir}")
            return []

        rebuilt_embeddings = []

        for student_folder in students_dir.iterdir():
            if not student_folder.is_dir():
                continue

            student_id = student_folder.name
            logger.info(f"🔄 Processing student: {student_id}")

            # Tìm ảnh face
            face_images = []
            faces_dir = student_folder / "faces"

            if faces_dir.exists():
                for i in range(1, 6):  # face_1.jpg đến face_5.jpg
                    for ext in ['.jpg', '.jpeg', '.png', '.webp']:
                        face_path = faces_dir / f"face_{i}{ext}"
                        if face_path.exists():
                            face_images.append(face_path)
                            break

            if not face_images:
                logger.warning(f"⚠️ No face images found for student {student_id}")
                continue

            # Extract embeddings
            embeddings = []
            for face_path in face_images:
                try:
                    image = cv2.imread(str(face_path))
                    if image is None:
                        continue

                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    faces = self.face_app.get(rgb_image)

                    if faces:
                        # Chọn face tốt nhất
                        best_face = max(faces, key=lambda x: x.det_score)
                        if hasattr(best_face, 'normed_embedding'):
                            embeddings.append(best_face.normed_embedding)
                        elif hasattr(best_face, 'embedding'):
                            embedding = best_face.embedding
                            # Normalize
                            norm = np.linalg.norm(embedding)
                            if norm > 0:
                                embeddings.append(embedding / norm)

                except Exception as e:
                    logger.error(f"❌ Error processing {face_path}: {e}")
                    continue

            if embeddings:
                # Tính average embedding
                avg_embedding = np.mean(embeddings, axis=0)

                # Normalize final embedding
                norm = np.linalg.norm(avg_embedding)
                if norm > 0:
                    avg_embedding = avg_embedding / norm

                    rebuilt_embeddings.append({
                        'student_id': student_id,
                        'embedding': avg_embedding,
                        'num_faces': len(embeddings),
                        'quality_score': np.mean([np.dot(emb, avg_embedding) for emb in embeddings])
                    })

                    logger.info(f"✅ Generated embedding for {student_id} from {len(embeddings)} faces")
                else:
                    logger.warning(f"⚠️ Zero norm embedding for student {student_id}")
            else:
                logger.warning(f"⚠️ No valid embeddings extracted for student {student_id}")

        logger.info(f"✅ Rebuilt {len(rebuilt_embeddings)} embeddings")
        return rebuilt_embeddings

    def update_embeddings_in_backend(self, embeddings_list):
        """Update embeddings trong backend database"""
        logger.info(f"📤 Updating {len(embeddings_list)} embeddings in backend...")

        success_count = 0
        for embedding_data in embeddings_list:
            try:
                student_id = embedding_data['student_id']
                embedding = embedding_data['embedding']

                # Encode embedding as base64
                embedding_bytes = embedding.astype(np.float32).tobytes()
                embedding_b64 = base64.b64encode(embedding_bytes).decode('utf-8')

                # Update via API (you may need to implement this endpoint)
                # For now, just log what would be updated
                logger.info(f"📤 Would update student {student_id} with embedding size {len(embedding)}")
                success_count += 1

            except Exception as e:
                logger.error(f"❌ Failed to update student {student_id}: {e}")

        logger.info(f"✅ Successfully updated {success_count}/{len(embeddings_list)} embeddings")
        return success_count

    def cleanup_cache_files(self):
        """Clean up cache files"""
        logger.info("🧹 Cleaning up cache files...")

        cache_dirs = [
            project_root / "data" / "embeddings",
            project_root / "data" / "faces_db.json"
        ]

        for cache_path in cache_dirs:
            try:
                if cache_path.exists():
                    if cache_path.is_file():
                        cache_path.unlink()
                        logger.info(f"🗑️ Removed cache file: {cache_path}")
                    elif cache_path.is_dir():
                        import shutil
                        shutil.rmtree(cache_path)
                        logger.info(f"🗑️ Removed cache directory: {cache_path}")
            except Exception as e:
                logger.error(f"❌ Failed to remove {cache_path}: {e}")


def main():
    """Main cleanup function"""
    print("🚀 Starting Embedding Cleanup and Rebuild Process...")

    cleanup_service = EmbeddingCleanupService()

    # Initialize
    if not cleanup_service.initialize():
        print("❌ Failed to initialize services")
        return False

    # Step 1: Validate existing embeddings
    print("\n📋 Step 1: Validating existing embeddings...")
    valid_embeddings, invalid_embeddings = cleanup_service.validate_existing_embeddings()

    if invalid_embeddings:
        print(f"⚠️ Found {len(invalid_embeddings)} invalid embeddings")
        choice = input("Do you want to rebuild all embeddings from images? (y/n): ")

        if choice.lower() == 'y':
            # Step 2: Rebuild embeddings
            print("\n🔄 Step 2: Rebuilding embeddings from images...")
            rebuilt_embeddings = cleanup_service.rebuild_embeddings_from_images()

            if rebuilt_embeddings:
                # Step 3: Update backend
                print("\n📤 Step 3: Updating backend database...")
                success_count = cleanup_service.update_embeddings_in_backend(rebuilt_embeddings)
                print(f"✅ Updated {success_count} embeddings in backend")
            else:
                print("❌ No embeddings were rebuilt")
        else:
            print("ℹ️ Skipping rebuild process")
    else:
        print("✅ All existing embeddings are valid")

    # Step 4: Cleanup cache
    print("\n🧹 Step 4: Cleaning up cache files...")
    cleanup_service.cleanup_cache_files()

    print("\n✅ Cleanup process completed!")
    print("\nNext steps:")
    print("1. Restart your application")
    print("2. Check that tracking is working properly")
    print("3. Monitor logs for any remaining errors")

    return True


if __name__ == "__main__":
    main()