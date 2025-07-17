"""
Embedding cache service for student embeddings
"""
import os
import pickle
import numpy as np
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json

from config.config import config
from utils.logger import face_logger

class EmbeddingCacheService:
    """
    Service để cache embeddings từ database vào file local
    cho việc face recognition nhanh chóng
    """

    def __init__(self, cache_dir: str = None):
        self.cache_dir = Path(cache_dir or config.EMBEDDING_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache files
        self.embeddings_file = self.cache_dir / "student_embeddings.pkl"
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.index_file = self.cache_dir / "student_index.json"

        # In-memory cache
        self.embeddings_cache = {}
        self.student_index = {}
        self.last_sync = None

        face_logger.info(f"EmbeddingCacheService initialized with cache dir: {self.cache_dir}")

    def load_cache_from_files(self) -> bool:
        """Load embeddings cache from local files"""
        try:
            # Load embeddings
            if self.embeddings_file.exists():
                with open(self.embeddings_file, 'rb') as f:
                    self.embeddings_cache = pickle.load(f)
                face_logger.info(f"Loaded {len(self.embeddings_cache)} embeddings from cache")

            # Load student index
            if self.index_file.exists():
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    self.student_index = json.load(f)
                face_logger.info(f"Loaded student index with {len(self.student_index)} students")

            # Load metadata
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                    self.last_sync = datetime.fromisoformat(metadata.get('last_sync', ''))
                face_logger.info(f"Cache last synced: {self.last_sync}")

            return len(self.embeddings_cache) > 0

        except Exception as e:
            face_logger.error(f"Error loading cache from files: {e}")
            return False

    def save_cache_to_files(self) -> bool:
        """Save embeddings cache to local files"""
        try:
            # Save embeddings
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)

            # Save student index
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.student_index, f, ensure_ascii=False, indent=2)

            # Save metadata
            metadata = {
                'last_sync': datetime.now().isoformat(),
                'cache_size': len(self.embeddings_cache),
                'index_size': len(self.student_index)
            }
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            self.last_sync = datetime.now()
            face_logger.info(f"Cache saved successfully. {len(self.embeddings_cache)} embeddings cached.")
            return True

        except Exception as e:
            face_logger.error(f"Error saving cache to files: {e}")
            return False

    def sync_embeddings_from_database(self, backend_api) -> Dict[str, any]:
        """
        Sync embeddings từ Spring Boot database về local cache

        Returns:
            Dict with sync results
        """
        face_logger.info("🔄 Starting embeddings sync from database...")

        try:
            # Get all students with embeddings from API
            response = backend_api._make_request('GET', '/python/embeddings')

            if not response.success:
                return {
                    'success': False,
                    'message': f"Failed to get students from API: {response.message}",
                    'count': 0
                }

            students_data = response.data
            if not isinstance(students_data, list):
                return {
                    'success': False,
                    'message': "Invalid data format from API",
                    'count': 0
                }

            # Process students and embeddings
            processed_count = 0
            skipped_count = 0
            updated_embeddings = {}
            updated_index = {}

            for student in students_data:
                try:
                    student_id = student.get('studentId')
                    name = student.get('name')
                    embedding_data = student.get('embedding')

                    if not student_id:
                        face_logger.warning("Student missing studentId, skipping")
                        skipped_count += 1
                        continue

                    # Update student index
                    updated_index[student_id] = {
                        'studentId': student_id,
                        'name': name or f"Student_{student_id}",
                        'last_updated': datetime.now().isoformat()
                    }

                    # Process embedding if available
                    if embedding_data:
                        try:
                            # Decode embedding
                            embedding_array = self._decode_embedding(embedding_data)

                            if embedding_array is not None:
                                updated_embeddings[student_id] = embedding_array
                                processed_count += 1
                                face_logger.debug(f"✅ Processed embedding for {student_id}")
                            else:
                                face_logger.warning(f"Failed to decode embedding for {student_id}")
                                skipped_count += 1

                        except Exception as e:
                            face_logger.warning(f"Failed to process embedding for {student_id}: {e}")
                            skipped_count += 1
                    else:
                        face_logger.debug(f"No embedding for student {student_id}")
                        skipped_count += 1

                except Exception as e:
                    face_logger.error(f"Error processing student data: {e}")
                    skipped_count += 1
                    continue

            # Update in-memory cache
            self.embeddings_cache = updated_embeddings
            self.student_index = updated_index

            # Save to files
            save_success = self.save_cache_to_files()

            result = {
                'success': save_success,
                'count': processed_count,
                'total_students': len(updated_index),
                'skipped': skipped_count,
                'last_sync': self.last_sync.isoformat() if self.last_sync else None
            }

            if save_success:
                face_logger.info(f"✅ Embeddings sync completed: {processed_count} processed, {skipped_count} skipped")
            else:
                face_logger.error("❌ Failed to save cache after sync")

            return result

        except Exception as e:
            face_logger.error(f"❌ Error syncing embeddings: {e}")
            return {
                'success': False,
                'message': f"Sync error: {str(e)}",
                'count': 0
            }

    def _decode_embedding(self, embedding_data) -> Optional[np.ndarray]:
        """
        Decode embedding data từ backend với nhiều format support
        """
        try:
            if isinstance(embedding_data, str):
                # Try decode as base64 first
                try:
                    embedding_bytes = base64.b64decode(embedding_data)
                    embedding_array = np.frombuffer(embedding_bytes, dtype=np.float32)
                    face_logger.debug(f"Decoded base64 embedding, size: {len(embedding_array)}")
                except Exception:
                    # Try parse as JSON array
                    try:
                        embedding_list = json.loads(embedding_data)
                        embedding_array = np.array(embedding_list, dtype=np.float32)
                        face_logger.debug(f"Decoded JSON embedding, size: {len(embedding_array)}")
                    except Exception:
                        face_logger.warning(f"Failed to decode string embedding: {embedding_data[:50]}...")
                        return None

            elif isinstance(embedding_data, list):
                embedding_array = np.array(embedding_data, dtype=np.float32)
                face_logger.debug(f"Converted list embedding, size: {len(embedding_array)}")

            else:
                face_logger.warning(f"Unknown embedding data type: {type(embedding_data)}")
                return None

            # Validate embedding size (should be 512 for InsightFace)
            if len(embedding_array) == 512:
                return embedding_array
            else:
                face_logger.warning(f"Invalid embedding size: {len(embedding_array)}, expected 512")
                return None

        except Exception as e:
            face_logger.error(f"Error decoding embedding: {e}")
            return None

    def get_embedding(self, student_id: str) -> Optional[np.ndarray]:
        """Get embedding for student"""
        return self.embeddings_cache.get(student_id)

    def get_student_info(self, student_id: str) -> Optional[Dict]:
        """Get student information"""
        return self.student_index.get(student_id)

    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        """Get all embeddings"""
        return self.embeddings_cache.copy()

    def get_all_students(self) -> Dict[str, Dict]:
        """Get all student information"""
        return self.student_index.copy()

    def search_students(self, query: str) -> List[Dict]:
        """Search students by name or ID"""
        query = query.lower()
        results = []

        for student_id, student_info in self.student_index.items():
            name = student_info.get('name', '').lower()
            if query in name or query in student_id.lower():
                results.append({
                    'studentId': student_id,
                    'name': student_info.get('name'),
                    'hasEmbedding': student_id in self.embeddings_cache
                })

        return results

    def get_statistics(self) -> Dict:
        """Get cache statistics"""
        return {
            'total_students': len(self.student_index),
            'students_with_embeddings': len(self.embeddings_cache),
            'cache_hit_rate': len(self.embeddings_cache) / len(self.student_index) if self.student_index else 0,
            'last_sync': self.last_sync.isoformat() if self.last_sync else None,
            'cache_size_mb': self._get_cache_size_mb(),
            'embedding_dimension': 512  # InsightFace standard
        }

    def _get_cache_size_mb(self) -> float:
        """Calculate cache size in MB"""
        try:
            total_size = 0
            for file_path in [self.embeddings_file, self.index_file, self.metadata_file]:
                if file_path.exists():
                    total_size += file_path.stat().st_size
            return total_size / (1024 * 1024)  # Convert to MB
        except:
            return 0.0

    def clear_cache(self):
        """Clear all cache data"""
        self.embeddings_cache.clear()
        self.student_index.clear()
        self.last_sync = None

        # Remove cache files
        for file_path in [self.embeddings_file, self.index_file, self.metadata_file]:
            if file_path.exists():
                file_path.unlink()

        face_logger.info("🗑️ Cache cleared successfully")

    def is_cache_expired(self, max_age_hours: int = 24) -> bool:
        """Check if cache is expired"""
        if not self.last_sync:
            return True

        age = datetime.now() - self.last_sync
        return age.total_seconds() > (max_age_hours * 3600)

    def get_cache_age(self) -> Optional[timedelta]:
        """Get cache age"""
        if self.last_sync:
            return datetime.now() - self.last_sync
        return None

    def backup_cache(self, backup_path: str) -> bool:
        """Backup cache to specified path"""
        try:
            backup_data = {
                'embeddings': {k: v.tolist() for k, v in self.embeddings_cache.items()},
                'index': self.student_index,
                'metadata': {
                    'backup_date': datetime.now().isoformat(),
                    'original_sync': self.last_sync.isoformat() if self.last_sync else None,
                    'version': '2.0'
                }
            }

            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, ensure_ascii=False, indent=2)

            face_logger.info(f"💾 Cache backed up to {backup_path}")
            return True

        except Exception as e:
            face_logger.error(f"❌ Error backing up cache: {e}")
            return False

    def restore_cache(self, backup_path: str) -> bool:
        """Restore cache from backup"""
        try:
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)

            # Restore embeddings
            if 'embeddings' in backup_data:
                self.embeddings_cache = {
                    k: np.array(v, dtype=np.float32)
                    for k, v in backup_data['embeddings'].items()
                }

            # Restore index
            if 'index' in backup_data:
                self.student_index = backup_data['index']

            # Restore metadata
            if 'metadata' in backup_data:
                metadata = backup_data['metadata']
                if 'original_sync' in metadata and metadata['original_sync']:
                    self.last_sync = datetime.fromisoformat(metadata['original_sync'])

            # Save to files
            self.save_cache_to_files()

            face_logger.info(f"✅ Cache restored from {backup_path}")
            return True

        except Exception as e:
            face_logger.error(f"❌ Error restoring cache: {e}")
            return False