"""
Camera service for managing cameras and RTSP streams
"""
import cv2
from typing import Dict, List, Optional
from dataclasses import dataclass
from .backend_api import BackendAPI
from config.config import config
from utils.logger import camera_logger


@dataclass
class Camera:
    """Camera data model"""
    id: int
    name: str
    rtsp_url: str
    location: str = ""
    active: bool = True
    hls_url: str = ""
    current_schedule: Optional[Dict] = None

    def __post_init__(self):
        """Post initialization processing"""
        if not self.hls_url:
            self.hls_url = f"/stream/camera_{self.id}/playlist.m3u8"


class CameraService:
    """Service để quản lý cameras"""

    def __init__(self, backend_api: BackendAPI):
        self.backend_api = backend_api
        self.cameras: List[Camera] = []
        self.active_streams: Dict[int, cv2.VideoCapture] = {}

        camera_logger.info("📹 Initialized Camera Service")

    def load_cameras_from_backend(self) -> Dict[str, any]:
        """Load cameras từ backend với improved mapping"""
        camera_logger.info("📹 Loading cameras from backend...")

        try:
            # Try Flask-specific endpoint first
            response = self.backend_api.get_active_cameras()

            if not response.success:
                # Fallback to general cameras endpoint
                camera_logger.warning("Flask endpoint failed, trying general endpoint...")
                response = self.backend_api.get_cameras()

                if not response.success:
                    return {
                        'success': False,
                        'message': f"Backend connection failed: {response.message}",
                        'cameras': [],
                        'count': 0
                    }

            # Parse camera data with multiple field mapping
            cameras_data = response.data
            if not isinstance(cameras_data, list):
                # Handle paginated response
                if isinstance(cameras_data, dict):
                    cameras_data = cameras_data.get('content', [])
                    if not cameras_data:
                        cameras_data = cameras_data.get('data', [])

                if not isinstance(cameras_data, list):
                    return {
                        'success': False,
                        'message': "Invalid camera data format from backend",
                        'cameras': [],
                        'count': 0
                    }

            # Convert to Camera objects với improved field mapping
            self.cameras = []
            for cam_data in cameras_data:
                try:
                    camera = self._parse_camera_data(cam_data)
                    if camera:
                        self.cameras.append(camera)

                except Exception as e:
                    camera_logger.warning(f"Error parsing camera data: {e}")
                    continue

            camera_logger.info(f"✅ Loaded {len(self.cameras)} cameras")
            return {
                'success': True,
                'cameras': [self._camera_to_dict(cam) for cam in self.cameras],
                'count': len(self.cameras)
            }

        except Exception as e:
            camera_logger.error(f"❌ Error loading cameras: {e}")
            return {
                'success': False,
                'message': f"Failed to load cameras: {str(e)}",
                'cameras': [],
                'count': 0
            }

    def _parse_camera_data(self, cam_data: Dict) -> Optional[Camera]:
        """Parse camera data from various backend formats"""
        try:
            # Map various field names for RTSP URL
            rtsp_url = (
                    cam_data.get('rtspUrl') or
                    cam_data.get('ipAddress') or
                    cam_data.get('ip_address') or
                    cam_data.get('url') or
                    cam_data.get('streamUrl') or
                    ''
            )

            # Map various field names for camera name
            camera_name = (
                    cam_data.get('tenCamera') or
                    cam_data.get('name') or
                    cam_data.get('cameraName') or
                    f"Camera {cam_data.get('id', 'Unknown')}"
            )

            # Map various field names for location
            location = (
                    cam_data.get('tenPhong') or
                    cam_data.get('maPhong') or
                    cam_data.get('location') or
                    cam_data.get('room') or
                    'Unknown Location'
            )

            # Map active status
            active = cam_data.get('isActive', cam_data.get('active', True))

            # Create camera object
            camera = Camera(
                id=cam_data.get('id'),
                name=camera_name,
                rtsp_url=rtsp_url,
                location=location,
                active=active,
                hls_url=cam_data.get('hlsUrl', ''),
                current_schedule=cam_data.get('currentSchedule')
            )

            # Validate RTSP URL
            if rtsp_url and ('rtsp://' in rtsp_url.lower() or 'http://' in rtsp_url.lower()):
                camera_logger.info(f"✅ Valid camera: {camera.name} - {rtsp_url[:30]}...")
                return camera
            else:
                camera_logger.warning(f"⚠️ Camera {camera.name} has invalid/missing RTSP URL")
                return None

        except Exception as e:
            camera_logger.error(f"❌ Error parsing camera data: {e}")
            return None

    def _camera_to_dict(self, camera: Camera) -> Dict:
        """Convert Camera object to dictionary with proper RTSP mapping"""
        return {
            'id': camera.id,
            'name': camera.name,
            'rtspUrl': camera.rtsp_url,
            'location': camera.location,
            'active': camera.active,
            'hlsUrl': camera.hls_url,
            'schedule': camera.current_schedule,
            # Additional debugging info
            'hasValidRTSP': bool(camera.rtsp_url and 'rtsp://' in camera.rtsp_url.lower()),
            'rtspPreview': camera.rtsp_url[:50] + '...' if len(camera.rtsp_url) > 50 else camera.rtsp_url
        }

    def get_camera_by_id(self, camera_id: int) -> Optional[Camera]:
        """Get camera by ID"""
        for camera in self.cameras:
            if camera.id == camera_id:
                return camera
        return None

    def get_active_cameras(self) -> List[Camera]:
        """Get only active cameras"""
        return [cam for cam in self.cameras if cam.active]

    def format_cameras_for_frontend(self) -> List[Dict]:
        """Format cameras data for frontend display"""
        return [self._camera_to_dict(cam) for cam in self.cameras]

    def validate_camera_connection(self, camera: Camera) -> bool:
        """Validate if camera RTSP connection is working"""
        try:
            camera_logger.info(f"🔍 Testing connection for camera {camera.id}: {camera.name}")

            # Test OpenCV connection
            cap = cv2.VideoCapture(camera.rtsp_url)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()

                if ret and frame is not None:
                    camera_logger.info(f"✅ Camera {camera.id} connection successful")
                    return True
                else:
                    camera_logger.warning(f"⚠️ Camera {camera.id} opened but no frame received")
                    return False
            else:
                camera_logger.warning(f"❌ Camera {camera.id} failed to open")
                return False

        except Exception as e:
            camera_logger.error(f"❌ Error testing camera {camera.id}: {e}")
            return False

    def create_video_capture(self, camera: Camera) -> Optional[cv2.VideoCapture]:
        """Create video capture for camera"""
        try:
            camera_logger.info(f"📹 Creating video capture for camera {camera.id}")

            cap = cv2.VideoCapture(camera.rtsp_url)

            # Set buffer size to reduce latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Set timeout
            cap.set(cv2.CAP_PROP_POS_MSEC, config.RTSP_TIMEOUT * 1000)

            if cap.isOpened():
                self.active_streams[camera.id] = cap
                camera_logger.info(f"✅ Video capture created for camera {camera.id}")
                return cap
            else:
                camera_logger.error(f"❌ Failed to create video capture for camera {camera.id}")
                return None

        except Exception as e:
            camera_logger.error(f"❌ Error creating video capture: {e}")
            return None

    def release_video_capture(self, camera_id: int):
        """Release video capture for camera"""
        if camera_id in self.active_streams:
            try:
                self.active_streams[camera_id].release()
                del self.active_streams[camera_id]
                camera_logger.info(f"🔚 Released video capture for camera {camera_id}")
            except Exception as e:
                camera_logger.error(f"❌ Error releasing video capture: {e}")

    def get_camera_stats(self) -> Dict[str, int]:
        """Get camera statistics"""
        stats = {
            'total': len(self.cameras),
            'active': len([cam for cam in self.cameras if cam.active]),
            'inactive': len([cam for cam in self.cameras if not cam.active]),
            'streaming': len(self.active_streams)
        }

        camera_logger.info(f"📊 Camera stats: {stats}")
        return stats

    def search_cameras(self, query: str) -> List[Camera]:
        """Search cameras by name or location"""
        query = query.lower()
        results = []

        for camera in self.cameras:
            if (query in camera.name.lower() or
                    query in camera.location.lower() or
                    query in str(camera.id)):
                results.append(camera)

        camera_logger.info(f"🔍 Search '{query}' found {len(results)} cameras")
        return results

    def add_custom_camera(self, name: str, rtsp_url: str, location: str = "") -> Camera:
        """Add custom camera"""
        camera_id = max([cam.id for cam in self.cameras], default=0) + 1

        camera = Camera(
            id=camera_id,
            name=name,
            rtsp_url=rtsp_url,
            location=location,
            active=True
        )

        self.cameras.append(camera)
        camera_logger.info(f"➕ Added custom camera: {name}")
        return camera

    def remove_camera(self, camera_id: int) -> bool:
        """Remove camera from list"""
        try:
            # Release video capture if active
            self.release_video_capture(camera_id)

            # Remove from list
            self.cameras = [cam for cam in self.cameras if cam.id != camera_id]

            camera_logger.info(f"🗑️ Removed camera {camera_id}")
            return True

        except Exception as e:
            camera_logger.error(f"❌ Error removing camera {camera_id}: {e}")
            return False

    def refresh_cameras(self):
        """Refresh camera list from backend"""
        camera_logger.info("🔄 Refreshing camera list...")

        # Release all active streams
        for camera_id in list(self.active_streams.keys()):
            self.release_video_capture(camera_id)

        # Reload from backend
        return self.load_cameras_from_backend()