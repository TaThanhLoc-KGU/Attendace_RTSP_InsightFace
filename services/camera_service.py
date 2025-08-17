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

        camera_logger.info("Initialized Camera Service")

    def load_cameras_from_backend(self) -> Dict[str, any]:
        """Load cameras từ backend - SIMPLIFIED VERSION"""
        camera_logger.info("Loading cameras from backend...")

        try:
            # Get cameras from backend API
            response = self.backend_api.get_active_cameras()

            if not response.success:
                camera_logger.error(f"Backend API call failed: {response.message}")
                return {
                    'success': False,
                    'message': response.message,
                    'cameras': [],
                    'count': 0
                }

            # Parse response data
            cameras_data = response.data
            camera_logger.info(f"Got response data type: {type(cameras_data)}")

            # Handle different response formats
            if cameras_data is None:
                camera_logger.error("No data received from backend")
                cameras_data = []
            elif isinstance(cameras_data, dict):
                # Check if it's a wrapped response
                if 'content' in cameras_data:
                    cameras_data = cameras_data['content']
                elif 'data' in cameras_data:
                    cameras_data = cameras_data['data']
                elif 'cameras' in cameras_data:
                    cameras_data = cameras_data['cameras']
                elif 'text' in cameras_data:
                    # This means we got HTML instead of JSON
                    camera_logger.error("Received HTML response instead of JSON - authentication issue")
                    return {
                        'success': False,
                        'message': "Authentication failed - received HTML response",
                        'cameras': [],
                        'count': 0
                    }
                else:
                    # Single camera object, convert to list
                    cameras_data = [cameras_data] if cameras_data else []
            elif not isinstance(cameras_data, list):
                camera_logger.error(f"Unexpected data type: {type(cameras_data)}")
                cameras_data = []

            camera_logger.info(f"Processing {len(cameras_data)} camera records...")

            # Clear existing cameras
            self.cameras = []
            successful_count = 0

            # Process each camera
            for i, camera_data in enumerate(cameras_data):
                try:
                    camera_logger.info(f"Processing camera {i+1}/{len(cameras_data)}")

                    # Skip invalid data
                    if not isinstance(camera_data, dict):
                        camera_logger.warning(f"Skipping invalid camera data at index {i}: {type(camera_data)}")
                        continue

                    # Create camera object
                    camera = self._create_camera_from_data(camera_data)
                    if camera:
                        self.cameras.append(camera)
                        successful_count += 1
                        camera_logger.info(f"Successfully added camera: {camera.name}")
                    else:
                        camera_logger.warning(f"Failed to create camera from data at index {i}")

                except Exception as e:
                    camera_logger.error(f"Error processing camera {i}: {e}")
                    continue

            camera_logger.info(f"Successfully loaded {successful_count} cameras out of {len(cameras_data)}")

            return {
                'success': True,
                'cameras': [self._camera_to_dict(cam) for cam in self.cameras],
                'count': len(self.cameras),
                'processed': successful_count,
                'total_received': len(cameras_data)
            }

        except Exception as e:
            camera_logger.error(f"Error loading cameras from backend: {e}")
            return {
                'success': False,
                'message': f"Failed to load cameras: {str(e)}",
                'cameras': [],
                'count': 0
            }

    def _create_camera_from_data(self, camera_data: Dict) -> Optional[Camera]:
        """Create Camera object from backend data"""
        try:
            # Extract camera info with multiple field name possibilities
            camera_id = camera_data.get('id')
            if camera_id is None:
                camera_logger.warning("Camera data missing ID")
                return None

            # Get camera name
            name = (
                camera_data.get('name') or
                camera_data.get('tenCamera') or
                camera_data.get('cameraName') or
                f"Camera {camera_id}"
            )

            # Get RTSP URL
            rtsp_url = (
                camera_data.get('rtspUrl') or
                camera_data.get('rtsp_url') or
                camera_data.get('ipAddress') or
                camera_data.get('url') or
                camera_data.get('streamUrl') or
                ""
            )

            # Get location
            location = (
                camera_data.get('location') or
                camera_data.get('tenPhong') or
                camera_data.get('maPhong') or
                camera_data.get('room') or
                "Unknown Location"
            )

            # Get active status
            active = camera_data.get('active', camera_data.get('isActive', True))

            # Create camera object
            camera = Camera(
                id=camera_id,
                name=name,
                rtsp_url=rtsp_url,
                location=location,
                active=active,
                hls_url=camera_data.get('hlsUrl', ''),
                current_schedule=camera_data.get('currentSchedule')
            )

            # Validate RTSP URL
            if not rtsp_url:
                camera_logger.warning(f"Camera {name} has no RTSP URL")
                return None

            if not ('rtsp://' in rtsp_url.lower() or 'http://' in rtsp_url.lower()):
                camera_logger.warning(f"Camera {name} has invalid RTSP URL: {rtsp_url}")
                return None

            camera_logger.info(f"Created camera: {name} with RTSP: {rtsp_url[:50]}...")
            return camera

        except Exception as e:
            camera_logger.error(f"Error creating camera from data: {e}")
            return None

    def _camera_to_dict(self, camera: Camera) -> Dict:
        """Convert Camera object to dictionary"""
        return {
            'id': camera.id,
            'name': camera.name,
            'rtspUrl': camera.rtsp_url,
            'location': camera.location,
            'active': camera.active,
            'hlsUrl': camera.hls_url,
            'schedule': camera.current_schedule,
            'hasValidRTSP': bool(camera.rtsp_url and ('rtsp://' in camera.rtsp_url.lower() or 'http://' in camera.rtsp_url.lower())),
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
        """Test if camera RTSP connection is working"""
        try:
            camera_logger.info(f"Testing connection for camera {camera.id}: {camera.name}")

            # Test OpenCV connection
            cap = cv2.VideoCapture(camera.rtsp_url)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()

                if ret and frame is not None:
                    camera_logger.info(f"Camera {camera.id} connection successful")
                    return True
                else:
                    camera_logger.warning(f"Camera {camera.id} opened but no frame received")
                    return False
            else:
                camera_logger.warning(f"Camera {camera.id} failed to open")
                return False

        except Exception as e:
            camera_logger.error(f"Error testing camera {camera.id}: {e}")
            return False

    def create_video_capture(self, camera: Camera) -> Optional[cv2.VideoCapture]:
        """Create video capture for camera"""
        try:
            camera_logger.info(f"Creating video capture for camera {camera.id}")

            cap = cv2.VideoCapture(camera.rtsp_url)

            # Set buffer size to reduce latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Set timeout
            cap.set(cv2.CAP_PROP_POS_MSEC, config.RTSP_TIMEOUT * 1000)

            if cap.isOpened():
                self.active_streams[camera.id] = cap
                camera_logger.info(f"Video capture created for camera {camera.id}")
                return cap
            else:
                camera_logger.error(f"Failed to create video capture for camera {camera.id}")
                return None

        except Exception as e:
            camera_logger.error(f"Error creating video capture: {e}")
            return None

    def release_video_capture(self, camera_id: int):
        """Release video capture for camera"""
        if camera_id in self.active_streams:
            try:
                self.active_streams[camera_id].release()
                del self.active_streams[camera_id]
                camera_logger.info(f"Released video capture for camera {camera_id}")
            except Exception as e:
                camera_logger.error(f"Error releasing video capture: {e}")

    def get_camera_stats(self) -> Dict[str, int]:
        """Get camera statistics"""
        stats = {
            'total': len(self.cameras),
            'active': len([cam for cam in self.cameras if cam.active]),
            'inactive': len([cam for cam in self.cameras if not cam.active]),
            'streaming': len(self.active_streams)
        }

        camera_logger.info(f"Camera stats: {stats}")
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

        camera_logger.info(f"Search '{query}' found {len(results)} cameras")
        return results

    def refresh_cameras(self):
        """Refresh camera list from backend"""
        camera_logger.info("Refreshing camera list...")

        # Release all active streams
        for camera_id in list(self.active_streams.keys()):
            self.release_video_capture(camera_id)

        # Reload from backend
        return self.load_cameras_from_backend()

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
        camera_logger.info(f"Added custom camera: {name}")
        return camera

    def remove_camera(self, camera_id: int) -> bool:
        """Remove camera from list"""
        try:
            # Release video capture if active
            self.release_video_capture(camera_id)

            # Remove from list
            self.cameras = [cam for cam in self.cameras if cam.id != camera_id]

            camera_logger.info(f"Removed camera {camera_id}")
            return True

        except Exception as e:
            camera_logger.error(f"Error removing camera {camera_id}: {e}")
            return False