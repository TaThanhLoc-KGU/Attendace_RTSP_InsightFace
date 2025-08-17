"""
Camera service for managing cameras and RTSP streams
OPTIMIZED VERSION with webcam support and high performance
"""
import cv2
from typing import Dict, List, Optional
from dataclasses import dataclass
from .backend_api import BackendAPI
from config.config import config
from utils.logger import camera_logger


@dataclass
class Camera:
    """Camera data model with webcam support"""
    id: int
    name: str
    rtsp_url: str
    location: str = ""
    active: bool = True
    hls_url: str = ""
    current_schedule: Optional[Dict] = None
    camera_type: str = "rtsp"  # NEW: "rtsp" or "webcam"
    device_index: Optional[int] = None  # NEW: For webcam devices

    def __post_init__(self):
        """Post initialization processing"""
        if not self.hls_url:
            self.hls_url = f"/stream/camera_{self.id}/playlist.m3u8"

        # Determine camera type
        if isinstance(self.rtsp_url, int) or self.rtsp_url.isdigit():
            self.camera_type = "webcam"
            self.device_index = int(self.rtsp_url)
        elif self.rtsp_url.startswith(('http://', 'https://')):
            self.camera_type = "ip"
        else:
            self.camera_type = "rtsp"

    def is_webcam(self) -> bool:
        """Check if this is a webcam"""
        return self.camera_type == "webcam"

    def get_capture_url(self):
        """Get the appropriate capture URL/index"""
        if self.is_webcam():
            return self.device_index or int(self.rtsp_url)
        return self.rtsp_url


class CameraService:
    """High Performance Camera Service with webcam support"""

    def __init__(self, backend_api: BackendAPI):
        self.backend_api = backend_api
        self.cameras: List[Camera] = []
        self.active_streams: Dict[int, cv2.VideoCapture] = {}
        self.webcam_devices: List[Dict] = []

        camera_logger.info("🚀 Initialized High Performance Camera Service")

        # Auto-detect webcams on startup
        if config.ENABLE_WEBCAM_OPTION:
            self.detect_webcam_devices()

    def detect_webcam_devices(self) -> List[Dict]:
        """Detect available webcam devices"""
        camera_logger.info("🔍 Detecting webcam devices...")
        devices = []

        try:
            for i in range(10):  # Check first 10 indices
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    # Test if we can actually read a frame
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # Get basic camera info
                        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                        fps = cap.get(cv2.CAP_PROP_FPS)

                        device = {
                            'id': i,
                            'name': f'Webcam {i}',
                            'device_index': i,
                            'resolution': f"{int(width)}x{int(height)}",
                            'fps': int(fps) if fps > 0 else 30,
                            'type': 'webcam',
                            'url': str(i)  # Store as string for compatibility
                        }
                        devices.append(device)
                        camera_logger.info(f"📹 Found webcam {i}: {device['resolution']} @ {device['fps']}fps")

                    cap.release()

        except Exception as e:
            camera_logger.error(f"❌ Error detecting webcams: {e}")

        self.webcam_devices = devices
        camera_logger.info(f"✅ Detected {len(devices)} webcam devices")
        return devices

    def get_available_webcams(self) -> List[Dict]:
        """Get list of available webcam devices"""
        return self.webcam_devices

    def add_webcam_to_classroom(self, classroom_name: str, webcam_index: int) -> Camera:
        """Add webcam camera for a classroom"""
        try:
            # Generate new camera ID
            camera_id = max([cam.id for cam in self.cameras], default=1000) + 1

            # Create webcam camera
            camera = Camera(
                id=camera_id,
                name=f"Webcam - {classroom_name}",
                rtsp_url=str(webcam_index),  # Store device index as string
                location=classroom_name,
                active=True,
                camera_type="webcam",
                device_index=webcam_index
            )

            # Test webcam before adding
            if self.test_webcam_connection(webcam_index):
                self.cameras.append(camera)
                camera_logger.info(f"✅ Added webcam {webcam_index} for classroom: {classroom_name}")
                return camera
            else:
                camera_logger.error(f"❌ Failed to connect to webcam {webcam_index}")
                raise Exception(f"Cannot connect to webcam {webcam_index}")

        except Exception as e:
            camera_logger.error(f"❌ Error adding webcam: {e}")
            raise

    def test_webcam_connection(self, device_index: int) -> bool:
        """Test webcam connection"""
        try:
            cap = cv2.VideoCapture(device_index)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                return ret and frame is not None
            return False
        except Exception as e:
            camera_logger.error(f"❌ Webcam test error: {e}")
            return False

    def load_cameras_from_backend(self) -> Dict[str, any]:
        """Load cameras from backend - ENHANCED with webcam integration"""
        camera_logger.info("🔄 Loading cameras from backend...")

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

            # Handle different response formats
            if cameras_data is None:
                camera_logger.error("No data received from backend")
                cameras_data = []
            elif isinstance(cameras_data, dict):
                if 'content' in cameras_data:
                    cameras_data = cameras_data['content']
                elif 'data' in cameras_data:
                    cameras_data = cameras_data['data']
                else:
                    cameras_data = []

            if not isinstance(cameras_data, list):
                camera_logger.warning(f"Unexpected data format: {type(cameras_data)}")
                cameras_data = []

            camera_logger.info(f"📊 Processing {len(cameras_data)} cameras from backend")

            # Clear existing cameras
            self.cameras.clear()

            # Process each camera
            successful_count = 0
            for camera_data in cameras_data:
                try:
                    camera = self._create_camera_from_data(camera_data)
                    if camera:
                        self.cameras.append(camera)
                        successful_count += 1
                except Exception as e:
                    camera_logger.error(f"Error creating camera: {e}")

            # Add detected webcams as additional options
            if config.ENABLE_WEBCAM_OPTION:
                self._add_webcam_options()

            camera_logger.info(f"✅ Successfully loaded {successful_count} cameras")
            camera_logger.info(f"📊 Total cameras available: {len(self.cameras)}")

            return {
                'success': True,
                'cameras': [self._camera_to_dict(cam) for cam in self.cameras],
                'count': len(self.cameras),
                'processed': successful_count,
                'total_received': len(cameras_data),
                'webcams_available': len(self.webcam_devices)
            }

        except Exception as e:
            camera_logger.error(f"Error loading cameras from backend: {e}")
            return {
                'success': False,
                'message': f"Failed to load cameras: {str(e)}",
                'cameras': [],
                'count': 0
            }

    def _add_webcam_options(self):
        """Add webcam devices as camera options"""
        for webcam in self.webcam_devices:
            # Create pseudo-camera for each webcam
            camera_id = 9000 + webcam['id']  # Use high IDs to avoid conflicts

            camera = Camera(
                id=camera_id,
                name=f"🎥 {webcam['name']} ({webcam['resolution']})",
                rtsp_url=str(webcam['device_index']),
                location="Webcam Device",
                active=True,
                camera_type="webcam",
                device_index=webcam['device_index']
            )

            self.cameras.append(camera)
            camera_logger.info(f"📹 Added webcam option: {camera.name}")

    def _create_camera_from_data(self, camera_data: Dict) -> Optional[Camera]:
        """Create Camera object from backend data"""
        try:
            # Extract camera info
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

            # Validate URL for RTSP cameras
            if not camera.is_webcam():
                if not rtsp_url:
                    camera_logger.warning(f"Camera {name} has no RTSP URL")
                    return None

                if not ('rtsp://' in rtsp_url.lower() or 'http://' in rtsp_url.lower()):
                    camera_logger.warning(f"Camera {name} has invalid RTSP URL: {rtsp_url}")
                    return None

            camera_logger.info(f"✅ Created camera: {name} ({camera.camera_type})")
            return camera

        except Exception as e:
            camera_logger.error(f"Error creating camera from data: {e}")
            return None

    def _camera_to_dict(self, camera: Camera) -> Dict:
        """Convert Camera object to dictionary with webcam support"""
        base_dict = {
            'id': camera.id,
            'name': camera.name,
            'rtspUrl': camera.rtsp_url,
            'location': camera.location,
            'active': camera.active,
            'hlsUrl': camera.hls_url,
            'schedule': camera.current_schedule,
            'camera_type': camera.camera_type,
            'is_webcam': camera.is_webcam()
        }

        if camera.is_webcam():
            base_dict.update({
                'device_index': camera.device_index,
                'rtspPreview': f"Webcam Device {camera.device_index}",
                'hasValidRTSP': True  # Webcams are always "valid"
            })
        else:
            base_dict.update({
                'hasValidRTSP': bool(camera.rtsp_url and ('rtsp://' in camera.rtsp_url.lower() or 'http://' in camera.rtsp_url.lower())),
                'rtspPreview': camera.rtsp_url[:50] + '...' if len(camera.rtsp_url) > 50 else camera.rtsp_url
            })

        return base_dict

    def create_video_capture(self, camera: Camera) -> Optional[cv2.VideoCapture]:
        """Create optimized video capture for camera"""
        try:
            camera_logger.info(f"🎬 Creating video capture for {camera.name} ({camera.camera_type})")

            # Get capture source
            capture_source = camera.get_capture_url()
            cap = cv2.VideoCapture(capture_source)

            if not cap.isOpened():
                camera_logger.error(f"❌ Failed to open camera {camera.id}")
                return None

            # PERFORMANCE OPTIMIZATIONS
            # Critical: Set minimal buffer size for low latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, config.CAMERA_BUFFER_SIZE)

            # Set FPS based on camera type
            if camera.is_webcam():
                cap.set(cv2.CAP_PROP_FPS, config.WEBCAM_FPS_TARGET)
                # Set webcam resolution for better performance
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.WEBCAM_PROCESSING_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.WEBCAM_PROCESSING_HEIGHT)
            else:
                cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
                # Set timeout for RTSP
                cap.set(cv2.CAP_PROP_POS_MSEC, config.RTSP_TIMEOUT)

            # Additional performance settings
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # Use MJPEG for speed

            # Test frame reading
            ret, frame = cap.read()
            if not ret:
                camera_logger.error(f"❌ Cannot read frame from camera {camera.id}")
                cap.release()
                return None

            # Store active stream
            self.active_streams[camera.id] = cap
            camera_logger.info(f"✅ Video capture ready for {camera.name}")

            return cap

        except Exception as e:
            camera_logger.error(f"❌ Error creating video capture: {e}")
            return None

    def validate_camera_connection(self, camera: Camera) -> bool:
        """Test camera connection with optimizations"""
        try:
            camera_logger.info(f"🔍 Testing connection for {camera.name} ({camera.camera_type})")

            # Create temporary capture
            capture_source = camera.get_capture_url()
            cap = cv2.VideoCapture(capture_source)

            if cap.isOpened():
                # Set minimal timeout for test
                if not camera.is_webcam():
                    cap.set(cv2.CAP_PROP_POS_MSEC, 3000)  # 3 second timeout

                ret, frame = cap.read()
                cap.release()

                if ret and frame is not None:
                    camera_logger.info(f"✅ {camera.name} connection successful")
                    return True
                else:
                    camera_logger.warning(f"⚠️ {camera.name} opened but no frame received")
                    return False
            else:
                camera_logger.warning(f"❌ {camera.name} failed to open")
                return False

        except Exception as e:
            camera_logger.error(f"❌ Error testing {camera.name}: {e}")
            return False

    def get_camera_by_id(self, camera_id: int) -> Optional[Camera]:
        """Get camera by ID"""
        for camera in self.cameras:
            if camera.id == camera_id:
                return camera
        return None

    def get_active_cameras(self) -> List[Camera]:
        """Get only active cameras"""
        return [cam for cam in self.cameras if cam.active]

    def get_rtsp_cameras(self) -> List[Camera]:
        """Get only RTSP cameras"""
        return [cam for cam in self.cameras if not cam.is_webcam()]

    def get_webcam_cameras(self) -> List[Camera]:
        """Get only webcam cameras"""
        return [cam for cam in self.cameras if cam.is_webcam()]

    def format_cameras_for_frontend(self) -> List[Dict]:
        """Format cameras data for frontend display"""
        return [self._camera_to_dict(cam) for cam in self.cameras]

    def release_video_capture(self, camera_id: int):
        """Release video capture for camera"""
        if camera_id in self.active_streams:
            try:
                self.active_streams[camera_id].release()
                del self.active_streams[camera_id]
                camera_logger.info(f"🔄 Released video capture for camera {camera_id}")
            except Exception as e:
                camera_logger.error(f"❌ Error releasing video capture: {e}")

    def get_camera_stats(self) -> Dict[str, int]:
        """Get camera statistics"""
        stats = {
            'total': len(self.cameras),
            'active': len([cam for cam in self.cameras if cam.active]),
            'inactive': len([cam for cam in self.cameras if not cam.active]),
            'streaming': len(self.active_streams),
            'rtsp_cameras': len(self.get_rtsp_cameras()),
            'webcam_cameras': len(self.get_webcam_cameras()),
            'available_webcams': len(self.webcam_devices)
        }

        camera_logger.info(f"📊 Camera stats: {stats}")
        return stats

    def refresh_cameras(self):
        """Refresh camera list from backend and webcams"""
        camera_logger.info("🔄 Refreshing camera list...")

        # Release all active streams
        for camera_id in list(self.active_streams.keys()):
            self.release_video_capture(camera_id)

        # Re-detect webcams
        if config.ENABLE_WEBCAM_OPTION:
            self.detect_webcam_devices()

        # Reload from backend
        return self.load_cameras_from_backend()