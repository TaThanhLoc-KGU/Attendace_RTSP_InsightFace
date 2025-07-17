"""
Backend API client for Spring Boot integration
"""
import requests
from typing import List, Dict, Optional
from dataclasses import dataclass
from config.config import config
from utils.logger import api_logger


@dataclass
class APIResponse:
    """API response wrapper"""
    success: bool
    data: Optional[Dict] = None
    message: str = ""
    status_code: int = 0


class BackendAPI:
    """API client để communicate với Spring Boot backend"""

    def __init__(self, base_url: str = None):
        self.base_url = base_url or config.BACKEND_URL
        self.session = requests.Session()
        self.session.timeout = 10

        # Set default headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })

        api_logger.info(f"🔗 Initialized Backend API client for {self.base_url}")

    def _make_request(self, method: str, endpoint: str, **kwargs) -> APIResponse:
        """Make HTTP request with error handling"""
        url = f"{self.base_url}{endpoint}"

        try:
            api_logger.debug(f"📡 {method} {url}")
            response = self.session.request(method, url, **kwargs)

            # Log response
            api_logger.debug(f"📨 Response: {response.status_code}")

            if response.status_code == 200:
                try:
                    data = response.json()
                    return APIResponse(
                        success=True,
                        data=data,
                        status_code=response.status_code
                    )
                except ValueError:
                    # Response không phải JSON
                    return APIResponse(
                        success=True,
                        data={"text": response.text},
                        status_code=response.status_code
                    )
            else:
                return APIResponse(
                    success=False,
                    message=f"HTTP {response.status_code}: {response.text}",
                    status_code=response.status_code
                )

        except requests.exceptions.ConnectionError:
            api_logger.error(f"❌ Connection failed to {url}")
            return APIResponse(
                success=False,
                message=f"Connection failed to {url}",
                status_code=0
            )
        except requests.exceptions.Timeout:
            api_logger.error(f"⏱️ Request timeout to {url}")
            return APIResponse(
                success=False,
                message=f"Request timeout to {url}",
                status_code=0
            )
        except Exception as e:
            api_logger.error(f"❌ Unexpected error: {e}")
            return APIResponse(
                success=False,
                message=f"Unexpected error: {str(e)}",
                status_code=0
            )

    def get_cameras(self) -> APIResponse:
        """Lấy danh sách tất cả cameras"""
        api_logger.info("📹 Fetching all cameras...")
        return self._make_request('GET', '/cameras')

    def get_active_cameras(self) -> APIResponse:
        """Lấy danh sách cameras đang hoạt động"""
        api_logger.info("📹 Fetching active cameras...")

        # Thử endpoint Flask integration trước
        response = self._make_request('GET', '/flask/cameras/active')

        if response.success:
            return response

        # Fallback về endpoint chung
        api_logger.warning("Flask endpoint failed, trying general endpoint...")
        return self.get_cameras()

    def get_camera_by_id(self, camera_id: int) -> APIResponse:
        """Lấy thông tin camera theo ID"""
        api_logger.info(f"📹 Fetching camera {camera_id}...")
        return self._make_request('GET', f'/cameras/{camera_id}')

    def update_camera_status(self, camera_id: int, active: bool) -> APIResponse:
        """Cập nhật trạng thái camera"""
        api_logger.info(f"📹 Updating camera {camera_id} status: {active}")
        return self._make_request('PUT', f'/cameras/{camera_id}/status',
                                  json={'active': active})

    def test_connection(self) -> APIResponse:
        """Test kết nối đến backend"""
        api_logger.info("🔍 Testing backend connection...")
        return self._make_request('GET', '/health')

    def get_camera_stats(self) -> APIResponse:
        """Lấy thống kê cameras"""
        api_logger.info("📊 Fetching camera statistics...")
        return self._make_request('GET', '/cameras/stats')

    def validate_rtsp_url(self, rtsp_url: str) -> APIResponse:
        """Validate RTSP URL"""
        api_logger.info(f"🔍 Validating RTSP URL: {rtsp_url[:30]}...")
        return self._make_request('POST', '/cameras/validate-rtsp',
                                  json={'rtspUrl': rtsp_url})

    def get_camera_schedule(self, camera_id: int) -> APIResponse:
        """Lấy lịch trình camera"""
        api_logger.info(f"📅 Fetching schedule for camera {camera_id}...")
        return self._make_request('GET', f'/cameras/{camera_id}/schedule')

    def create_camera(self, camera_data: Dict) -> APIResponse:
        """Tạo camera mới"""
        api_logger.info(f"➕ Creating new camera: {camera_data.get('name', 'Unknown')}")
        return self._make_request('POST', '/cameras', json=camera_data)

    def update_camera(self, camera_id: int, camera_data: Dict) -> APIResponse:
        """Cập nhật thông tin camera"""
        api_logger.info(f"✏️ Updating camera {camera_id}...")
        return self._make_request('PUT', f'/cameras/{camera_id}', json=camera_data)

    def delete_camera(self, camera_id: int) -> APIResponse:
        """Xóa camera"""
        api_logger.info(f"🗑️ Deleting camera {camera_id}...")
        return self._make_request('DELETE', f'/cameras/{camera_id}')

    def search_cameras(self, query: str) -> APIResponse:
        """Tìm kiếm cameras"""
        api_logger.info(f"🔍 Searching cameras: {query}")
        return self._make_request('GET', '/cameras/search', params={'q': query})

    def get_camera_logs(self, camera_id: int, limit: int = 100) -> APIResponse:
        """Lấy logs của camera"""
        api_logger.info(f"📝 Fetching logs for camera {camera_id}...")
        return self._make_request('GET', f'/cameras/{camera_id}/logs',
                                  params={'limit': limit})