"""
Enhanced Backend API client for Spring Boot integration with Python API endpoints
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
    """Enhanced API client để communicate với Spring Boot backend"""

    def __init__(self, base_url: str = None):
        self.base_url = base_url or config.BACKEND_URL
        self.session = requests.Session()
        self.session.timeout = 30
        self.is_authenticated = False

        # Fixed credentials
        self.username = "admin"
        self.password = "admin@123"

        # Set browser-like headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })

        api_logger.info(f"Initialized Backend API client for {self.base_url}")

    def authenticate(self) -> bool:
        """Perform login authentication"""
        try:
            api_logger.info("Attempting to authenticate...")

            # Step 1: Get login page to establish session
            base_url_without_api = self.base_url.replace('/api', '')
            login_page_url = f"{base_url_without_api}/login"

            api_logger.info(f"Getting login page: {login_page_url}")
            response = self.session.get(login_page_url)

            if response.status_code != 200:
                api_logger.error(f"Cannot access login page: {response.status_code}")
                return False

            # Step 2: Submit login credentials
            login_data = {
                'username': self.username,
                'password': self.password
            }

            # Set form headers
            form_headers = {
                'Content-Type': 'application/x-www-form-urlencoded',
                'Referer': login_page_url,
                'Origin': base_url_without_api
            }

            api_logger.info("Submitting login form...")
            login_response = self.session.post(
                login_page_url,
                data=login_data,
                headers=form_headers,
                allow_redirects=True
            )

            # Check if login was successful
            if login_response.status_code == 200:
                # Look for success indicators in response
                response_text = login_response.text.lower()

                # If we're still on login page, authentication failed
                if 'login' in login_response.url.lower() and any(keyword in response_text for keyword in ['error', 'invalid', 'incorrect']):
                    api_logger.error("Login failed - invalid credentials")
                    return False

                # If we're redirected away from login or see dashboard content
                if '/login' not in login_response.url or any(keyword in response_text for keyword in ['dashboard', 'logout', 'welcome']):
                    api_logger.info("Authentication successful")
                    self.is_authenticated = True
                    return True

            api_logger.error(f"Authentication failed: {login_response.status_code}")
            return False

        except Exception as e:
            api_logger.error(f"Authentication error: {e}")
            return False

    def _make_request(self, method: str, endpoint: str, **kwargs) -> APIResponse:
        """Make HTTP request with error handling"""
        # Ensure authentication for protected endpoints
        if not endpoint.startswith('/auth') and not self.is_authenticated:
            if not self.authenticate():
                return APIResponse(
                    success=False,
                    message="Authentication failed",
                    status_code=401
                )

        url = f"{self.base_url}{endpoint}"

        try:
            api_logger.debug(f"{method} {url}")

            # Set JSON headers for API calls
            if 'headers' not in kwargs:
                kwargs['headers'] = {}

            kwargs['headers'].update({
                'Accept': 'application/json, text/plain, */*',
                'Content-Type': 'application/json',
                'X-Requested-With': 'XMLHttpRequest'
            })

            response = self.session.request(method, url, **kwargs)
            api_logger.debug(f"Response: {response.status_code}")

            # Handle authentication redirect
            if response.status_code == 302 or 'login' in response.url:
                api_logger.warning("Session expired, re-authenticating...")
                self.is_authenticated = False

                if self.authenticate():
                    response = self.session.request(method, url, **kwargs)
                else:
                    return APIResponse(
                        success=False,
                        message="Re-authentication failed",
                        status_code=401
                    )

            if response.status_code == 200:
                try:
                    data = response.json()
                    return APIResponse(
                        success=True,
                        data=data,
                        status_code=response.status_code
                    )
                except ValueError:
                    # Check if response is HTML
                    if 'DOCTYPE html' in response.text or '<html' in response.text:
                        api_logger.error("Received HTML response - authentication required")
                        self.is_authenticated = False
                        return APIResponse(
                            success=False,
                            message="Authentication required",
                            status_code=response.status_code
                        )

                    # Return text response
                    return APIResponse(
                        success=True,
                        data={"text": response.text},
                        status_code=response.status_code
                    )
            else:
                return APIResponse(
                    success=False,
                    message=f"HTTP {response.status_code}: {response.text[:200]}",
                    status_code=response.status_code
                )

        except requests.exceptions.ConnectionError:
            api_logger.error(f"Connection failed to {url}")
            return APIResponse(
                success=False,
                message=f"Connection failed to {url}",
                status_code=0
            )
        except requests.exceptions.Timeout:
            api_logger.error(f"Request timeout to {url}")
            return APIResponse(
                success=False,
                message=f"Request timeout to {url}",
                status_code=0
            )
        except Exception as e:
            api_logger.error(f"Unexpected error: {e}")
            return APIResponse(
                success=False,
                message=f"Unexpected error: {str(e)}",
                status_code=0
            )

    # Camera API endpoints
    def get_cameras(self) -> APIResponse:
        """Lấy danh sách tất cả cameras"""
        api_logger.info("Fetching all cameras...")
        return self._make_request('GET', '/cameras')

    def get_active_cameras(self) -> APIResponse:
        """Lấy danh sách cameras đang hoạt động"""
        api_logger.info("Fetching active cameras...")

        # Try multiple endpoints
        endpoints_to_try = [
            '/cameras',
            '/flask/cameras/active',
            '/cameras/active'
        ]

        for endpoint in endpoints_to_try:
            api_logger.info(f"Trying endpoint: {endpoint}")
            response = self._make_request('GET', endpoint)

            if response.success and response.data:
                # Check if we got actual data (not HTML)
                if isinstance(response.data, list) or (isinstance(response.data, dict) and 'text' not in response.data):
                    api_logger.info(f"Successfully got data from: {endpoint}")
                    return response

        # If all endpoints failed, return the last response
        api_logger.warning("All camera endpoints failed")
        return response

    def get_camera_by_id(self, camera_id: int) -> APIResponse:
        """Lấy thông tin camera theo ID"""
        api_logger.info(f"Fetching camera {camera_id}...")
        return self._make_request('GET', f'/cameras/{camera_id}')

    def update_camera_status(self, camera_id: int, active: bool) -> APIResponse:
        """Cập nhật trạng thái camera"""
        api_logger.info(f"Updating camera {camera_id} status: {active}")
        return self._make_request('PUT', f'/cameras/{camera_id}/status',
                                json={'active': active})

    def test_connection(self) -> APIResponse:
        """Test kết nối đến backend"""
        api_logger.info("Testing backend connection...")

        # Test authentication first
        if not self.authenticate():
            return APIResponse(
                success=False,
                message="Authentication failed"
            )

        # Test camera endpoint
        response = self.get_cameras()
        if response.success:
            return APIResponse(
                success=True,
                message="Connection and authentication successful"
            )
        else:
            return response

    # Python API endpoints for embeddings and students
    def get_all_embeddings(self) -> APIResponse:
        """Lấy tất cả embeddings từ Python API"""
        api_logger.info("Fetching all student embeddings...")
        return self._make_request('GET', '/python/embeddings')

    def get_student_embeddings(self) -> APIResponse:
        """Lấy student embeddings từ Python API endpoint"""
        api_logger.info("Fetching student embeddings from Python API...")
        return self._make_request('GET', '/python/students/embeddings')

    def save_embedding(self, student_id: str, embedding_data: Dict) -> APIResponse:
        """Lưu embedding cho sinh viên"""
        api_logger.info(f"Saving embedding for student {student_id}...")
        return self._make_request('POST', f'/python/students/{student_id}/embedding',
                                json=embedding_data)

    def record_attendance(self, attendance_data: Dict) -> APIResponse:
        """Ghi nhận điểm danh"""
        api_logger.info("Recording attendance...")
        return self._make_request('POST', '/python/attendance', json=attendance_data)

    # Batch operations
    def batch_record_attendance(self, attendance_records: List[Dict]) -> APIResponse:
        """Ghi nhận điểm danh hàng loạt"""
        api_logger.info(f"Batch recording {len(attendance_records)} attendance records...")
        return self._make_request('POST', '/python/attendance/batch',
                                json={'records': attendance_records})

    def batch_save_embeddings(self, embeddings_data: List[Dict]) -> APIResponse:
        """Lưu embeddings hàng loạt"""
        api_logger.info(f"Batch saving {len(embeddings_data)} embeddings...")
        return self._make_request('POST', '/python/embeddings/batch',
                                json={'embeddings': embeddings_data})

    # Additional utility methods
    def update_camera(self, camera_id: int, camera_data: Dict) -> APIResponse:
        """Cập nhật thông tin camera"""
        api_logger.info(f"Updating camera {camera_id}...")
        return self._make_request('PUT', f'/cameras/{camera_id}', json=camera_data)

    def delete_camera(self, camera_id: int) -> APIResponse:
        """Xóa camera"""
        api_logger.info(f"Deleting camera {camera_id}...")
        return self._make_request('DELETE', f'/cameras/{camera_id}')

    def search_cameras(self, query: str) -> APIResponse:
        """Tìm kiếm cameras"""
        api_logger.info(f"Searching cameras: {query}")
        return self._make_request('GET', '/cameras/search', params={'q': query})

    def get_camera_logs(self, camera_id: int, limit: int = 100) -> APIResponse:
        """Lấy logs của camera"""
        api_logger.info(f"Fetching logs for camera {camera_id}...")
        return self._make_request('GET', f'/cameras/{camera_id}/logs',
                                params={'limit': limit})

    def get_camera_stats(self) -> APIResponse:
        """Lấy thống kê cameras"""
        api_logger.info("Fetching camera statistics...")
        return self._make_request('GET', '/cameras/stats')