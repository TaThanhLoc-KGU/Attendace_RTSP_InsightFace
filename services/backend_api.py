"""
Fixed Backend API with correct camera endpoints
"""
import requests
from typing import Dict, Optional, List
from dataclasses import dataclass
from config.config import config
from utils.logger import api_logger
import time

@dataclass
class APIResponse:
    """API response wrapper"""
    success: bool
    data: Optional[Dict] = None
    message: str = ""
    status_code: int = 0

class BackendAPI:
    """Backend API client with CORRECT endpoints"""

    def __init__(self, base_url: str = None):
        self.base_url = base_url or config.BACKEND_URL
        self.session = requests.Session()
        self.session.timeout = 30
        self.is_authenticated = False

        # Use credentials from config
        self.username = config.BACKEND_USERNAME  # admin
        self.password = config.BACKEND_PASSWORD  # admin@123

        # Set proper headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/html, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive'
        })

        api_logger.info(f"🔗 Initialized Backend API client for {self.base_url}")

        # Auto-authenticate if enabled
        if config.AUTO_LOGIN:
            self._auto_authenticate()

    def _auto_authenticate(self):
        """Auto authenticate on initialization"""
        try:
            if self.authenticate():
                api_logger.info("✅ Auto-authentication successful")
            else:
                api_logger.warning("⚠️ Auto-authentication failed")
        except Exception as e:
            api_logger.error(f"❌ Auto-authentication error: {e}")

    def authenticate(self) -> bool:
        """Perform web-based login authentication"""
        try:
            api_logger.info("🔐 Starting authentication...")

            # Get base URL without /api
            base_url_without_api = self.base_url.replace('/api', '')
            login_url = f"{base_url_without_api}/login"

            # Get login page
            login_page = self.session.get(login_url, allow_redirects=True)

            if login_page.status_code != 200:
                api_logger.error(f"❌ Cannot access login page: {login_page.status_code}")
                return False

            # Prepare login data
            login_data = {
                'username': self.username,
                'password': self.password
            }

            # Submit login
            login_response = self.session.post(
                login_url,
                data=login_data,
                headers={
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'Referer': login_url,
                    'Origin': base_url_without_api
                },
                allow_redirects=True
            )

            # Check authentication success
            if login_response.status_code == 200:
                response_text = login_response.text.lower()

                # If we're still on login page with errors, auth failed
                if '/login' in login_response.url and any(err in response_text for err in ['error', 'invalid', 'incorrect']):
                    api_logger.error("❌ Authentication failed - invalid credentials")
                    return False

                # Check for success indicators
                if any(indicator in response_text for indicator in ['dashboard', 'logout', 'admin', 'menu']) or '/login' not in login_response.url:
                    self.is_authenticated = True
                    api_logger.info("✅ Authentication successful")
                    return True

            api_logger.error(f"❌ Authentication failed: {login_response.status_code}")
            return False

        except Exception as e:
            api_logger.error(f"❌ Authentication error: {e}")
            return False

    def _make_request(self, method: str, endpoint: str, **kwargs) -> APIResponse:
        """Make HTTP request with authentication"""
        # Ensure authentication
        if not self.is_authenticated:
            if not self.authenticate():
                return APIResponse(
                    success=False,
                    message="Authentication failed",
                    status_code=401
                )

        url = f"{self.base_url}{endpoint}"

        try:
            # Set JSON headers for API calls
            if 'headers' not in kwargs:
                kwargs['headers'] = {}

            kwargs['headers'].update({
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            })

            response = self.session.request(method, url, **kwargs)

            # Handle authentication redirect
            if response.status_code == 302 or '/login' in response.url:
                api_logger.warning("🔄 Session expired, re-authenticating...")
                self.is_authenticated = False

                if self.authenticate():
                    response = self.session.request(method, url, **kwargs)
                else:
                    return APIResponse(
                        success=False,
                        message="Re-authentication failed",
                        status_code=401
                    )

            # Handle successful response
            if response.status_code == 200:
                try:
                    data = response.json()
                    api_logger.debug(f"✅ Success: {method} {endpoint}")
                    return APIResponse(
                        success=True,
                        data=data,
                        status_code=response.status_code
                    )
                except ValueError:
                    # Check if HTML response (auth issue)
                    if any(tag in response.text.lower() for tag in ['<html', '<!doctype']):
                        api_logger.error("❌ Received HTML - authentication required")
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

            # Handle error responses
            else:
                try:
                    error_data = response.json()
                    error_msg = error_data.get('message', f"HTTP {response.status_code}")
                except:
                    error_msg = f"HTTP {response.status_code}"

                api_logger.error(f"❌ API error: {error_msg}")
                return APIResponse(
                    success=False,
                    message=error_msg,
                    status_code=response.status_code
                )

        except requests.exceptions.ConnectionError:
            api_logger.error("❌ Connection error - is backend running?")
            return APIResponse(
                success=False,
                message="Connection error - is backend running?",
                status_code=503
            )
        except Exception as e:
            api_logger.error(f"❌ Request error: {e}")
            return APIResponse(
                success=False,
                message=str(e),
                status_code=500
            )

    # FIXED: Camera API endpoints with correct paths
    def get_cameras(self) -> APIResponse:
        """Get all cameras - CORRECT endpoint"""
        api_logger.info("📥 Fetching all cameras...")
        return self._make_request('GET', '/cameras')

    def get_active_cameras(self) -> APIResponse:
        """Get active cameras - TRYING multiple endpoints"""
        api_logger.info("📥 Fetching active cameras...")

        # Try different possible endpoints
        endpoints_to_try = [
            '/cameras',           # Basic endpoint - filter active on client side
            '/flask/cameras/active', # Flask integration endpoint
            '/cameras/active',    # If this endpoint exists
        ]

        for endpoint in endpoints_to_try:
            api_logger.info(f"🔍 Trying endpoint: {endpoint}")
            response = self._make_request('GET', endpoint)

            if response.success and response.data:
                # Check if we got valid camera data
                data = response.data

                if isinstance(data, list):
                    # Filter active cameras if needed
                    if endpoint == '/cameras':
                        # Filter active cameras client-side
                        active_cameras = [cam for cam in data if cam.get('active', True)]
                        api_logger.info(f"✅ Got {len(active_cameras)} active cameras from {len(data)} total")
                        return APIResponse(
                            success=True,
                            data=active_cameras,
                            status_code=response.status_code
                        )
                    else:
                        api_logger.info(f"✅ Successfully got {len(data)} cameras from {endpoint}")
                        return response

                elif isinstance(data, dict) and not data.get('text'):
                    # Might be wrapped response
                    api_logger.info(f"✅ Got data from {endpoint}")
                    return response

            api_logger.warning(f"⚠️ Endpoint {endpoint} failed or returned no data")

        # If all failed, return error
        api_logger.error("❌ All camera endpoints failed")
        return APIResponse(
            success=False,
            message="No working camera endpoint found",
            status_code=404
        )

    def get_camera_by_id(self, camera_id: int) -> APIResponse:
        """Get camera by ID"""
        api_logger.info(f"📥 Fetching camera {camera_id}...")
        return self._make_request('GET', f'/cameras/{camera_id}')

    # FIXED: Student embeddings endpoints
    def get_all_embeddings(self) -> APIResponse:
        """Get all student embeddings - CORRECT endpoint"""
        api_logger.info("📥 Fetching all student embeddings...")
        return self._make_request('GET', '/python/embeddings')

    def get_student_embeddings_alt(self) -> APIResponse:
        """Alternative endpoint for student embeddings"""
        api_logger.info("📥 Trying alternative student embeddings endpoint...")
        return self._make_request('GET', '/sinhvien/embeddings')

    # Test endpoints to debug
    def test_endpoints(self) -> Dict[str, bool]:
        """Test various endpoints to see what works"""
        api_logger.info("🧪 Testing available endpoints...")

        endpoints_to_test = [
            '/cameras',
            '/cameras/active',
            '/flask/cameras/active',
            '/python/embeddings',
            '/sinhvien/embeddings',
            '/cameras/count'
        ]

        results = {}

        for endpoint in endpoints_to_test:
            try:
                response = self._make_request('GET', endpoint)
                results[endpoint] = response.success
                status = "✅ WORKS" if response.success else "❌ FAILS"
                api_logger.info(f"  {endpoint}: {status}")
            except Exception as e:
                results[endpoint] = False
                api_logger.info(f"  {endpoint}: ❌ ERROR - {e}")

        return results

    def test_connection(self) -> APIResponse:
        """Test backend connection with endpoint testing"""
        api_logger.info("🔍 Testing backend connection...")

        if not self.authenticate():
            return APIResponse(
                success=False,
                message="Authentication failed"
            )

        # Test endpoints
        working_endpoints = self.test_endpoints()

        if any(working_endpoints.values()):
            working_list = [ep for ep, works in working_endpoints.items() if works]
            return APIResponse(
                success=True,
                message=f"Connection successful. Working endpoints: {working_list}",
                data={"working_endpoints": working_list}
            )
        else:
            return APIResponse(
                success=False,
                message="Authentication successful but no API endpoints working"
            )

    # Additional methods
    def record_attendance(self, attendance_data: Dict) -> APIResponse:
        """Record attendance"""
        api_logger.info(f"📝 Recording attendance...")
        return self._make_request('POST', '/python/attendance', json=attendance_data)

    def save_embedding(self, student_id: str, embedding_data: Dict) -> APIResponse:
        """Save embedding for student"""
        api_logger.info(f"💾 Saving embedding for student {student_id}...")
        return self._make_request('POST', f'/python/students/{student_id}/embedding', json=embedding_data)

    def get_camera_count(self) -> APIResponse:
        """Get camera statistics"""
        api_logger.info("📊 Fetching camera count...")
        return self._make_request('GET', '/cameras/count')