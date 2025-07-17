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

    # Camera API endpoints
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
        # Try camera endpoint as health check
        response = self._make_request('GET', '/cameras')
        if response.success:
            return APIResponse(success=True, message="Connection successful")
        else:
            return response

    # Python API endpoints for embeddings and students
    def get_all_embeddings(self) -> APIResponse:
        """Lấy tất cả embeddings từ Python API"""
        api_logger.info("🎓 Fetching all student embeddings...")
        return self._make_request('GET', '/python/embeddings')

    def get_student_embedding(self, student_id: str) -> APIResponse:
        """Lấy embedding của một sinh viên"""
        api_logger.info(f"🎓 Fetching embedding for student {student_id}...")
        return self._make_request('GET', f'/python/students/{student_id}/embedding')

    def save_student_embedding(self, student_id: str, embedding_data) -> APIResponse:
        """Lưu embedding cho sinh viên"""
        api_logger.info(f"💾 Saving embedding for student {student_id}...")
        return self._make_request('POST', f'/python/students/{student_id}/embedding',
                                json={'embedding': embedding_data})

    def record_attendance(self, student_id: str, camera_id: int) -> APIResponse:
        """Ghi nhận điểm danh"""
        api_logger.info(f"📝 Recording attendance for {student_id} at camera {camera_id}...")
        return self._make_request('POST', '/python/attendance',
                                json={
                                    'studentId': student_id,
                                    'cameraId': camera_id
                                })

    # Student management endpoints
    def get_all_students(self) -> APIResponse:
        """Lấy danh sách tất cả sinh viên"""
        api_logger.info("👥 Fetching all students...")
        return self._make_request('GET', '/sinhvien')

    def get_student_by_id(self, student_id: str) -> APIResponse:
        """Lấy thông tin sinh viên theo ID"""
        api_logger.info(f"👤 Fetching student {student_id}...")
        return self._make_request('GET', f'/sinhvien/{student_id}')

    def search_students(self, query: str) -> APIResponse:
        """Tìm kiếm sinh viên"""
        api_logger.info(f"🔍 Searching students with query: {query}")
        return self._make_request('GET', '/sinhvien/search', params={'q': query})

    # Attendance endpoints
    def get_attendance_history(self, student_id: str = None, camera_id: int = None,
                             start_date: str = None, end_date: str = None) -> APIResponse:
        """Lấy lịch sử điểm danh"""
        api_logger.info("📊 Fetching attendance history...")

        params = {}
        if student_id:
            params['studentId'] = student_id
        if camera_id:
            params['cameraId'] = camera_id
        if start_date:
            params['startDate'] = start_date
        if end_date:
            params['endDate'] = end_date

        return self._make_request('GET', '/diemdanh', params=params)

    def get_attendance_statistics(self) -> APIResponse:
        """Lấy thống kê điểm danh"""
        api_logger.info("📈 Fetching attendance statistics...")
        return self._make_request('GET', '/diemdanh/statistics')

    def get_daily_attendance(self, date: str) -> APIResponse:
        """Lấy điểm danh theo ngày"""
        api_logger.info(f"📅 Fetching attendance for date: {date}")
        return self._make_request('GET', f'/diemdanh/daily/{date}')

    # Class management endpoints
    def get_all_classes(self) -> APIResponse:
        """Lấy danh sách tất cả lớp"""
        api_logger.info("🏫 Fetching all classes...")
        return self._make_request('GET', '/lop')

    def get_class_students(self, class_id: str) -> APIResponse:
        """Lấy danh sách sinh viên theo lớp"""
        api_logger.info(f"🏫 Fetching students for class {class_id}...")
        return self._make_request('GET', f'/lop/{class_id}/students')

    # System health and monitoring
    def get_system_health(self) -> APIResponse:
        """Kiểm tra system health"""
        api_logger.info("🏥 Checking system health...")
        return self._make_request('GET', '/actuator/health')

    def get_system_info(self) -> APIResponse:
        """Lấy thông tin hệ thống"""
        api_logger.info("ℹ️ Fetching system info...")
        return self._make_request('GET', '/actuator/info')

    def get_system_metrics(self) -> APIResponse:
        """Lấy system metrics"""
        api_logger.info("📊 Fetching system metrics...")
        return self._make_request('GET', '/actuator/metrics')

    # Utility methods
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

    def get_camera_stats(self) -> APIResponse:
        """Lấy thống kê cameras"""
        api_logger.info("📊 Fetching camera statistics...")
        return self._make_request('GET', '/cameras/stats')

    # Batch operations
    def batch_record_attendance(self, attendance_records: List[Dict]) -> APIResponse:
        """Ghi nhận điểm danh hàng loạt"""
        api_logger.info(f"📝 Batch recording {len(attendance_records)} attendance records...")
        return self._make_request('POST', '/python/attendance/batch',
                                json={'records': attendance_records})

    def batch_save_embeddings(self, embeddings_data: List[Dict]) -> APIResponse:
        """Lưu embeddings hàng loạt"""
        api_logger.info(f"💾 Batch saving {len(embeddings_data)} embeddings...")
        return self._make_request('POST', '/python/embeddings/batch',
                                json={'embeddings': embeddings_data})

    # Export/Import functionality
    def export_attendance_data(self, start_date: str, end_date: str, format: str = 'csv') -> APIResponse:
        """Export attendance data"""
        api_logger.info(f"📤 Exporting attendance data ({format}) from {start_date} to {end_date}")
        return self._make_request('GET', '/diemdanh/export',
                                params={
                                    'startDate': start_date,
                                    'endDate': end_date,
                                    'format': format
                                })

    def export_student_data(self, format: str = 'csv') -> APIResponse:
        """Export student data"""
        api_logger.info(f"📤 Exporting student data ({format})")
        return self._make_request('GET', '/sinhvien/export',
                                params={'format': format})

    # Advanced features
    def get_recognition_accuracy(self, start_date: str = None, end_date: str = None) -> APIResponse:
        """Lấy độ chính xác nhận diện"""
        api_logger.info("📊 Fetching recognition accuracy...")
        params = {}
        if start_date:
            params['startDate'] = start_date
        if end_date:
            params['endDate'] = end_date

        return self._make_request('GET', '/python/recognition/accuracy', params=params)

    def get_system_performance(self) -> APIResponse:
        """Lấy hiệu suất hệ thống"""
        api_logger.info("⚡ Fetching system performance...")
        return self._make_request('GET', '/python/system/performance')

    def trigger_embedding_sync(self) -> APIResponse:
        """Trigger sync embeddings"""
        api_logger.info("🔄 Triggering embedding sync...")
        return self._make_request('POST', '/python/embeddings/sync')

    def get_embedding_statistics(self) -> APIResponse:
        """Lấy thống kê embeddings"""
        api_logger.info("📊 Fetching embedding statistics...")
        return self._make_request('GET', '/python/embeddings/statistics')

    # Configuration endpoints
    def get_system_config(self) -> APIResponse:
        """Lấy cấu hình hệ thống"""
        api_logger.info("⚙️ Fetching system configuration...")
        return self._make_request('GET', '/config')

    def update_system_config(self, config_data: Dict) -> APIResponse:
        """Cập nhật cấu hình hệ thống"""
        api_logger.info("⚙️ Updating system configuration...")
        return self._make_request('PUT', '/config', json=config_data)

    def get_recognition_settings(self) -> APIResponse:
        """Lấy cài đặt nhận diện"""
        api_logger.info("🔍 Fetching recognition settings...")
        return self._make_request('GET', '/python/recognition/settings')

    def update_recognition_settings(self, settings: Dict) -> APIResponse:
        """Cập nhật cài đặt nhận diện"""
        api_logger.info("🔍 Updating recognition settings...")
        return self._make_request('PUT', '/python/recognition/settings', json=settings)