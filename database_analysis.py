#!/usr/bin/env python3
"""
API Debug Script with Authentication
Kiểm tra API với login admin/admin@123
"""
import requests
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.config import config


class APIDebugger:
    def __init__(self):
        # Sử dụng BACKEND_URL (đã có /api)
        self.base_url = config.BACKEND_URL .rstrip('/')  # http://localhost:8080/api
        print(f"🔍 Using BACKEND_URL: {self.base_url}")

        self.session = requests.Session()
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'DebugScript/1.0'
        }

    def login(self, username="admin", password="admin@123"):
        """Login để lấy authentication token/session"""
        print(f"🔐 Attempting login with {username}...")

        # Thử các endpoint login khác nhau
        login_endpoints = [
            '/auth/login',
            '/login',
            '/api/auth/login',
            '/authentication/login'
        ]

        login_data = {
            "username": username,
            "password": password
        }

        for endpoint in login_endpoints:
            try:
                # Thử với base URL trực tiếp (không có /api)
                base_without_api = self.base_url.replace('/api', '')
                login_url = f"{base_without_api}{endpoint}"

                print(f"  🌐 Trying: {login_url}")

                response = self.session.post(login_url, json=login_data, headers=self.headers, timeout=10)

                print(f"    Status: {response.status_code}")

                if response.status_code == 200:
                    print(f"    ✅ Login successful!")

                    # Check if we got a token
                    try:
                        resp_data = response.json()
                        if 'token' in resp_data:
                            token = resp_data['token']
                            self.headers['Authorization'] = f'Bearer {token}'
                            print(f"    🎫 Got token: {token[:20]}...")
                        elif 'accessToken' in resp_data:
                            token = resp_data['accessToken']
                            self.headers['Authorization'] = f'Bearer {token}'
                            print(f"    🎫 Got accessToken: {token[:20]}...")
                    except:
                        pass

                    # Session cookies should be maintained automatically
                    return True
                elif response.status_code == 302:
                    print(f"    🔄 Redirect - possible success")
                    return True

            except Exception as e:
                print(f"    ❌ Error: {e}")

        print("⚠️ All login attempts failed, trying without authentication...")
        return False

    def test_endpoint(self, endpoint, with_auth=False):
        """Test một endpoint cụ thể"""
        print(f"\n{'=' * 60}")
        print(f"🌐 Testing: {endpoint}")
        if with_auth:
            print("🔐 With authentication")
        print(f"{'=' * 60}")

        try:
            # BACKEND_URL đã có /api, chỉ cần thêm endpoint
            url = f"{self.base_url}{endpoint}"
            print(f"📡 Full URL: {url}")

            # Use session for authentication
            if with_auth:
                response = self.session.get(url, headers=self.headers, timeout=30)
            else:
                response = requests.get(url, headers=self.headers, timeout=30)

            print(f"📊 Status Code: {response.status_code}")
            print(f"🔤 Content Type: {response.headers.get('content-type', 'Unknown')}")
            print(f"📏 Content Length: {len(response.content)}")

            # Show response preview
            content = response.text
            print(f"📄 Response Preview (first 300 chars):")
            print("-" * 50)
            print(content[:300])
            if len(content) > 300:
                print("... (truncated)")
            print("-" * 50)

            # Analyze response
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')

                if 'application/json' in content_type:
                    try:
                        json_data = response.json()
                        print(f"✅ JSON Parse: SUCCESS")
                        print(f"📊 Data Type: {type(json_data)}")

                        if isinstance(json_data, list):
                            print(f"📝 Array Length: {len(json_data)}")
                            if len(json_data) > 0:
                                sample = json_data[0]
                                print(
                                    f"📋 Sample Item Keys: {list(sample.keys()) if isinstance(sample, dict) else 'Not a dict'}")

                                # Special handling for embeddings endpoint
                                if 'embedding' in endpoint and isinstance(sample, dict):
                                    print(f"📋 Sample Item: {sample}")

                                    if 'embedding' in sample:
                                        emb = sample['embedding']
                                        if isinstance(emb, str):
                                            print(f"📊 Embedding format: {type(emb)} - length {len(emb)}")
                                            print(f"📊 Embedding preview: {emb[:50]}...")
                                        else:
                                            print(f"📊 Embedding type: {type(emb)}")

                        elif isinstance(json_data, dict):
                            print(f"📋 Object Keys: {list(json_data.keys())}")

                        return True, json_data

                    except json.JSONDecodeError as e:
                        print(f"❌ JSON Parse FAILED: {e}")
                        return False, None

                elif 'text/html' in content_type:
                    print(f"⚠️ Got HTML instead of JSON (probably login page or error)")
                    if "login" in content.lower():
                        print("💡 This endpoint requires authentication")
                    return False, None

                else:
                    print(f"⚠️ Unexpected content type: {content_type}")
                    return False, None

            elif response.status_code == 302:
                print("🔄 Redirect - might need authentication")
                return False, None
            elif response.status_code == 401:
                print("🔐 Unauthorized - authentication required")
                return False, None
            elif response.status_code == 403:
                print("🚫 Forbidden - access denied")
                return False, None
            elif response.status_code == 404:
                print("❌ Not Found - endpoint doesn't exist")
                return False, None
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                return False, None

        except requests.exceptions.ConnectionError:
            print(f"❌ CONNECTION ERROR: Cannot connect to {url}")
            print("💡 Check if Spring Boot server is running")
            return False, None

        except Exception as e:
            print(f"❌ UNEXPECTED ERROR: {e}")
            return False, None

    def run_full_test(self):
        """Chạy test đầy đủ"""
        print("🚀 API DEBUG TOOL WITH AUTHENTICATION")
        print("=" * 80)

        # Test các endpoint công khai trước
        public_endpoints = [
            '/python/embeddings'  # Endpoint này không cần auth
        ]

        print("🌐 TESTING PUBLIC ENDPOINTS:")
        print("-" * 40)

        working_endpoints = []
        for endpoint in public_endpoints:
            success, data = self.test_endpoint(endpoint, with_auth=False)
            if success:
                working_endpoints.append((endpoint, data))

        # Thử login
        print(f"\n{'=' * 80}")
        print("🔐 ATTEMPTING AUTHENTICATION")
        print("=" * 80)

        login_success = self.login()

        # Test các endpoint cần auth
        protected_endpoints = [
            '/sinhvien/embeddings',
            '/sinhvien/all',
            '/sinhvien'
        ]

        print(f"\n{'=' * 80}")
        print("🔐 TESTING PROTECTED ENDPOINTS:")
        print("=" * 80)

        if login_success:
            for endpoint in protected_endpoints:
                success, data = self.test_endpoint(endpoint, with_auth=True)
                if success:
                    working_endpoints.append((endpoint, data))
        else:
            print("⚠️ Skipping protected endpoints - login failed")

        # Summary
        print(f"\n{'=' * 80}")
        print("📊 SUMMARY & RECOMMENDATIONS")
        print("=" * 80)

        print(f"✅ Working endpoints: {len(working_endpoints)}")
        for endpoint, data in working_endpoints:
            if isinstance(data, list):
                print(f"  {endpoint}: {len(data)} records")
            else:
                print(f"  {endpoint}: OK")

        if working_endpoints:
            print(f"\n💡 RECOMMENDATIONS:")
            print(f"1. Use endpoint: {working_endpoints[0][0]}")
            print(f"2. Update face_service.py to use this endpoint")
            print(f"3. Expected data format found and validated")

            # Show sample data structure
            endpoint, data = working_endpoints[0]
            if isinstance(data, list) and len(data) > 0:
                sample = data[0]
                print(f"\n📋 Sample data structure:")
                print(json.dumps(sample, indent=2)[:500])
        else:
            print(f"\n❌ NO WORKING ENDPOINTS FOUND!")
            print(f"1. Check Spring Boot server is running")
            print(f"2. Verify endpoints exist in controllers")
            print(f"3. Check authentication configuration")


def main():
    """Main function"""
    debugger = APIDebugger()
    debugger.run_full_test()


if __name__ == "__main__":
    main()