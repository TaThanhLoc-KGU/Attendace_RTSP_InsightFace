#!/usr/bin/env python3
"""
Student Attendance System - Camera RTSP Application với InsightFace
Enhanced version with auto-start cameras and student embeddings integration

Features:
- Auto-start all cameras on startup
- Load student embeddings from backend
- Real-time face recognition with attendance recording
- Modern GUI with comprehensive monitoring
- Backend integration with Spring Boot

Author: AI Assistant
Version: 2.0.0
License: MIT
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox
import traceback
from pathlib import Path
import argparse
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import application modules
try:
    from config.config import config
    from utils.logger import app_logger, setup_logger
    from ui.main_window import MainWindow
    from services.embedding_cache import EmbeddingCacheService
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all required packages are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)


class StudentAttendanceApplication:
    """Enhanced Student Attendance Application with auto-start"""

    def __init__(self):
        self.root = None
        self.main_window = None
        self.embedding_cache = None
        self.args = None

    def parse_arguments(self):
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(
            description='Student Attendance System with Face Recognition',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python main.py                    # Normal startup with auto-start
  python main.py --no-auto-start    # Skip auto-start cameras
  python main.py --debug            # Enable debug logging
  python main.py --sync-embeddings  # Force sync embeddings on startup
  python main.py --config custom.env # Use custom config file
            """
        )

        parser.add_argument('--no-splash', action='store_true',
                            help='Skip splash screen')
        parser.add_argument('--no-auto-start', action='store_true',
                            help='Skip auto-starting cameras')
        parser.add_argument('--debug', action='store_true',
                            help='Enable debug mode')
        parser.add_argument('--sync-embeddings', action='store_true',
                            help='Force sync embeddings from backend on startup')
        parser.add_argument('--config', type=str,
                            help='Path to configuration file')
        parser.add_argument('--log-level',
                            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                            default='INFO',
                            help='Set logging level')
        parser.add_argument('--cache-dir', type=str,
                            help='Custom cache directory for embeddings')
        parser.add_argument('--backend-url', type=str,
                            help='Backend API URL (overrides config)')
        parser.add_argument('--window-size', type=str,
                            help='Window size (WIDTHxHEIGHT)')

        return parser.parse_args()

    def apply_command_line_args(self):
        """Apply command line arguments to environment"""
        if self.args.debug:
            os.environ['LOG_LEVEL'] = 'DEBUG'
        elif self.args.log_level:
            os.environ['LOG_LEVEL'] = self.args.log_level

        if self.args.config:
            os.environ['CONFIG_FILE'] = self.args.config

        if self.args.cache_dir:
            os.environ['EMBEDDING_CACHE_DIR'] = self.args.cache_dir

        if self.args.backend_url:
            os.environ['SPRING_BOOT_API_URL'] = self.args.backend_url

        if self.args.window_size:
            try:
                width, height = self.args.window_size.split('x')
                os.environ['WINDOW_WIDTH'] = width
                os.environ['WINDOW_HEIGHT'] = height
            except:
                print(f"❌ Invalid window size format: {self.args.window_size}")

    def check_requirements(self):
        """Check if all required packages are available"""
        required_packages = [
            ('cv2', 'opencv-python'),
            ('PIL', 'Pillow'),
            ('numpy', 'numpy'),
            ('requests', 'requests'),
            ('insightface', 'insightface')
        ]

        missing_packages = []
        optional_missing = []

        for package_name, pip_name in required_packages:
            try:
                __import__(package_name)
            except ImportError:
                if package_name == 'insightface':
                    optional_missing.append(pip_name)
                else:
                    missing_packages.append(pip_name)

        if missing_packages:
            error_msg = f"""❌ Missing required packages:
{', '.join(missing_packages)}

Please install them using:
pip install {' '.join(missing_packages)}

Or install all requirements:
pip install -r requirements.txt"""

            print(error_msg)

            # Show GUI error if possible
            try:
                root = tk.Tk()
                root.withdraw()
                messagebox.showerror("Missing Dependencies", error_msg)
                root.destroy()
            except:
                pass

            return False

        if optional_missing:
            print(f"⚠️ Optional packages missing: {', '.join(optional_missing)}")
            print("Face recognition features will be limited.")

        return True

    def check_system_requirements(self):
        """Check system requirements"""
        # Check Python version
        if sys.version_info < (3, 8):
            error_msg = f"""❌ Python 3.8+ required, found {sys.version_info.major}.{sys.version_info.minor}

Please upgrade Python to version 3.8 or higher.
Current version: {sys.version}
"""
            print(error_msg)
            try:
                root = tk.Tk()
                root.withdraw()
                messagebox.showerror("Python Version Error", error_msg)
                root.destroy()
            except:
                pass
            return False

        # Check if running in supported environment
        try:
            import tkinter
            test_root = tkinter.Tk()
            test_root.withdraw()
            test_root.destroy()
        except Exception as e:
            error_msg = f"""❌ GUI environment not available: {e}

Please ensure you're running in a graphical environment.
On Linux, you may need to install python3-tk:
sudo apt-get install python3-tk

On Windows, ensure you have a complete Python installation.
"""
            print(error_msg)
            try:
                messagebox.showerror("GUI Environment Error", error_msg)
            except:
                pass
            return False

        return True

    def setup_environment(self):
        """Setup application environment"""
        try:
            # Apply command line arguments
            self.apply_command_line_args()

            # Create necessary directories
            config.create_directories()

            # Setup logging
            app_logger.info("🚀 Starting Student Attendance System")
            app_logger.info(f"📂 Project root: {project_root}")
            app_logger.info(f"🐍 Python version: {sys.version}")
            app_logger.info(f"💻 Platform: {sys.platform}")
            app_logger.info(f"🎯 Arguments: {vars(self.args)}")

            # Log configuration
            app_logger.info(f"⚙️ Configuration:")
            app_logger.info(f"  - Backend URL: {config.BACKEND_URL}")
            app_logger.info(f"  - Recognition threshold: {config.RECOGNITION_THRESHOLD}")
            app_logger.info(f"  - Frame processing interval: {config.FRAME_PROCESSING_INTERVAL}")
            app_logger.info(f"  - Max concurrent cameras: {config.MAX_CONCURRENT_CAMERAS}")
            app_logger.info(f"  - Cache directory: {config.EMBEDDING_CACHE_DIR}")

            # Initialize embedding cache
            self.embedding_cache = EmbeddingCacheService()

            # Check cache age
            cache_age = self.embedding_cache.get_cache_age()
            if cache_age:
                app_logger.info(f"📂 Cache age: {cache_age}")
                if self.embedding_cache.is_cache_expired():
                    app_logger.warning("⚠️ Cache is expired, will sync from backend")
            else:
                app_logger.info("📂 No cache found, will sync from backend")

            return True

        except Exception as e:
            app_logger.error(f"❌ Failed to setup environment: {e}")
            app_logger.error(traceback.format_exc())
            return False

    def show_splash_screen(self):
        """Show enhanced splash screen"""
        if self.args.no_splash:
            return None

        try:
            splash = tk.Tk()
            splash.title("Student Attendance System")
            splash.geometry("500x400")
            splash.resizable(False, False)

            # Center splash screen
            splash.update_idletasks()
            x = (splash.winfo_screenwidth() // 2) - (500 // 2)
            y = (splash.winfo_screenheight() // 2) - (400 // 2)
            splash.geometry(f"500x400+{x}+{y}")

            # Remove window decorations
            splash.overrideredirect(True)

            # Create content
            main_frame = tk.Frame(splash, bg='#1a1a2e', padx=30, pady=30)
            main_frame.pack(fill=tk.BOTH, expand=True)

            # Logo/Title
            title_label = tk.Label(
                main_frame,
                text="🎓 Student Attendance System",
                font=('Arial', 20, 'bold'),
                fg='#4fc3f7',
                bg='#1a1a2e'
            )
            title_label.pack(pady=(20, 10))

            # Subtitle
            subtitle_label = tk.Label(
                main_frame,
                text="Camera RTSP với Face Recognition",
                font=('Arial', 12),
                fg='#81c784',
                bg='#1a1a2e'
            )
            subtitle_label.pack(pady=(0, 20))

            # Features
            features_text = """🎯 Features:
• Auto-start all cameras
• Real-time face recognition
• Automatic attendance recording
• Student database integration
• Performance monitoring"""

            features_label = tk.Label(
                main_frame,
                text=features_text,
                font=('Arial', 10),
                fg='#ffffff',
                bg='#1a1a2e',
                justify=tk.LEFT
            )
            features_label.pack(pady=(0, 20))

            # Loading message
            self.loading_label = tk.Label(
                main_frame,
                text="🔄 Initializing system...",
                font=('Arial', 11),
                fg='#ffb74d',
                bg='#1a1a2e'
            )
            self.loading_label.pack(pady=(0, 10))

            # Progress bar
            from tkinter import ttk
            self.progress = ttk.Progressbar(
                main_frame,
                mode='indeterminate',
                length=300
            )
            self.progress.pack(pady=(0, 20))
            self.progress.start()

            # Version info
            version_label = tk.Label(
                main_frame,
                text="Version 2.0.0 - Enhanced Edition",
                font=('Arial', 8),
                fg='#757575',
                bg='#1a1a2e'
            )
            version_label.pack(side=tk.BOTTOM)

            # Update display
            splash.update()

            return splash

        except Exception as e:
            app_logger.error(f"❌ Error creating splash screen: {e}")
            return None

    def update_splash_status(self, splash, message: str):
        """Update splash screen status"""
        if splash and hasattr(self, 'loading_label'):
            try:
                self.loading_label.config(text=message)
                splash.update()
            except:
                pass

    def create_gui(self):
        """Create main GUI"""
        try:
            # Create root window
            self.root = tk.Tk()

            # Set application properties
            self.root.title("🎓 Student Attendance System")

            # Create main window with auto-start disabled if specified
            if hasattr(self, 'args') and self.args.no_auto_start:
                # Temporarily disable auto-start
                original_auto_start = True
                self.main_window = MainWindow(self.root)
                self.main_window.auto_start_enabled = False
            else:
                self.main_window = MainWindow(self.root)

            app_logger.info("✅ GUI created successfully")
            return True

        except Exception as e:
            app_logger.error(f"❌ Failed to create GUI: {e}")
            app_logger.error(traceback.format_exc())
            return False

    def run(self):
        """Run the application"""
        try:
            # Parse arguments
            self.args = self.parse_arguments()

            # Check system requirements
            if not self.check_system_requirements():
                return 1

            # Show splash screen
            splash = self.show_splash_screen()

            # Setup environment
            self.update_splash_status(splash, "⚙️ Setting up environment...")
            if not self.setup_environment():
                return 1

            # Force sync embeddings if requested
            if self.args.sync_embeddings:
                self.update_splash_status(splash, "🎓 Syncing student embeddings...")
                try:
                    from services.backend_api import BackendAPI
                    backend_api = BackendAPI()
                    sync_result = self.embedding_cache.sync_embeddings_from_database(backend_api)
                    if sync_result['success']:
                        app_logger.info(f"✅ Synced {sync_result['count']} embeddings")
                    else:
                        app_logger.warning(f"⚠️ Sync failed: {sync_result['message']}")
                except Exception as e:
                    app_logger.error(f"❌ Error syncing embeddings: {e}")

            # Create GUI
            self.update_splash_status(splash, "🖥️ Creating user interface...")
            if not self.create_gui():
                return 1

            # Close splash screen
            if splash:
                self.update_splash_status(splash, "✅ System ready!")
                time.sleep(1)
                self.progress.stop()
                splash.destroy()

            # Start main event loop
            app_logger.info("🎬 Starting GUI event loop")
            self.root.mainloop()

            app_logger.info("👋 Application closed normally")
            return 0

        except KeyboardInterrupt:
            app_logger.info("🛑 Application interrupted by user")
            return 0

        except Exception as e:
            app_logger.error(f"❌ Fatal error: {e}")
            app_logger.error(traceback.format_exc())

            # Show error dialog
            try:
                if self.root:
                    messagebox.showerror(
                        "Fatal Error",
                        f"Application encountered a fatal error:\n\n{str(e)}\n\nSee logs for details."
                    )
            except:
                pass

            return 1

    def cleanup(self):
        """Cleanup application resources"""
        try:
            if self.main_window:
                self.main_window.on_closing()

            if self.root:
                self.root.destroy()

            if self.embedding_cache:
                # Save cache if needed
                pass

        except Exception as e:
            app_logger.error(f"❌ Error during cleanup: {e}")


def show_system_info():
    """Show system information"""
    print("🎓 Student Attendance System - System Information")
    print("=" * 60)
    print(f"Version: 2.0.0")
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Project Root: {project_root}")
    print()

    # Check dependencies
    print("📦 Dependencies:")
    required_packages = [
        ('cv2', 'opencv-python'),
        ('PIL', 'Pillow'),
        ('numpy', 'numpy'),
        ('requests', 'requests'),
        ('insightface', 'insightface')
    ]

    for package_name, pip_name in required_packages:
        try:
            module = __import__(package_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"  ✅ {pip_name}: {version}")
        except ImportError:
            print(f"  ❌ {pip_name}: Not installed")

    print()

    # Check configuration
    try:
        from config.config import config
        print("⚙️ Configuration:")
        print(f"  - Backend URL: {config.BACKEND_URL}")
        print(f"  - Cache Directory: {config.EMBEDDING_CACHE_DIR}")
        print(f"  - Recognition Threshold: {config.RECOGNITION_THRESHOLD}")
        print(f"  - Window Size: {config.WINDOW_WIDTH}x{config.WINDOW_HEIGHT}")
    except Exception as e:
        print(f"  ❌ Configuration error: {e}")


def main():
    """Main entry point"""
    try:
        # Handle special commands
        if len(sys.argv) > 1:
            if sys.argv[1] == '--version':
                print("Student Attendance System v2.0.0")
                return 0
            elif sys.argv[1] == '--system-info':
                show_system_info()
                return 0

        # Create and run application
        app = StudentAttendanceApplication()
        exit_code = app.run()

        # Cleanup
        app.cleanup()

        return exit_code

    except Exception as e:
        print(f"❌ Fatal error in main: {e}")
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    # Run application
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        print(traceback.format_exc())
        sys.exit(1)