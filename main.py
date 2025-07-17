#!/usr/bin/env python3
"""
Camera RTSP Application với InsightFace Integration
Main entry point for the application

Features:
- Multi-camera RTSP streaming
- Real-time face recognition with InsightFace
- Face database management
- Modern GUI with Tkinter
- Backend integration with Spring Boot

Author: AI Assistant
Version: 1.0.0
License: MIT
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import application modules
try:
    from config.config import config
    from utils.logger import app_logger, setup_logger
    from ui.main_window import MainWindow
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all required packages are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)


class Application:
    """Main application class"""

    def __init__(self):
        self.root = None
        self.main_window = None

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

        for package_name, pip_name in required_packages:
            try:
                __import__(package_name)
            except ImportError:
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

        return True

    def check_system_requirements(self):
        """Check system requirements"""
        # Check Python version
        if sys.version_info < (3, 8):
            error_msg = f"""❌ Python 3.8+ required, found {sys.version_info.major}.{sys.version_info.minor}

Please upgrade Python to version 3.8 or higher.
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
            tkinter.Tk().withdraw()
        except Exception as e:
            error_msg = f"""❌ GUI environment not available: {e}

Please ensure you're running in a graphical environment.
On Linux, you may need to install python3-tk:
sudo apt-get install python3-tk
"""
            print(error_msg)
            return False

        return True

    def setup_environment(self):
        """Setup application environment"""
        try:
            # Create necessary directories
            config.create_directories()

            # Setup logging
            app_logger.info("🚀 Starting Camera RTSP Application")
            app_logger.info(f"📂 Project root: {project_root}")
            app_logger.info(f"🐍 Python version: {sys.version}")
            app_logger.info(f"💻 Platform: {sys.platform}")

            # Log configuration
            app_logger.info(f"⚙️ Configuration:")
            app_logger.info(f"  - Backend URL: {config.BACKEND_URL}")
            app_logger.info(f"  - Recognition threshold: {config.RECOGNITION_THRESHOLD}")
            app_logger.info(f"  - Frame processing interval: {config.FRAME_PROCESSING_INTERVAL}")
            app_logger.info(f"  - Max concurrent cameras: {config.MAX_CONCURRENT_CAMERAS}")

            return True

        except Exception as e:
            app_logger.error(f"❌ Failed to setup environment: {e}")
            return False

    def create_gui(self):
        """Create main GUI"""
        try:
            # Create root window
            self.root = tk.Tk()

            # Set application properties
            self.root.title("🎥 Camera RTSP Application")

            # Create main window
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
            if not self.check_system_requirements():
                return 1

            if not self.check_requirements():
                return 1

            if not self.setup_environment():
                return 1

            if not self.create_gui():
                return 1

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

        except Exception as e:
            app_logger.error(f"❌ Error during cleanup: {e}")


def show_splash_screen():
    """Show splash screen while loading"""
    try:
        splash = tk.Tk()
        splash.title("Loading...")
        splash.geometry("400x300")
        splash.resizable(False, False)

        # Center splash screen
        splash.update_idletasks()
        x = (splash.winfo_screenwidth() // 2) - (400 // 2)
        y = (splash.winfo_screenheight() // 2) - (300 // 2)
        splash.geometry(f"400x300+{x}+{y}")

        # Remove window decorations
        splash.overrideredirect(True)

        # Create content
        main_frame = tk.Frame(splash, bg='#2c3e50', padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Logo/Title
        title_label = tk.Label(
            main_frame,
            text="🎥 Camera RTSP Application",
            font=('Arial', 18, 'bold'),
            fg='white',
            bg='#2c3e50'
        )
        title_label.pack(pady=(20, 10))

        # Subtitle
        subtitle_label = tk.Label(
            main_frame,
            text="với InsightFace Recognition",
            font=('Arial', 12),
            fg='#ecf0f1',
            bg='#2c3e50'
        )
        subtitle_label.pack(pady=(0, 30))

        # Loading message
        loading_label = tk.Label(
            main_frame,
            text="Loading application...",
            font=('Arial', 10),
            fg='#bdc3c7',
            bg='#2c3e50'
        )
        loading_label.pack(pady=(0, 10))

        # Progress bar
        from tkinter import ttk
        progress = ttk.Progressbar(main_frame, mode='indeterminate')
        progress.pack(fill=tk.X, pady=(0, 20))
        progress.start()

        # Version info
        version_label = tk.Label(
            main_frame,
            text="Version 1.0.0",
            font=('Arial', 8),
            fg='#95a5a6',
            bg='#2c3e50'
        )
        version_label.pack(side=tk.BOTTOM)

        # Update display
        splash.update()

        # Auto-close after 3 seconds
        splash.after(3000, splash.destroy)

        return splash

    except Exception as e:
        print(f"❌ Error creating splash screen: {e}")
        return None


def main():
    """Main entry point"""
    try:
        # Show splash screen
        splash = show_splash_screen()
        if splash:
            splash.mainloop()

        # Create and run application
        app = Application()
        exit_code = app.run()

        # Cleanup
        app.cleanup()

        return exit_code

    except Exception as e:
        print(f"❌ Fatal error in main: {e}")
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    # Handle command line arguments
    import argparse

    parser = argparse.ArgumentParser(description='Camera RTSP Application with InsightFace')
    parser.add_argument('--no-splash', action='store_true', help='Skip splash screen')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='Set logging level')

    args = parser.parse_args()

    # Apply command line arguments
    if args.debug:
        os.environ['LOG_LEVEL'] = 'DEBUG'
    elif args.log_level:
        os.environ['LOG_LEVEL'] = args.log_level

    if args.config:
        os.environ['CONFIG_FILE'] = args.config

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