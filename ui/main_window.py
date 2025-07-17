"""
Main window UI with auto-start cameras and student embeddings - FIXED VERSION
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
from typing import Dict, Optional

from services import BackendAPI, CameraService, FaceRecognitionService, Camera
from ui.camera_dialog import CameraSelectionDialog
from ui.stream_widget import CameraStreamWidget
from services.advanced_face_service import AdvancedFaceService
from ui.advanced_stream_widget import AdvancedCameraStreamWidget
from config.config import config
from utils.logger import ui_logger, app_logger

class MainWindow:
    """Main application window with auto-start functionality - FIXED VERSION"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.active_streams: Dict[int, CameraStreamWidget] = {}
        self.notebook: Optional[ttk.Notebook] = None
        self.welcome_frame: Optional[ttk.Frame] = None
        self.auto_start_enabled = True

        # CRITICAL FIX: Initialize all UI elements to None first
        self.connection_label = None
        self.students_label = None
        self.streams_label = None
        self.status_label = None
        self.system_info_label = None
        self.system_status_label = None
        self.progress_bar = None
        self.stats_label = None

        # Initialize services
        self.backend_api = BackendAPI()
        self.camera_service = CameraService(self.backend_api)
        self.face_service = AdvancedFaceService()

        # Setup UI in proper order - CRITICAL
        self.setup_window()
        self.create_menu()
        self.create_toolbar()
        self.create_main_content()
        self.create_status_bar()

        # CRITICAL FIX: Load initial data with proper delay to ensure UI is ready
        self.root.after(1000, self.load_initial_data)  # Delay to ensure UI is ready

        app_logger.info("🚀 Main window initialized with auto-start")

    def setup_window(self):
        """Setup main window properties"""
        self.root.title("🎥 Camera RTSP Application - Student Attendance System")
        self.root.geometry(f"{config.WINDOW_WIDTH}x{config.WINDOW_HEIGHT}")
        self.root.minsize(800, 600)

        # Configure grid weights
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Set icon (if available)
        try:
            self.root.iconbitmap('assets/icon.ico')
        except:
            pass  # Icon not found, continue without it

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="📹 Add Camera", command=self.add_camera, accelerator="Ctrl+A")
        file_menu.add_command(label="🔄 Refresh System", command=self.refresh_system, accelerator="F5")
        file_menu.add_separator()
        file_menu.add_command(label="🎓 Load Students", command=self.load_student_embeddings)
        file_menu.add_command(label="🚀 Auto Start All", command=self.auto_start_all_cameras)
        file_menu.add_separator()
        file_menu.add_command(label="❌ Exit", command=self.on_closing, accelerator="Ctrl+Q")

        # Camera menu
        camera_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Camera", menu=camera_menu)
        camera_menu.add_command(label="📹 Add Camera", command=self.add_camera)
        camera_menu.add_command(label="🚀 Auto Start All", command=self.auto_start_all_cameras)
        camera_menu.add_separator()
        camera_menu.add_command(label="⏹️ Stop All Streams", command=self.stop_all_streams)
        camera_menu.add_command(label="🗑️ Close All Tabs", command=self.close_all_tabs)

        # Student menu
        student_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Students", menu=student_menu)
        student_menu.add_command(label="🎓 Load Students", command=self.load_student_embeddings)
        student_menu.add_command(label="📊 Student Statistics", command=self.show_student_statistics)
        student_menu.add_separator()
        student_menu.add_command(label="📝 View Attendance", command=self.view_attendance)
        student_menu.add_command(label="💾 Export Attendance", command=self.export_attendance)

        # System menu
        system_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="System", menu=system_menu)
        system_menu.add_command(label="🔄 Refresh All", command=self.refresh_system)
        system_menu.add_command(label="📊 System Statistics", command=self.show_system_statistics)
        system_menu.add_separator()
        system_menu.add_command(label="⚙️ Settings", command=self.open_settings)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="📖 User Guide", command=self.show_user_guide)
        help_menu.add_command(label="ℹ️ About", command=self.show_about)

        # Bind keyboard shortcuts
        self.root.bind('<Control-a>', lambda e: self.add_camera())
        self.root.bind('<F5>', lambda e: self.refresh_system())
        self.root.bind('<Control-q>', lambda e: self.on_closing())

    def create_toolbar(self):
        """Create toolbar - SAFE VERSION"""
        toolbar = ttk.Frame(self.root)
        toolbar.grid(row=0, column=0, sticky="ew", padx=10, pady=5)

        # Left side buttons
        left_frame = ttk.Frame(toolbar)
        left_frame.pack(side=tk.LEFT)

        ttk.Button(
            left_frame,
            text="📹 Add Camera",
            command=self.add_camera
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            left_frame,
            text="🚀 Auto Start All",
            command=self.auto_start_all_cameras
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            left_frame,
            text="🎓 Load Students",
            command=self.load_student_embeddings
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            left_frame,
            text="🔄 Refresh",
            command=self.refresh_system
        ).pack(side=tk.LEFT, padx=5)

        # Separator
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # Right side info - CRITICAL FIX: Create labels safely
        right_frame = ttk.Frame(toolbar)
        right_frame.pack(side=tk.RIGHT)

        # Connection status
        self.connection_label = ttk.Label(
            right_frame,
            text="🔴 Disconnected",
            font=('Arial', 9)
        )
        self.connection_label.pack(side=tk.RIGHT, padx=5)

        # Students count
        self.students_label = ttk.Label(
            right_frame,
            text="🎓 0 students",
            font=('Arial', 9)
        )
        self.students_label.pack(side=tk.RIGHT, padx=5)

        # Active streams counter
        self.streams_label = ttk.Label(
            right_frame,
            text="📹 0 streams",
            font=('Arial', 9)
        )
        self.streams_label.pack(side=tk.RIGHT, padx=5)

    def create_main_content(self):
        """Create main content area"""
        main_frame = ttk.Frame(self.root)
        main_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        # Welcome screen
        self.create_welcome_screen(main_frame)

        # Store reference to main frame
        self.main_frame = main_frame

    def create_welcome_screen(self, parent):
        """Create welcome screen"""
        self.welcome_frame = ttk.Frame(parent)
        self.welcome_frame.grid(row=0, column=0, sticky="nsew")
        self.welcome_frame.grid_rowconfigure(0, weight=1)
        self.welcome_frame.grid_columnconfigure(0, weight=1)

        # Welcome content
        content_frame = ttk.Frame(self.welcome_frame)
        content_frame.grid(row=0, column=0)

        # Title
        title_label = ttk.Label(
            content_frame,
            text="🎥 Student Attendance System",
            font=('Arial', 24, 'bold')
        )
        title_label.pack(pady=(0, 10))

        # Subtitle
        subtitle_label = ttk.Label(
            content_frame,
            text="Camera RTSP với Face Recognition",
            font=('Arial', 14),
            foreground='gray'
        )
        subtitle_label.pack(pady=(0, 30))

        # System status - SAFE CREATION
        self.system_status_label = ttk.Label(
            content_frame,
            text="🔄 Initializing system...",
            font=('Arial', 12),
            foreground='blue'
        )
        self.system_status_label.pack(pady=(0, 20))

        # Description
        desc_text = """🎯 System Features:
• 📹 Automatic camera streaming
• 🎓 Student face recognition
• 📝 Automatic attendance recording
• 📊 Real-time statistics
• 🔄 Auto-sync with database

🚀 Auto-Start Process:
1. Loading student embeddings from database
2. Discovering available cameras
3. Auto-starting camera streams
4. Ready for attendance recording

💡 Status: System is auto-starting all cameras..."""

        desc_label = ttk.Label(
            content_frame,
            text=desc_text,
            font=('Arial', 11),
            justify=tk.LEFT
        )
        desc_label.pack(pady=(0, 20))

        # Progress bar - SAFE CREATION
        self.progress_bar = ttk.Progressbar(
            content_frame,
            mode='indeterminate',
            length=400
        )
        self.progress_bar.pack(pady=(0, 20))

        # Action buttons
        button_frame = ttk.Frame(content_frame)
        button_frame.pack()

        ttk.Button(
            button_frame,
            text="📹 Add Camera",
            command=self.add_camera,
            width=15
        ).pack(side=tk.LEFT, padx=10)

        ttk.Button(
            button_frame,
            text="🎓 Load Students",
            command=self.load_student_embeddings,
            width=15
        ).pack(side=tk.LEFT, padx=10)

        ttk.Button(
            button_frame,
            text="🚀 Auto Start All",
            command=self.auto_start_all_cameras,
            width=15
        ).pack(side=tk.LEFT, padx=10)

    def create_status_bar(self):
        """Create status bar - SAFE VERSION"""
        status_frame = ttk.Frame(self.root)
        status_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=5)

        # Status label
        self.status_label = ttk.Label(
            status_frame,
            text="Initializing system...",
            font=('Arial', 9),
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # System info
        self.system_info_label = ttk.Label(
            status_frame,
            text="System Starting",
            font=('Arial', 9),
            relief=tk.SUNKEN,
            anchor=tk.E,
            width=20
        )
        self.system_info_label.pack(side=tk.RIGHT, padx=(5, 0))

    def load_initial_data(self):
        """Load initial data and auto-start system - SAFE VERSION"""
        self.update_status("🔄 Initializing system...")

        # SAFE progress bar start
        if hasattr(self, 'progress_bar') and self.progress_bar:
            self.progress_bar.start()

        # Use after() to delay initialization until UI is ready
        self.root.after(500, self._start_initialization)

    def _start_initialization(self):
        """Start initialization after UI is ready"""
        # Run in thread to avoid blocking UI
        thread = threading.Thread(target=self._load_initial_data_thread, daemon=True)
        thread.start()

    def _load_initial_data_thread(self):
        """Thread function to load initial data"""
        try:
            # 1. Test backend connection
            self.root.after(0, self._update_system_status, "🔗 Testing backend connection...")
            response = self.backend_api.test_connection()

            if response.success:
                self.root.after(0, self._on_backend_connected)
            else:
                self.root.after(0, self._on_backend_disconnected, response.message)
                return

            # 2. Load student embeddings
            self.root.after(0, self._update_system_status, "🎓 Loading student embeddings...")
            embeddings_result = self.face_service.load_student_embeddings()

            if embeddings_result['success']:
                self.root.after(0, self._on_embeddings_loaded, embeddings_result['count'])
            else:
                self.root.after(0, self._on_embeddings_error, embeddings_result['message'])

            # 3. Load cameras
            self.root.after(0, self._update_system_status, "📹 Loading cameras...")
            cameras_result = self.camera_service.load_cameras_from_backend()

            if cameras_result['success']:
                self.root.after(0, self._on_cameras_loaded, cameras_result['count'])
            else:
                self.root.after(0, self._on_cameras_error, cameras_result['message'])
                return

            # 4. Auto-start cameras if enabled (with proper delay)
            if self.auto_start_enabled:
                self.root.after(0, self._update_system_status, "🚀 Auto-starting cameras...")
                self.root.after(8000, self._auto_start_cameras)  # 8 second delay

            # 5. System ready
            self.root.after(2000, self._on_system_ready)  # Delay system ready

        except Exception as e:
            ui_logger.error(f"❌ Error loading initial data: {e}")
            self.root.after(0, self._on_initial_data_error, str(e))

    def _update_system_status(self, message: str):
        """Update system status message - SAFE VERSION"""
        if hasattr(self, 'system_status_label') and self.system_status_label and self.system_status_label.winfo_exists():
            self.system_status_label.config(text=message)
        self.update_status(message)

    def _on_backend_connected(self):
        """Handle backend connection success - SAFE VERSION"""
        if hasattr(self, 'connection_label') and self.connection_label and self.connection_label.winfo_exists():
            self.connection_label.config(text="🟢 Connected", foreground='green')
        ui_logger.info("✅ Backend connected successfully")

    def _on_backend_disconnected(self, message: str):
        """Handle backend connection failure - SAFE VERSION"""
        if hasattr(self, 'connection_label') and self.connection_label and self.connection_label.winfo_exists():
            self.connection_label.config(text="🔴 Disconnected", foreground='red')
        self.update_status(f"❌ Backend disconnected: {message}")
        messagebox.showerror("Backend Error", f"Cannot connect to backend:\n{message}")

    def _on_embeddings_loaded(self, count: int):
        """Handle embeddings loaded - SAFE VERSION"""
        if hasattr(self, 'students_label') and self.students_label and self.students_label.winfo_exists():
            self.students_label.config(text=f"🎓 {count} students")
        ui_logger.info(f"✅ Loaded {count} student embeddings")

    def _on_embeddings_error(self, message: str):
        """Handle embeddings load error - SAFE VERSION"""
        if hasattr(self, 'students_label') and self.students_label and self.students_label.winfo_exists():
            self.students_label.config(text="🎓 0 students")
        ui_logger.warning(f"⚠️ Failed to load embeddings: {message}")

    def _on_cameras_loaded(self, count: int):
        """Handle cameras loaded"""
        ui_logger.info(f"✅ Loaded {count} cameras")

    def _on_cameras_error(self, message: str):
        """Handle cameras load error"""
        ui_logger.error(f"❌ Failed to load cameras: {message}")
        messagebox.showerror("Camera Error", f"Cannot load cameras:\n{message}")

    def _auto_start_cameras(self):
        """Auto-start all available cameras"""
        try:
            cameras = self.camera_service.get_active_cameras()

            if not cameras:
                self.update_status("⚠️ No cameras available for auto-start")
                return

            ui_logger.info(f"🚀 Auto-starting {len(cameras)} cameras")

            # Start cameras with delay
            for i, camera in enumerate(cameras):
                # Delay between camera starts
                self.root.after((i + 1) * 3000, lambda c=camera: self.create_camera_stream(c))

            self.update_status(f"🚀 Auto-starting {len(cameras)} cameras...")

        except Exception as e:
            ui_logger.error(f"❌ Error auto-starting cameras: {e}")
            self.update_status(f"❌ Auto-start failed: {str(e)}")

    def _on_system_ready(self):
        """Handle system ready - SAFE VERSION"""
        if hasattr(self, 'progress_bar') and self.progress_bar:
            self.progress_bar.stop()

        if hasattr(self, 'system_status_label') and self.system_status_label and self.system_status_label.winfo_exists():
            self.system_status_label.config(text="✅ System ready for attendance recording", foreground='green')

        self.update_status("✅ System ready - All cameras auto-started")

        # Update system info
        student_count = len(self.face_service.student_service.student_embeddings)
        camera_count = len(self.camera_service.cameras)

        if hasattr(self, 'system_info_label') and self.system_info_label and self.system_info_label.winfo_exists():
            self.system_info_label.config(text=f"📹 {camera_count} cameras | 🎓 {student_count} students")

    def _on_initial_data_error(self, error: str):
        """Handle initial data load error - SAFE VERSION"""
        if hasattr(self, 'progress_bar') and self.progress_bar:
            self.progress_bar.stop()

        if hasattr(self, 'system_status_label') and self.system_status_label and self.system_status_label.winfo_exists():
            self.system_status_label.config(text="❌ System initialization failed", foreground='red')

        self.update_status(f"❌ Initialization error: {error}")
        messagebox.showerror("Initialization Error", f"Failed to initialize system:\n{error}")

    def create_camera_stream(self, camera: Camera):
        """Create camera stream tab - SAFE VERSION"""
        try:
            if camera.id in self.active_streams:
                ui_logger.info(f"📹 Camera '{camera.name}' already active")
                return

            # Hide welcome screen
            if self.welcome_frame:
                self.welcome_frame.grid_forget()

            # Create notebook if not exists
            if not self.notebook:
                self.notebook = ttk.Notebook(self.main_frame)
                self.notebook.grid(row=0, column=0, sticky="nsew")

                # Bind tab close event
                self.notebook.bind('<Button-3>', self.show_tab_context_menu)

            # Create new tab
            tab_frame = ttk.Frame(self.notebook)
            tab_name = f"📹 {camera.name}"
            self.notebook.add(tab_frame, text=tab_name)

            # CRITICAL FIX: Create camera stream widget with proper parent
            stream_widget = AdvancedCameraStreamWidget(tab_frame, camera, self.face_service)

            # Select new tab
            self.notebook.select(tab_frame)

            # Update status
            self.update_status(f"✅ Added camera: {camera.name}")
            self.update_streams_counter()

            ui_logger.info(f"✅ Created stream for camera: {camera.name}")

        except Exception as e:
            ui_logger.error(f"❌ Error creating camera stream: {e}")
            messagebox.showerror("Error", f"Failed to create camera stream:\n{str(e)}")

    def update_status(self, message: str):
        """Update status message - SAFE VERSION"""
        try:
            if hasattr(self, 'status_label') and self.status_label and self.status_label.winfo_exists():
                self.status_label.config(text=message)
            else:
                # If status_label doesn't exist yet, just log it
                ui_logger.info(f"Status: {message}")
        except Exception as e:
            ui_logger.error(f"❌ Error updating status: {e}")

    def update_streams_counter(self):
        """Update active streams counter - SAFE VERSION"""
        try:
            count = len(self.active_streams)
            if hasattr(self, 'streams_label') and self.streams_label and self.streams_label.winfo_exists():
                self.streams_label.config(text=f"📹 {count} streams")
        except Exception as e:
            ui_logger.error(f"❌ Error updating streams counter: {e}")

    # Rest of the methods remain the same but with safe UI access patterns...
    def load_student_embeddings(self):
        """Load student embeddings from backend"""
        self.update_status("Loading student embeddings...")
        thread = threading.Thread(target=self._load_embeddings_thread, daemon=True)
        thread.start()

    def _load_embeddings_thread(self):
        """Thread function to load embeddings"""
        try:
            result = self.face_service.load_student_embeddings()
            self.root.after(0, self._on_embeddings_loaded_manual, result)
        except Exception as e:
            self.root.after(0, self._on_embeddings_error_manual, str(e))

    def _on_embeddings_loaded_manual(self, result):
        """Handle manual embeddings load"""
        if result['success']:
            count = result['count']
            if hasattr(self, 'students_label') and self.students_label and self.students_label.winfo_exists():
                self.students_label.config(text=f"🎓 {count} students")
            self.update_status(f"✅ Loaded {count} student embeddings")
            messagebox.showinfo("Success", f"✅ Successfully loaded {count} student embeddings")
        else:
            self.update_status(f"❌ Failed to load embeddings: {result['message']}")
            messagebox.showerror("Error", f"Failed to load embeddings:\n{result['message']}")

    def _on_embeddings_error_manual(self, error: str):
        """Handle manual embeddings load error"""
        self.update_status(f"❌ Error loading embeddings: {error}")
        messagebox.showerror("Error", f"Error loading embeddings:\n{error}")

    def auto_start_all_cameras(self):
        """Auto-start all available cameras"""
        try:
            cameras = self.camera_service.get_active_cameras()

            if not cameras:
                messagebox.showwarning("No Cameras", "No active cameras available for auto-start")
                return

            # Close existing streams first
            if self.active_streams:
                self.close_all_tabs()

            # Start all cameras
            for i, camera in enumerate(cameras):
                # Add delay between starts
                self.root.after(i * 1000, lambda c=camera: self.create_camera_stream(c))

            self.update_status(f"🚀 Auto-starting {len(cameras)} cameras...")

        except Exception as e:
            ui_logger.error(f"❌ Error auto-starting cameras: {e}")
            messagebox.showerror("Error", f"Failed to auto-start cameras:\n{str(e)}")

    def add_camera(self):
        """Add camera stream manually"""
        try:
            cameras = self.camera_service.cameras

            if not cameras:
                messagebox.showwarning(
                    "No Cameras",
                    "No cameras available!\n\nPlease check:\n• Backend connection\n• Camera database\n• Network connectivity"
                )
                return

            # Show camera selection dialog
            dialog = CameraSelectionDialog(self.root, self.camera_service)
            self.root.wait_window(dialog.dialog)

            result, selected_camera = dialog.get_result()

            if result == 'ok' and selected_camera:
                self.create_camera_stream(selected_camera)

        except Exception as e:
            ui_logger.error(f"❌ Error adding camera: {e}")
            messagebox.showerror("Error", f"Failed to add camera:\n{str(e)}")

    def refresh_system(self):
        """Refresh entire system"""
        self.update_status("🔄 Refreshing system...")
        thread = threading.Thread(target=self._refresh_system_thread, daemon=True)
        thread.start()

    def _refresh_system_thread(self):
        """Thread function to refresh system"""
        try:
            # Refresh cameras
            cameras_result = self.camera_service.refresh_cameras()

            # Refresh embeddings
            embeddings_result = self.face_service.load_student_embeddings()

            self.root.after(0, self._on_system_refreshed, cameras_result, embeddings_result)

        except Exception as e:
            self.root.after(0, self._on_system_refresh_error, str(e))

    def _on_system_refreshed(self, cameras_result, embeddings_result):
        """Handle system refreshed"""
        camera_count = cameras_result.get('count', 0) if cameras_result.get('success') else 0
        student_count = embeddings_result.get('count', 0) if embeddings_result.get('success') else 0

        # Update UI safely
        if hasattr(self, 'students_label') and self.students_label and self.students_label.winfo_exists():
            self.students_label.config(text=f"🎓 {student_count} students")

        if hasattr(self, 'system_info_label') and self.system_info_label and self.system_info_label.winfo_exists():
            self.system_info_label.config(text=f"📹 {camera_count} cameras | 🎓 {student_count} students")

        self.update_status(f"✅ System refreshed - {camera_count} cameras, {student_count} students")

        messagebox.showinfo("Success", f"✅ System refreshed successfully!\n\nCameras: {camera_count}\nStudents: {student_count}")

    def _on_system_refresh_error(self, error: str):
        """Handle system refresh error"""
        self.update_status(f"❌ System refresh error: {error}")
        messagebox.showerror("Error", f"Failed to refresh system:\n{error}")

    def show_tab_context_menu(self, event):
        """Show context menu for tabs"""
        try:
            tab_index = self.notebook.index(f"@{event.x},{event.y}")

            context_menu = tk.Menu(self.root, tearoff=0)
            context_menu.add_command(label="📸 Take Snapshot", command=lambda: self.take_snapshot(tab_index))
            context_menu.add_command(label="🔄 Restart Stream", command=lambda: self.restart_stream(tab_index))
            context_menu.add_separator()
            context_menu.add_command(label="❌ Close Tab", command=lambda: self.close_tab(tab_index))

            context_menu.post(event.x_root, event.y_root)

        except:
            pass  # Click outside tab

    def close_tab(self, tab_index: int):
        """Close specific tab"""
        try:
            tab_frame = self.notebook.tabs()[tab_index]

            # Find and cleanup stream widget
            for camera_id, stream_widget in list(self.active_streams.items()):
                if stream_widget.main_frame.winfo_parent() == str(tab_frame):
                    stream_widget.cleanup()
                    del self.active_streams[camera_id]
                    break

            # Remove tab
            self.notebook.forget(tab_index)

            # Show welcome screen if no tabs left
            if not self.notebook.tabs():
                self.show_welcome_screen()

            self.update_streams_counter()

        except Exception as e:
            ui_logger.error(f"❌ Error closing tab: {e}")

    def close_all_tabs(self):
        """Close all camera tabs"""
        if not self.notebook or not self.notebook.tabs():
            return

        # Stop all streams
        for stream_widget in self.active_streams.values():
            stream_widget.cleanup()

        # Clear active streams
        self.active_streams.clear()

        # Remove notebook
        self.notebook.destroy()
        self.notebook = None

        # Show welcome screen
        self.show_welcome_screen()

        self.update_streams_counter()
        self.update_status("All streams closed")

    def stop_all_streams(self):
        """Stop all active streams"""
        for stream_widget in self.active_streams.values():
            stream_widget.stop_stream()

        self.update_status("All streams stopped")

    def show_welcome_screen(self):
        """Show welcome screen"""
        if not self.welcome_frame:
            self.create_welcome_screen(self.main_frame)
        else:
            self.welcome_frame.grid(row=0, column=0, sticky="nsew")

    # Placeholder methods for menu items
    def show_student_statistics(self):
        """Show student statistics"""
        try:
            stats = self.face_service.get_statistics()

            stats_text = f"""🎓 Student Statistics:

Total Students: {stats['total_students']}
Loaded in Cache: {stats['cache_size']}
Recognition Threshold: {stats['recognition_threshold']}

System Status:
InsightFace: {'✅ Ready' if stats['is_initialized'] else '❌ Not Available'}
Last Loaded: {stats['last_loaded'][:19] if stats['last_loaded'] else 'Never'}

Active Cameras: {len(self.active_streams)}
"""

            messagebox.showinfo("Student Statistics", stats_text)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load statistics:\n{str(e)}")

    def show_system_statistics(self):
        """Show system statistics"""
        try:
            face_stats = self.face_service.get_statistics()
            camera_stats = self.camera_service.get_camera_stats()

            stats_text = f"""🖥️ System Statistics:

📹 Cameras:
• Total: {camera_stats['total']}
• Active: {camera_stats['active']}
• Streaming: {camera_stats['streaming']}

🎓 Students:
• Total: {face_stats['total_students']}
• Cached: {face_stats['cache_size']}

⚙️ System:
• Recognition Threshold: {face_stats['recognition_threshold']}
• InsightFace: {'Ready' if face_stats['is_initialized'] else 'Not Available'}
• Active Streams: {len(self.active_streams)}
"""

            messagebox.showinfo("System Statistics", stats_text)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load statistics:\n{str(e)}")

    def view_attendance(self):
        messagebox.showinfo("Info", "Attendance view feature will be implemented")

    def export_attendance(self):
        messagebox.showinfo("Info", "Export attendance feature will be implemented")

    def open_settings(self):
        messagebox.showinfo("Info", "Settings dialog will be implemented")

    def take_snapshot(self, tab_index: int):
        messagebox.showinfo("Info", "Snapshot feature will be implemented")

    def restart_stream(self, tab_index: int):
        messagebox.showinfo("Info", "Restart stream feature will be implemented")

    def show_user_guide(self):
        messagebox.showinfo("Info", "User guide will be implemented")

    def show_about(self):
        about_text = """🎥 Student Attendance System

Version: 2.0.0
Built for: Student Face Recognition & Attendance

Features:
• Auto-start all cameras
• Real-time face recognition
• Automatic attendance recording
• Student database integration
• Performance monitoring

Built with:
• Python 3.8+
• OpenCV + InsightFace
• Tkinter GUI
• Spring Boot Backend

© 2024 Student Attendance System"""

        messagebox.showinfo("About", about_text)

    def on_closing(self):
        """Handle application closing"""
        if self.active_streams:
            result = messagebox.askyesno(
                "Confirm Exit",
                f"There are {len(self.active_streams)} active streams.\n\nDo you want to exit?"
            )

            if not result:
                return

        try:
            # Cleanup all streams
            for stream_widget in self.active_streams.values():
                stream_widget.cleanup()

            # Save settings if needed
            app_logger.info("👋 Application closing")

            # Destroy window
            self.root.destroy()

        except Exception as e:
            ui_logger.error(f"❌ Error during closing: {e}")
            self.root.destroy()
