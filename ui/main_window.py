"""
Main window UI for Camera RTSP Application
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
from typing import Dict, Optional

from services import BackendAPI, CameraService, FaceRecognitionService, Camera
from ui.camera_dialog import CameraSelectionDialog
from ui.stream_widget import CameraStreamWidget
from ui.face_manager import FaceManagerDialog
from config.config import config
from utils.logger import ui_logger, app_logger


class MainWindow:
    """Main application window"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.active_streams: Dict[int, CameraStreamWidget] = {}
        self.notebook: Optional[ttk.Notebook] = None
        self.welcome_frame: Optional[ttk.Frame] = None

        # Initialize services
        self.backend_api = BackendAPI()
        self.camera_service = CameraService(self.backend_api)
        self.face_service = FaceRecognitionService()

        # Setup UI
        self.setup_window()
        self.create_menu()
        self.create_toolbar()
        self.create_main_content()
        self.create_status_bar()

        # Load initial data
        self.load_initial_data()

        app_logger.info("🚀 Main window initialized")

    def setup_window(self):
        """Setup main window properties"""
        self.root.title("🎥 Camera RTSP Application với InsightFace")
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
        file_menu.add_command(label="🔄 Refresh Cameras", command=self.refresh_cameras, accelerator="F5")
        file_menu.add_separator()
        file_menu.add_command(label="📥 Import Settings", command=self.import_settings)
        file_menu.add_command(label="📤 Export Settings", command=self.export_settings)
        file_menu.add_separator()
        file_menu.add_command(label="❌ Exit", command=self.on_closing, accelerator="Ctrl+Q")

        # Camera menu
        camera_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Camera", menu=camera_menu)
        camera_menu.add_command(label="📹 Add Camera", command=self.add_camera)
        camera_menu.add_command(label="🔍 Test Connection", command=self.test_camera_connection)
        camera_menu.add_separator()
        camera_menu.add_command(label="⏹️ Stop All Streams", command=self.stop_all_streams)
        camera_menu.add_command(label="🗑️ Close All Tabs", command=self.close_all_tabs)

        # Face Recognition menu
        face_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Face Recognition", menu=face_menu)
        face_menu.add_command(label="👤 Add Face", command=self.add_face)
        face_menu.add_command(label="📋 Manage Faces", command=self.manage_faces)
        face_menu.add_separator()
        face_menu.add_command(label="💾 Backup Database", command=self.backup_face_database)
        face_menu.add_command(label="📥 Restore Database", command=self.restore_face_database)
        face_menu.add_separator()
        face_menu.add_command(label="⚙️ Face Settings", command=self.face_settings)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="🔍 Camera List", command=self.show_camera_list)
        view_menu.add_command(label="📊 Statistics", command=self.show_statistics)
        view_menu.add_separator()
        view_menu.add_command(label="🎨 Themes", command=self.change_theme)
        view_menu.add_command(label="⚙️ Preferences", command=self.open_preferences)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="📖 User Guide", command=self.show_user_guide)
        help_menu.add_command(label="🔧 Troubleshooting", command=self.show_troubleshooting)
        help_menu.add_separator()
        help_menu.add_command(label="ℹ️ About", command=self.show_about)

        # Bind keyboard shortcuts
        self.root.bind('<Control-a>', lambda e: self.add_camera())
        self.root.bind('<F5>', lambda e: self.refresh_cameras())
        self.root.bind('<Control-q>', lambda e: self.on_closing())

    def create_toolbar(self):
        """Create toolbar"""
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
            text="🔄 Refresh",
            command=self.refresh_cameras
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            left_frame,
            text="👤 Add Face",
            command=self.add_face
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            left_frame,
            text="📋 Manage Faces",
            command=self.manage_faces
        ).pack(side=tk.LEFT, padx=5)

        # Separator
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # Right side info
        right_frame = ttk.Frame(toolbar)
        right_frame.pack(side=tk.RIGHT)

        # Connection status
        self.connection_label = ttk.Label(
            right_frame,
            text="🔴 Disconnected",
            font=('Arial', 9)
        )
        self.connection_label.pack(side=tk.RIGHT, padx=5)

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
            text="🎥 Camera RTSP Application",
            font=('Arial', 24, 'bold')
        )
        title_label.pack(pady=(0, 20))

        # Subtitle
        subtitle_label = ttk.Label(
            content_frame,
            text="với InsightFace Recognition",
            font=('Arial', 14),
            foreground='gray'
        )
        subtitle_label.pack(pady=(0, 40))

        # Description
        desc_text = """👋 Welcome to Camera RTSP Application

🎯 Features:
• 📹 Multiple camera streaming support
• 🔍 Real-time face recognition
• 👤 Face database management
• 📊 Performance monitoring
• ⚙️ Advanced settings

🚀 Getting Started:
1. Click 'Add Camera' to select a camera from database
2. Start streaming and enjoy real-time face recognition
3. Add faces to database for better recognition

💡 Tips:
• Use F5 to refresh camera list
• Right-click on streams for more options
• Check face recognition settings for optimal performance
"""

        desc_label = ttk.Label(
            content_frame,
            text=desc_text,
            font=('Arial', 11),
            justify=tk.LEFT
        )
        desc_label.pack(pady=(0, 30))

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
            text="👤 Add Face",
            command=self.add_face,
            width=15
        ).pack(side=tk.LEFT, padx=10)

        ttk.Button(
            button_frame,
            text="📋 Manage Faces",
            command=self.manage_faces,
            width=15
        ).pack(side=tk.LEFT, padx=10)

    def create_status_bar(self):
        """Create status bar"""
        status_frame = ttk.Frame(self.root)
        status_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=5)

        # Status label
        self.status_label = ttk.Label(
            status_frame,
            text="Ready",
            font=('Arial', 9),
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Progress bar (hidden by default)
        self.progress_bar = ttk.Progressbar(
            status_frame,
            mode='indeterminate'
        )
        # Don't pack it yet, will be shown when needed

        # System info
        self.system_info_label = ttk.Label(
            status_frame,
            text="System Ready",
            font=('Arial', 9),
            relief=tk.SUNKEN,
            anchor=tk.E,
            width=20
        )
        self.system_info_label.pack(side=tk.RIGHT, padx=(5, 0))

    def load_initial_data(self):
        """Load initial data in background"""
        self.update_status("Loading initial data...")

        # Run in thread to avoid blocking UI
        thread = threading.Thread(target=self._load_initial_data_thread, daemon=True)
        thread.start()

    def _load_initial_data_thread(self):
        """Thread function to load initial data"""
        try:
            # Test backend connection
            response = self.backend_api.test_connection()
            if response.success:
                self.root.after(0, self._on_backend_connected)
            else:
                self.root.after(0, self._on_backend_disconnected, response.message)

            # Load cameras
            self.camera_service.load_cameras_from_backend()

            # Update UI
            self.root.after(0, self._on_initial_data_loaded)

        except Exception as e:
            ui_logger.error(f"❌ Error loading initial data: {e}")
            self.root.after(0, self._on_initial_data_error, str(e))

    def _on_backend_connected(self):
        """Handle backend connection success"""
        self.connection_label.config(text="🟢 Connected", foreground='green')
        self.update_status("Backend connected successfully")

    def _on_backend_disconnected(self, message: str):
        """Handle backend connection failure"""
        self.connection_label.config(text="🔴 Disconnected", foreground='red')
        self.update_status(f"Backend disconnected: {message}")

    def _on_initial_data_loaded(self):
        """Handle initial data loaded"""
        camera_count = len(self.camera_service.cameras)
        self.update_status(f"Ready - {camera_count} cameras available")

        # Update system info
        face_count = len(self.face_service.face_db)
        self.system_info_label.config(text=f"📹 {camera_count} cameras | 👤 {face_count} faces")

    def _on_initial_data_error(self, error: str):
        """Handle initial data load error"""
        self.update_status(f"Error loading data: {error}")
        messagebox.showerror("Initialization Error", f"Failed to load initial data:\n{error}")

    def add_camera(self):
        """Add camera stream"""
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

    def create_camera_stream(self, camera: Camera):
        """Create camera stream tab"""
        try:
            if camera.id in self.active_streams:
                messagebox.showinfo("Info", f"Camera '{camera.name}' is already active!")
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

            # Create camera stream widget
            stream_widget = CameraStreamWidget(tab_frame, camera, self.face_service)
            self.active_streams[camera.id] = stream_widget

            # Select new tab
            self.notebook.select(tab_frame)

            # Update status
            self.update_status(f"✅ Added camera: {camera.name}")
            self.update_streams_counter()

            ui_logger.info(f"✅ Created stream for camera: {camera.name}")

        except Exception as e:
            ui_logger.error(f"❌ Error creating camera stream: {e}")
            messagebox.showerror("Error", f"Failed to create camera stream:\n{str(e)}")

    def show_tab_context_menu(self, event):
        """Show context menu for tabs"""
        try:
            tab_index = self.notebook.index(f"@{event.x},{event.y}")

            context_menu = tk.Menu(self.root, tearoff=0)
            context_menu.add_command(label="📸 Take Snapshot", command=lambda: self.take_snapshot(tab_index))
            context_menu.add_command(label="⚙️ Settings", command=lambda: self.camera_settings(tab_index))
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

        result = messagebox.askyesno(
            "Close All Tabs",
            "Are you sure you want to close all camera streams?"
        )

        if result:
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

    def refresh_cameras(self):
        """Refresh camera list"""
        self.update_status("Refreshing cameras...")

        # Run in thread
        thread = threading.Thread(target=self._refresh_cameras_thread, daemon=True)
        thread.start()

    def _refresh_cameras_thread(self):
        """Thread function to refresh cameras"""
        try:
            result = self.camera_service.refresh_cameras()
            self.root.after(0, self._on_cameras_refreshed, result)
        except Exception as e:
            self.root.after(0, self._on_refresh_error, str(e))

    def _on_cameras_refreshed(self, result):
        """Handle cameras refreshed"""
        if result.get('success'):
            count = result.get('count', 0)
            self.update_status(f"✅ Refreshed {count} cameras")

            # Update system info
            face_count = len(self.face_service.face_db)
            self.system_info_label.config(text=f"📹 {count} cameras | 👤 {face_count} faces")
        else:
            message = result.get('message', 'Unknown error')
            self.update_status(f"❌ Refresh failed: {message}")

    def _on_refresh_error(self, error: str):
        """Handle refresh error"""
        self.update_status(f"❌ Refresh error: {error}")
        messagebox.showerror("Refresh Error", f"Failed to refresh cameras:\n{error}")

    def add_face(self):
        """Add face to database"""
        try:
            file_path = filedialog.askopenfilename(
                title="Select Face Image",
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                    ("All files", "*.*")
                ]
            )

            if not file_path:
                return

            # Get name from user
            name = tk.simpledialog.askstring(
                "Face Name",
                "Enter name for this face:",
                parent=self.root
            )

            if not name:
                return

            # Add to database
            self.update_status(f"Adding face: {name}...")

            # Run in thread
            thread = threading.Thread(
                target=self._add_face_thread,
                args=(name, file_path),
                daemon=True
            )
            thread.start()

        except Exception as e:
            ui_logger.error(f"❌ Error adding face: {e}")
            messagebox.showerror("Error", f"Failed to add face:\n{str(e)}")

    def _add_face_thread(self, name: str, file_path: str):
        """Thread function to add face"""
        try:
            import cv2

            # Load image
            image = cv2.imread(file_path)
            if image is None:
                self.root.after(0, self._on_add_face_error, "Cannot load image file")
                return

            # Add to database
            success = self.face_service.add_face_to_database(name, image)
            self.root.after(0, self._on_face_added, name, success)

        except Exception as e:
            self.root.after(0, self._on_add_face_error, str(e))

    def _on_face_added(self, name: str, success: bool):
        """Handle face added result"""
        if success:
            self.update_status(f"✅ Added face: {name}")
            messagebox.showinfo("Success", f"✅ Face '{name}' added successfully!")

            # Update system info
            face_count = len(self.face_service.face_db)
            camera_count = len(self.camera_service.cameras)
            self.system_info_label.config(text=f"📹 {camera_count} cameras | 👤 {face_count} faces")
        else:
            self.update_status(f"❌ Failed to add face: {name}")
            messagebox.showerror("Error", f"Failed to add face '{name}' to database!")

    def _on_add_face_error(self, error: str):
        """Handle add face error"""
        self.update_status(f"❌ Add face error: {error}")
        messagebox.showerror("Error", f"Error adding face:\n{error}")

    def manage_faces(self):
        """Open face management dialog"""
        try:
            FaceManagerDialog(self.root, self.face_service)
        except Exception as e:
            ui_logger.error(f"❌ Error opening face manager: {e}")
            messagebox.showerror("Error", f"Failed to open face manager:\n{str(e)}")

    def backup_face_database(self):
        """Backup face database"""
        try:
            file_path = filedialog.asksaveasfilename(
                title="Backup Face Database",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )

            if file_path:
                if self.face_service.backup_face_database(file_path):
                    messagebox.showinfo("Success", f"✅ Face database backed up to:\n{file_path}")
                    self.update_status(f"✅ Backup saved: {file_path}")
                else:
                    messagebox.showerror("Error", "Failed to backup face database")

        except Exception as e:
            ui_logger.error(f"❌ Error backing up face database: {e}")
            messagebox.showerror("Error", f"Failed to backup face database:\n{str(e)}")

    def restore_face_database(self):
        """Restore face database"""
        try:
            result = messagebox.askyesno(
                "Confirm Restore",
                "Restoring will replace the current face database.\n\nDo you want to continue?"
            )

            if not result:
                return

            file_path = filedialog.askopenfilename(
                title="Restore Face Database",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )

            if file_path:
                if self.face_service.restore_face_database(file_path):
                    messagebox.showinfo("Success", f"✅ Face database restored from:\n{file_path}")
                    self.update_status(f"✅ Restored from: {file_path}")

                    # Update system info
                    face_count = len(self.face_service.face_db)
                    camera_count = len(self.camera_service.cameras)
                    self.system_info_label.config(text=f"📹 {camera_count} cameras | 👤 {face_count} faces")
                else:
                    messagebox.showerror("Error", "Failed to restore face database")

        except Exception as e:
            ui_logger.error(f"❌ Error restoring face database: {e}")
            messagebox.showerror("Error", f"Failed to restore face database:\n{str(e)}")

    def update_streams_counter(self):
        """Update active streams counter"""
        count = len(self.active_streams)
        self.streams_label.config(text=f"📹 {count} streams")

    def update_status(self, message: str):
        """Update status message"""
        self.status_label.config(text=message)

    def show_progress(self, show: bool = True):
        """Show/hide progress bar"""
        if show:
            self.progress_bar.pack(side=tk.RIGHT, padx=(5, 0))
            self.progress_bar.start()
        else:
            self.progress_bar.stop()
            self.progress_bar.pack_forget()

    # Placeholder methods for menu items
    def test_camera_connection(self):
        messagebox.showinfo("Info", "Test camera connection feature will be implemented")

    def import_settings(self):
        messagebox.showinfo("Info", "Import settings feature will be implemented")

    def export_settings(self):
        messagebox.showinfo("Info", "Export settings feature will be implemented")

    def face_settings(self):
        messagebox.showinfo("Info", "Face recognition settings will be implemented")

    def show_camera_list(self):
        messagebox.showinfo("Info", "Camera list view will be implemented")

    def show_statistics(self):
        try:
            stats = self.face_service.get_statistics()
            camera_stats = self.camera_service.get_camera_stats()

            stats_text = f"""System Statistics:

📹 Cameras:
• Total: {camera_stats['total']}
• Active: {camera_stats['active']}
• Streaming: {camera_stats['streaming']}

👤 Faces:
• Total: {stats['total_faces']}
• Male: {stats['male_count']}
• Female: {stats['female_count']}
• Average Age: {stats['average_age']} years

⚙️ System:
• Recognition Threshold: {stats['recognition_threshold']}
• InsightFace: {'Initialized' if stats['is_initialized'] else 'Not Available'}
• Database Size: {stats['database_size']} bytes
"""

            messagebox.showinfo("System Statistics", stats_text)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load statistics:\n{str(e)}")

    def change_theme(self):
        messagebox.showinfo("Info", "Theme selection will be implemented")

    def open_preferences(self):
        messagebox.showinfo("Info", "Preferences dialog will be implemented")

    def take_snapshot(self, tab_index: int):
        messagebox.showinfo("Info", "Snapshot feature will be implemented")

    def camera_settings(self, tab_index: int):
        messagebox.showinfo("Info", "Camera settings will be implemented")

    def show_user_guide(self):
        messagebox.showinfo("Info", "User guide will be implemented")

    def show_troubleshooting(self):
        messagebox.showinfo("Info", "Troubleshooting guide will be implemented")

    def show_about(self):
        about_text = """🎥 Camera RTSP Application với InsightFace

Version: 1.0.0
Author: AI Assistant
License: MIT

Features:
• Multi-camera RTSP streaming
• Real-time face recognition
• Face database management
• Performance monitoring

Built with:
• Python 3.8+
• OpenCV
• InsightFace
• Tkinter

© 2024 Camera RTSP Application"""

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