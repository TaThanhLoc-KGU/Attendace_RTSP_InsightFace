"""
Enhanced camera selection dialog with webcam support and classroom integration
"""
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from typing import List, Optional
import threading
import json

from services.camera_service import CameraService, Camera
from config.config import config
from utils.logger import ui_logger


class CameraSelectionDialog:
    """Enhanced dialog for camera and webcam selection"""

    def __init__(self, parent, camera_service: CameraService):
        self.parent = parent
        self.camera_service = camera_service
        self.selected_camera: Optional[Camera] = None
        self.result = None
        self.dialog = None
        self.tree = None
        self.notebook = None

        # Controls
        self.refresh_btn = None
        self.select_btn = None
        self.test_btn = None
        self.add_webcam_btn = None

        # Webcam management
        self.available_webcams = []
        self.classroom_name_var = tk.StringVar()

        self.create_dialog()
        self.load_data()

        ui_logger.info("📹 Enhanced camera selection dialog created")

    def create_dialog(self):
        """Create enhanced dialog UI with tabs"""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("📹 Select Camera - Enhanced with Webcam Support")
        self.dialog.geometry("900x700")
        self.dialog.resizable(True, True)
        self.dialog.transient(self.parent)
        self.dialog.grab_set()

        # Center dialog
        self.center_dialog()

        # Configure grid weights
        self.dialog.grid_rowconfigure(1, weight=1)
        self.dialog.grid_columnconfigure(0, weight=1)

        # Create sections
        self.create_menu_bar()
        self.create_header()
        self.create_tabbed_interface()
        self.create_buttons()
        self.create_status_bar()

        # Bind events
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_cancel)
        self.dialog.bind('<Return>', self.on_select)
        self.dialog.bind('<Escape>', self.on_cancel)

    def create_menu_bar(self):
        """Create menu bar for enhanced functionality"""
        menubar = tk.Menu(self.dialog)
        self.dialog.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="📤 Export Camera List", command=self.export_camera_list)
        file_menu.add_command(label="📥 Import Custom Camera", command=self.import_custom_camera)
        file_menu.add_separator()
        file_menu.add_command(label="❌ Close", command=self.on_cancel)

        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="🔍 Detect All Webcams", command=self.detect_webcams)
        tools_menu.add_command(label="🔄 Refresh All", command=self.refresh_all)
        tools_menu.add_command(label="🧪 Batch Test Cameras", command=self.batch_test_cameras)
        tools_menu.add_separator()
        tools_menu.add_command(label="🤖 Auto-Optimize Settings", command=self.auto_detect_optimal_settings)
        tools_menu.add_command(label="⚙️ Advanced Webcam Settings", command=self.advanced_webcam_settings)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="ℹ️ About", command=self.show_about)
        help_menu.add_command(label="🆘 Usage Guide", command=self.show_usage_guide)

    def show_about(self):
        """Show about dialog"""
        about_text = """📹 Enhanced Camera Selection Dialog

Version: 2.0.0
Features:
• RTSP camera management
• Webcam device detection  
• Classroom webcam setup
• Performance optimization
• Real-time testing

Supports:
✅ RTSP/IP cameras
✅ USB webcams
✅ Integrated webcams
✅ High FPS optimization
"""
        messagebox.showinfo("About", about_text)

    def show_usage_guide(self):
        """Show usage guide"""
        guide_text = """🆘 Usage Guide

RTSP Cameras Tab:
• View cameras from backend database
• Double-click to select
• Right-click for context menu
• Test connection before use

Webcams Tab:
• Click 'Detect Webcams' to find devices
• Select any available webcam
• Test before selection

Classroom Webcam Tab:
• Enter classroom name
• Select webcam from dropdown
• Click 'Add Classroom Webcam'
• Camera will be added automatically

Tips:
💡 Use context menu for additional options
💡 Test cameras before selection
💡 Check performance settings for optimization
"""
        messagebox.showinfo("Usage Guide", guide_text)

    def center_dialog(self):
        """Center dialog on screen"""
        self.dialog.update_idletasks()
        width = self.dialog.winfo_width()
        height = self.dialog.winfo_height()
        x = (self.dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (height // 2)
        self.dialog.geometry(f"{width}x{height}+{x}+{y}")

    def create_header(self):
        """Create header section"""
        header_frame = ttk.Frame(self.dialog)
        header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=20)

        # Title with enhanced styling
        title_label = ttk.Label(
            header_frame,
            text="📹 Camera & Webcam Selection",
            font=('Arial', 16, 'bold')
        )
        title_label.pack(side=tk.LEFT)

        # Quick stats
        stats_frame = ttk.Frame(header_frame)
        stats_frame.pack(side=tk.RIGHT)

        self.stats_label = ttk.Label(
            stats_frame,
            text="Loading...",
            font=('Arial', 9),
            foreground='blue'
        )
        self.stats_label.pack()

    def create_tabbed_interface(self):
        """Create tabbed interface for different camera types"""
        # Create notebook
        self.notebook = ttk.Notebook(self.dialog)
        self.notebook.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))

        # Tab 1: RTSP Cameras from Backend
        self.rtsp_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.rtsp_frame, text="📡 RTSP Cameras")
        self.create_rtsp_tab()

        # Tab 2: Webcam Devices
        self.webcam_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.webcam_frame, text="🎥 Webcams")
        self.create_webcam_tab()

        # Tab 3: Add Webcam for Classroom
        self.classroom_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.classroom_frame, text="🏫 Add Classroom Webcam")
        self.create_classroom_tab()

    def create_rtsp_tab(self):
        """Create RTSP cameras tab"""
        # Instructions
        inst_label = ttk.Label(
            self.rtsp_frame,
            text="📡 RTSP cameras from backend database:",
            font=('Arial', 10, 'bold')
        )
        inst_label.pack(anchor=tk.W, padx=10, pady=(10, 5))

        # Create treeview frame with scrollbars
        tree_frame = ttk.Frame(self.rtsp_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Configure grid weights
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        # Create treeview
        columns = ('ID', 'Name', 'Location', 'RTSP URL', 'Status')
        self.rtsp_tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=10)

        # Configure columns
        self.rtsp_tree.heading('ID', text='ID')
        self.rtsp_tree.heading('Name', text='Camera Name')
        self.rtsp_tree.heading('Location', text='Location')
        self.rtsp_tree.heading('RTSP URL', text='RTSP URL')
        self.rtsp_tree.heading('Status', text='Status')

        # Set column widths
        self.rtsp_tree.column('ID', width=50)
        self.rtsp_tree.column('Name', width=200)
        self.rtsp_tree.column('Location', width=150)
        self.rtsp_tree.column('RTSP URL', width=250)
        self.rtsp_tree.column('Status', width=100)

        # Scrollbars
        v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.rtsp_tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.rtsp_tree.xview)
        self.rtsp_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        # Grid layout
        self.rtsp_tree.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")

        # Bind events
        self.rtsp_tree.bind('<Double-1>', self.on_select)
        self.rtsp_tree.bind('<<TreeviewSelect>>', self.on_rtsp_selection_change)

        # Create context menus
        self.create_context_menus()

    def create_webcam_tab(self):
        """Create webcam devices tab"""
        # Instructions
        inst_label = ttk.Label(
            self.webcam_frame,
            text="🎥 Available webcam devices on this computer:",
            font=('Arial', 10, 'bold')
        )
        inst_label.pack(anchor=tk.W, padx=10, pady=(10, 5))

        # Webcam detection frame
        detect_frame = ttk.Frame(self.webcam_frame)
        detect_frame.pack(fill=tk.X, padx=10, pady=5)

        detect_btn = ttk.Button(
            detect_frame,
            text="🔍 Detect Webcams",
            command=self.detect_webcams
        )
        detect_btn.pack(side=tk.LEFT)

        self.webcam_status_label = ttk.Label(
            detect_frame,
            text="Click 'Detect Webcams' to find available devices",
            font=('Arial', 9),
            foreground='gray'
        )
        self.webcam_status_label.pack(side=tk.LEFT, padx=(10, 0))

        # Webcam list frame
        webcam_list_frame = ttk.Frame(self.webcam_frame)
        webcam_list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Configure grid weights
        webcam_list_frame.grid_rowconfigure(0, weight=1)
        webcam_list_frame.grid_columnconfigure(0, weight=1)

        # Create webcam treeview
        webcam_columns = ('Index', 'Name', 'Resolution', 'FPS', 'Status')
        self.webcam_tree = ttk.Treeview(webcam_list_frame, columns=webcam_columns, show='headings', height=8)

        # Configure columns
        for col in webcam_columns:
            self.webcam_tree.heading(col, text=col)

        self.webcam_tree.column('Index', width=60)
        self.webcam_tree.column('Name', width=150)
        self.webcam_tree.column('Resolution', width=100)
        self.webcam_tree.column('FPS', width=60)
        self.webcam_tree.column('Status', width=100)

        # Scrollbars for webcam tree
        webcam_v_scrollbar = ttk.Scrollbar(webcam_list_frame, orient=tk.VERTICAL, command=self.webcam_tree.yview)
        self.webcam_tree.configure(yscrollcommand=webcam_v_scrollbar.set)

        # Grid layout
        self.webcam_tree.grid(row=0, column=0, sticky="nsew")
        webcam_v_scrollbar.grid(row=0, column=1, sticky="ns")

        # Bind events
        self.webcam_tree.bind('<Double-1>', self.on_select)
        self.webcam_tree.bind('<<TreeviewSelect>>', self.on_webcam_selection_change)

    def create_classroom_tab(self):
        """Create classroom webcam addition tab"""
        # Instructions
        inst_label = ttk.Label(
            self.classroom_frame,
            text="🏫 Add webcam for a specific classroom:",
            font=('Arial', 12, 'bold')
        )
        inst_label.pack(anchor=tk.W, padx=20, pady=(20, 10))

        info_label = ttk.Label(
            self.classroom_frame,
            text="This feature allows you to quickly add a webcam for classroom attendance monitoring.",
            font=('Arial', 9),
            foreground='gray'
        )
        info_label.pack(anchor=tk.W, padx=20, pady=(0, 20))

        # Classroom name frame
        name_frame = ttk.LabelFrame(self.classroom_frame, text="📝 Classroom Information", padding=15)
        name_frame.pack(fill=tk.X, padx=20, pady=10)

        ttk.Label(name_frame, text="Classroom Name:").pack(anchor=tk.W)
        classroom_entry = ttk.Entry(name_frame, textvariable=self.classroom_name_var, font=('Arial', 10), width=40)
        classroom_entry.pack(fill=tk.X, pady=(5, 0))

        # Webcam selection frame
        webcam_select_frame = ttk.LabelFrame(self.classroom_frame, text="🎥 Select Webcam", padding=15)
        webcam_select_frame.pack(fill=tk.X, padx=20, pady=10)

        self.webcam_var = tk.StringVar()
        self.webcam_combo = ttk.Combobox(webcam_select_frame, textvariable=self.webcam_var, state="readonly")
        self.webcam_combo.pack(fill=tk.X, pady=5)

        refresh_webcam_btn = ttk.Button(
            webcam_select_frame,
            text="🔄 Refresh Webcam List",
            command=self.refresh_webcam_combo
        )
        refresh_webcam_btn.pack(pady=(10, 0))

        # Test and add frame
        action_frame = ttk.Frame(self.classroom_frame)
        action_frame.pack(fill=tk.X, padx=20, pady=20)

        test_webcam_btn = ttk.Button(
            action_frame,
            text="🔍 Test Webcam",
            command=self.test_selected_webcam
        )
        test_webcam_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.add_classroom_btn = ttk.Button(
            action_frame,
            text="➕ Add Classroom Webcam",
            command=self.add_classroom_webcam
        )
        self.add_classroom_btn.pack(side=tk.LEFT)

        # Initialize webcam combo
        self.refresh_webcam_combo()

    def create_buttons(self):
        """Create button panel"""
        button_frame = ttk.Frame(self.dialog)
        button_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=20)

        # Left buttons
        left_buttons = ttk.Frame(button_frame)
        left_buttons.pack(side=tk.LEFT)

        self.refresh_btn = ttk.Button(
            left_buttons,
            text="🔄 Refresh All",
            command=self.refresh_all,
            width=15
        )
        self.refresh_btn.pack(side=tk.LEFT, padx=5)

        self.test_btn = ttk.Button(
            left_buttons,
            text="🔍 Test Connection",
            command=self.test_camera_connection,
            width=15,
            state='disabled'
        )
        self.test_btn.pack(side=tk.LEFT, padx=5)

        # Right buttons
        right_buttons = ttk.Frame(button_frame)
        right_buttons.pack(side=tk.RIGHT)

        cancel_btn = ttk.Button(
            right_buttons,
            text="❌ Cancel",
            command=self.on_cancel,
            width=12
        )
        cancel_btn.pack(side=tk.RIGHT, padx=5)

        self.select_btn = ttk.Button(
            right_buttons,
            text="✅ Select",
            command=self.on_select,
            width=12,
            state='disabled'
        )
        self.select_btn.pack(side=tk.RIGHT, padx=5)

    def create_status_bar(self):
        """Create status bar"""
        self.status_label = ttk.Label(
            self.dialog,
            text="Ready - Select a camera from any tab",
            font=('Arial', 9),
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_label.grid(row=3, column=0, sticky="ew", padx=20, pady=(0, 10))

    def load_data(self):
        """Load all data"""
        self.update_status("Loading cameras and detecting webcams...")
        self.refresh_btn.config(state='disabled')

        # Run in thread
        thread = threading.Thread(target=self._load_data_thread, daemon=True)
        thread.start()

    def _load_data_thread(self):
        """Thread function to load data"""
        try:
            # Load RTSP cameras from backend
            result = self.camera_service.load_cameras_from_backend()

            # Detect webcams
            webcams = self.camera_service.detect_webcam_devices()

            self.dialog.after(0, self._update_all_data, result, webcams)
        except Exception as e:
            ui_logger.error(f"❌ Error loading data: {e}")
            self.dialog.after(0, self._on_load_error, str(e))

    def _update_all_data(self, cameras_result, webcams):
        """Update all data displays"""
        try:
            # Update RTSP cameras
            self._update_rtsp_cameras(cameras_result)

            # Update webcams
            self._update_webcam_list(webcams)

            # Update stats
            self.stats_label.config(text=self.format_camera_stats())

            # Get counts for status message
            cameras_count = cameras_result.get('count', 0)
            webcams_count = len(webcams)
            self.update_status(f"✅ Loaded {cameras_count} cameras and {webcams_count} webcams")

        except Exception as e:
            ui_logger.error(f"❌ Error updating data: {e}")
            self.update_status("❌ Error updating camera list")
        finally:
            self.refresh_btn.config(state='normal')

    def _update_rtsp_cameras(self, result):
        """Update RTSP camera list"""
        # Clear existing items
        for item in self.rtsp_tree.get_children():
            self.rtsp_tree.delete(item)

        if result.get('success'):
            cameras = result.get('cameras', [])
            rtsp_cameras = [cam for cam in cameras if not cam.get('is_webcam', False)]

            for camera_data in rtsp_cameras:
                status = "✅ Active" if camera_data.get('active') else "❌ Inactive"
                rtsp_preview = camera_data.get('rtspPreview', '')

                self.rtsp_tree.insert('', tk.END, values=(
                    camera_data.get('id'),
                    camera_data.get('name'),
                    camera_data.get('location'),
                    rtsp_preview,
                    status
                ))

    def _update_webcam_list(self, webcams):
        """Update webcam list"""
        # Clear existing items
        for item in self.webcam_tree.get_children():
            self.webcam_tree.delete(item)

        self.available_webcams = webcams

        for webcam in webcams:
            self.webcam_tree.insert('', tk.END, values=(
                webcam['id'],
                webcam['name'],
                webcam['resolution'],
                webcam['fps'],
                "✅ Available"
            ))

        # Update webcam combo
        self.refresh_webcam_combo()

    def detect_webcams(self):
        """Detect webcam devices"""
        self.webcam_status_label.config(text="🔍 Detecting webcams...")

        thread = threading.Thread(target=self._detect_webcams_thread, daemon=True)
        thread.start()

    def _detect_webcams_thread(self):
        """Thread function to detect webcams"""
        try:
            webcams = self.camera_service.detect_webcam_devices()
            self.dialog.after(0, self._on_webcams_detected, webcams)
        except Exception as e:
            ui_logger.error(f"❌ Error detecting webcams: {e}")
            self.dialog.after(0, self._on_webcam_detection_error, str(e))

    def _on_webcams_detected(self, webcams):
        """Handle webcam detection result"""
        self._update_webcam_list(webcams)
        self.webcam_status_label.config(text=f"✅ Found {len(webcams)} webcam devices")

    def _on_webcam_detection_error(self, error):
        """Handle webcam detection error"""
        self.webcam_status_label.config(text=f"❌ Error: {error}")

    def refresh_webcam_combo(self):
        """Refresh webcam combobox"""
        webcam_options = [f"Webcam {w['id']} - {w['resolution']}" for w in self.available_webcams]
        self.webcam_combo['values'] = webcam_options
        if webcam_options:
            self.webcam_combo.current(0)

    def test_selected_webcam(self):
        """Test selected webcam"""
        if not self.webcam_var.get():
            messagebox.showwarning("Warning", "Please select a webcam first!")
            return

        # Extract webcam index
        try:
            webcam_text = self.webcam_var.get()
            webcam_index = int(webcam_text.split()[1])

            self.update_status(f"🔍 Testing webcam {webcam_index}...")

            # Test in thread
            thread = threading.Thread(
                target=self._test_webcam_thread,
                args=(webcam_index,),
                daemon=True
            )
            thread.start()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to parse webcam selection: {e}")

    def _test_webcam_thread(self, webcam_index):
        """Thread function to test webcam"""
        try:
            success = self.camera_service.test_webcam_connection(webcam_index)
            self.dialog.after(0, self._on_webcam_test_result, webcam_index, success)
        except Exception as e:
            ui_logger.error(f"❌ Webcam test error: {e}")
            self.dialog.after(0, self._on_webcam_test_result, webcam_index, False, str(e))

    def _on_webcam_test_result(self, webcam_index, success, error=None):
        """Handle webcam test result"""
        if success:
            self.update_status(f"✅ Webcam {webcam_index} test successful")
            messagebox.showinfo("Success", f"✅ Webcam {webcam_index} is working correctly!")
        else:
            error_msg = f"❌ Webcam {webcam_index} test failed"
            if error:
                error_msg += f": {error}"
            self.update_status(error_msg)
            messagebox.showerror("Test Failed", error_msg)

    def add_classroom_webcam(self):
        """Add webcam for classroom"""
        classroom_name = self.classroom_name_var.get().strip()
        if not classroom_name:
            messagebox.showwarning("Warning", "Please enter a classroom name!")
            return

        if not self.webcam_var.get():
            messagebox.showwarning("Warning", "Please select a webcam!")
            return

        try:
            # Extract webcam index
            webcam_text = self.webcam_var.get()
            webcam_index = int(webcam_text.split()[1])

            # Add webcam to service
            camera = self.camera_service.add_webcam_to_classroom(classroom_name, webcam_index)

            # Set as selected camera
            self.selected_camera = camera
            self.result = 'ok'

            messagebox.showinfo(
                "Success",
                f"✅ Webcam {webcam_index} added for classroom '{classroom_name}'\n\nCamera will be selected automatically."
            )

            self.dialog.destroy()

        except Exception as e:
            ui_logger.error(f"❌ Error adding classroom webcam: {e}")
            messagebox.showerror("Error", f"Failed to add classroom webcam:\n{str(e)}")

    def on_rtsp_selection_change(self, event):
        """Handle RTSP camera selection change"""
        selection = self.rtsp_tree.selection()
        has_selection = bool(selection)

        self.select_btn.config(state='normal' if has_selection else 'disabled')
        self.test_btn.config(state='normal' if has_selection else 'disabled')

        if has_selection:
            item = self.rtsp_tree.item(selection[0])
            camera_name = item['values'][1]
            self.update_status(f"Selected RTSP camera: {camera_name}")

    def on_webcam_selection_change(self, event):
        """Handle webcam selection change"""
        selection = self.webcam_tree.selection()
        has_selection = bool(selection)

        self.select_btn.config(state='normal' if has_selection else 'disabled')
        self.test_btn.config(state='normal' if has_selection else 'disabled')

        if has_selection:
            item = self.webcam_tree.item(selection[0])
            webcam_name = item['values'][1]
            self.update_status(f"Selected webcam: {webcam_name}")

    def on_select(self, event=None):
        """Handle camera selection"""
        current_tab = self.notebook.index(self.notebook.select())

        if current_tab == 0:  # RTSP tab
            self._select_rtsp_camera()
        elif current_tab == 1:  # Webcam tab
            self._select_webcam()
        else:
            messagebox.showinfo("Info", "Use 'Add Classroom Webcam' button in the classroom tab.")

    def _select_rtsp_camera(self):
        """Select RTSP camera"""
        selection = self.rtsp_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select an RTSP camera!")
            return

        item = self.rtsp_tree.item(selection[0])
        camera_id = item['values'][0]

        camera = self.camera_service.get_camera_by_id(camera_id)
        if camera:
            self.selected_camera = camera
            self.result = 'ok'
            ui_logger.info(f"✅ Selected RTSP camera: {camera.name}")
            self.dialog.destroy()
        else:
            messagebox.showerror("Error", f"Camera with ID {camera_id} not found!")

    def _select_webcam(self):
        """Select webcam"""
        selection = self.webcam_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a webcam!")
            return

        item = self.webcam_tree.item(selection[0])
        webcam_index = item['values'][0]

        # Create temporary webcam camera
        camera = Camera(
            id=9000 + webcam_index,  # Temporary high ID
            name=f"Webcam {webcam_index}",
            rtsp_url=str(webcam_index),
            location="Webcam Device",
            active=True,
            camera_type="webcam",
            device_index=webcam_index
        )

        self.selected_camera = camera
        self.result = 'ok'
        ui_logger.info(f"✅ Selected webcam: {camera.name}")
        self.dialog.destroy()

    def test_camera_connection(self):
        """Test camera connection"""
        current_tab = self.notebook.index(self.notebook.select())

        if current_tab == 0:  # RTSP tab
            self._test_rtsp_camera()
        elif current_tab == 1:  # Webcam tab
            self._test_webcam_from_list()

    def _test_rtsp_camera(self):
        """Test RTSP camera connection"""
        selection = self.rtsp_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select an RTSP camera to test!")
            return

        item = self.rtsp_tree.item(selection[0])
        camera_id = item['values'][0]
        camera = self.camera_service.get_camera_by_id(camera_id)

        if not camera:
            messagebox.showerror("Error", "Camera not found!")
            return

        self.update_status("🔍 Testing RTSP connection...")
        self.test_btn.config(state='disabled')

        thread = threading.Thread(
            target=self._test_connection_thread,
            args=(camera,),
            daemon=True
        )
        thread.start()

    def _test_webcam_from_list(self):
        """Test webcam from list"""
        selection = self.webcam_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a webcam to test!")
            return

        item = self.webcam_tree.item(selection[0])
        webcam_index = item['values'][0]

        self.update_status(f"🔍 Testing webcam {webcam_index}...")
        self.test_btn.config(state='disabled')

        thread = threading.Thread(
            target=self._test_webcam_thread,
            args=(webcam_index,),
            daemon=True
        )
        thread.start()

    def _test_connection_thread(self, camera: Camera):
        """Thread function to test connection"""
        try:
            success = self.camera_service.validate_camera_connection(camera)
            self.dialog.after(0, self._on_test_result, camera, success)
        except Exception as e:
            ui_logger.error(f"❌ Test connection error: {e}")
            self.dialog.after(0, self._on_test_result, camera, False, str(e))

    def _on_test_result(self, camera: Camera, success: bool, error: str = None):
        """Handle test result"""
        self.test_btn.config(state='normal')

        if success:
            self.update_status(f"✅ Connection successful: {camera.name}")
            messagebox.showinfo("Success", f"✅ Connection to '{camera.name}' successful!")
        else:
            error_msg = f"❌ Connection failed: {error}" if error else "❌ Connection failed"
            self.update_status(error_msg)
            messagebox.showerror("Connection Failed", f"Cannot connect to '{camera.name}':\n{error or 'Unknown error'}")

    def refresh_all(self):
        """Refresh all data"""
        ui_logger.info("🔄 Refreshing all camera data")
        self.load_data()

    def _on_load_error(self, error_message):
        """Handle load error"""
        self.update_status(f"❌ Error: {error_message}")
        messagebox.showerror("Error", f"Failed to load data:\n{error_message}")
        self.refresh_btn.config(state='normal')

    def on_cancel(self):
        """Cancel dialog"""
        self.result = 'cancel'
        self.dialog.destroy()
        ui_logger.info("❌ Camera selection canceled")

    def update_status(self, message: str):
        """Update status message"""
        self.status_label.config(text=message)

    def copy_rtsp_url(self):
        """Copy RTSP URL to clipboard"""
        current_tab = self.notebook.index(self.notebook.select())

        if current_tab == 0:  # RTSP tab
            selection = self.rtsp_tree.selection()
            if not selection:
                return

            item = self.rtsp_tree.item(selection[0])
            camera_id = item['values'][0]
            camera = self.camera_service.get_camera_by_id(camera_id)

            if camera:
                self.dialog.clipboard_clear()
                self.dialog.clipboard_append(camera.rtsp_url)
                self.update_status(f"📋 RTSP URL copied to clipboard")

    def show_camera_info(self):
        """Show detailed camera information"""
        current_tab = self.notebook.index(self.notebook.select())

        if current_tab == 0:  # RTSP tab
            selection = self.rtsp_tree.selection()
            if not selection:
                return

            item = self.rtsp_tree.item(selection[0])
            camera_id = item['values'][0]
            camera = self.camera_service.get_camera_by_id(camera_id)

            if camera:
                info = f"""RTSP Camera Information:

ID: {camera.id}
Name: {camera.name}
Type: {camera.camera_type.upper()}
Location: {camera.location}
RTSP URL: {camera.rtsp_url}
HLS URL: {camera.hls_url}
Status: {'Active' if camera.active else 'Inactive'}
Schedule: {camera.current_schedule if camera.current_schedule else 'None'}
"""
                messagebox.showinfo(f"Camera Info - {camera.name}", info)

        elif current_tab == 1:  # Webcam tab
            selection = self.webcam_tree.selection()
            if not selection:
                return

            item = self.webcam_tree.item(selection[0])
            webcam_index = item['values'][0]

            # Find webcam info
            webcam_info = None
            for webcam in self.available_webcams:
                if webcam['id'] == webcam_index:
                    webcam_info = webcam
                    break

            if webcam_info:
                info = f"""Webcam Device Information:

Device Index: {webcam_info['id']}
Name: {webcam_info['name']}
Type: Webcam Device
Resolution: {webcam_info['resolution']}
FPS: {webcam_info['fps']}
Status: Available
URL: {webcam_info['url']}
"""
                messagebox.showinfo(f"Webcam Info - {webcam_info['name']}", info)

    def camera_settings(self):
        """Open camera settings"""
        current_tab = self.notebook.index(self.notebook.select())

        if current_tab == 0:  # RTSP tab
            selection = self.rtsp_tree.selection()
            if not selection:
                return

            item = self.rtsp_tree.item(selection[0])
            camera_name = item['values'][1]
            messagebox.showinfo("Settings", f"Settings for RTSP camera '{camera_name}' will be implemented in future version")

        elif current_tab == 1:  # Webcam tab
            selection = self.webcam_tree.selection()
            if not selection:
                return

            item = self.webcam_tree.item(selection[0])
            webcam_name = item['values'][1]
            messagebox.showinfo("Settings", f"Settings for '{webcam_name}' will be implemented in future version")

    def create_context_menus(self):
        """Create context menus for treeviews"""
        # RTSP context menu
        self.rtsp_context_menu = tk.Menu(self.dialog, tearoff=0)
        self.rtsp_context_menu.add_command(label="🔍 Test Connection", command=self._test_rtsp_camera)
        self.rtsp_context_menu.add_command(label="📋 Copy RTSP URL", command=self.copy_rtsp_url)
        self.rtsp_context_menu.add_separator()
        self.rtsp_context_menu.add_command(label="ℹ️ Camera Info", command=self.show_camera_info)
        self.rtsp_context_menu.add_command(label="⚙️ Settings", command=self.camera_settings)

        # Webcam context menu
        self.webcam_context_menu = tk.Menu(self.dialog, tearoff=0)
        self.webcam_context_menu.add_command(label="🔍 Test Webcam", command=self._test_webcam_from_list)
        self.webcam_context_menu.add_separator()
        self.webcam_context_menu.add_command(label="ℹ️ Webcam Info", command=self.show_camera_info)
        self.webcam_context_menu.add_command(label="⚙️ Settings", command=self.camera_settings)

        # Bind context menus
        self.rtsp_tree.bind('<Button-3>', self.show_rtsp_context_menu)
        self.webcam_tree.bind('<Button-3>', self.show_webcam_context_menu)

    def show_rtsp_context_menu(self, event):
        """Show RTSP context menu"""
        item = self.rtsp_tree.identify_row(event.y)
        if item:
            self.rtsp_tree.selection_set(item)
            self.rtsp_context_menu.post(event.x_root, event.y_root)

    def show_webcam_context_menu(self, event):
        """Show webcam context menu"""
        item = self.webcam_tree.identify_row(event.y)
        if item:
            self.webcam_tree.selection_set(item)
            self.webcam_context_menu.post(event.x_root, event.y_root)

    def export_camera_list(self):
        """Export camera list to file"""
        try:
            from tkinter import filedialog
            import json

            # Get camera data
            cameras_data = []

            # RTSP cameras
            for item in self.rtsp_tree.get_children():
                values = self.rtsp_tree.item(item)['values']
                cameras_data.append({
                    'type': 'rtsp',
                    'id': values[0],
                    'name': values[1],
                    'location': values[2],
                    'rtsp_url': values[3],
                    'status': values[4]
                })

            # Webcam devices
            for webcam in self.available_webcams:
                cameras_data.append({
                    'type': 'webcam',
                    'id': webcam['id'],
                    'name': webcam['name'],
                    'resolution': webcam['resolution'],
                    'fps': webcam['fps'],
                    'device_index': webcam['id']
                })

            # Save file dialog
            filename = filedialog.asksaveasfilename(
                title="Export Camera List",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )

            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(cameras_data, f, indent=2, ensure_ascii=False)

                messagebox.showinfo("Export Success", f"Camera list exported to:\n{filename}")
                self.update_status(f"✅ Exported {len(cameras_data)} cameras")

        except Exception as e:
            ui_logger.error(f"❌ Export error: {e}")
            messagebox.showerror("Export Error", f"Failed to export camera list:\n{str(e)}")

    def import_custom_camera(self):
        """Import custom RTSP camera"""
        from tkinter import simpledialog

        try:
            # Get camera details
            name = simpledialog.askstring("Camera Name", "Enter camera name:")
            if not name:
                return

            rtsp_url = simpledialog.askstring("RTSP URL", "Enter RTSP URL:")
            if not rtsp_url:
                return

            location = simpledialog.askstring("Location", "Enter camera location:", initialvalue="Custom Location")
            if not location:
                location = "Custom Location"

            # Add custom camera
            camera = self.camera_service.add_custom_camera(name, rtsp_url, location)

            # Refresh RTSP list
            self._update_rtsp_cameras({'success': True, 'cameras': [self.camera_service._camera_to_dict(cam) for cam in self.camera_service.cameras]})

            messagebox.showinfo("Success", f"Custom camera '{name}' added successfully!")
            self.update_status(f"✅ Added custom camera: {name}")

        except Exception as e:
            ui_logger.error(f"❌ Import custom camera error: {e}")
            messagebox.showerror("Error", f"Failed to add custom camera:\n{str(e)}")

    def advanced_webcam_settings(self):
        """Show advanced webcam settings dialog"""
        try:
            settings_window = tk.Toplevel(self.dialog)
            settings_window.title("🎥 Advanced Webcam Settings")
            settings_window.geometry("400x300")
            settings_window.transient(self.dialog)
            settings_window.grab_set()

            # Center window
            settings_window.update_idletasks()
            x = (settings_window.winfo_screenwidth() // 2) - (200)
            y = (settings_window.winfo_screenheight() // 2) - (150)
            settings_window.geometry(f"400x300+{x}+{y}")

            # Settings content
            main_frame = ttk.Frame(settings_window, padding=20)
            main_frame.pack(fill=tk.BOTH, expand=True)

            ttk.Label(main_frame, text="🎥 Webcam Performance Settings", font=('Arial', 12, 'bold')).pack(pady=(0, 20))

            # Resolution settings
            res_frame = ttk.LabelFrame(main_frame, text="Resolution Settings", padding=10)
            res_frame.pack(fill=tk.X, pady=5)

            ttk.Label(res_frame, text="Processing Resolution:").pack(anchor=tk.W)
            res_var = tk.StringVar(value=f"{config.WEBCAM_PROCESSING_WIDTH}x{config.WEBCAM_PROCESSING_HEIGHT}")
            res_combo = ttk.Combobox(res_frame, textvariable=res_var, values=[
                "160x120", "320x240", "640x480", "800x600"
            ], state="readonly")
            res_combo.pack(fill=tk.X, pady=5)

            # FPS settings
            fps_frame = ttk.LabelFrame(main_frame, text="Performance Settings", padding=10)
            fps_frame.pack(fill=tk.X, pady=5)

            ttk.Label(fps_frame, text="Target FPS:").pack(anchor=tk.W)
            fps_var = tk.StringVar(value=str(config.WEBCAM_FPS_TARGET))
            fps_combo = ttk.Combobox(fps_frame, textvariable=fps_var, values=[
                "10", "15", "20", "25", "30"
            ], state="readonly")
            fps_combo.pack(fill=tk.X, pady=5)

            # Buttons
            btn_frame = ttk.Frame(main_frame)
            btn_frame.pack(fill=tk.X, pady=20)

            def apply_settings():
                try:
                    # Parse resolution
                    width, height = map(int, res_var.get().split('x'))
                    fps = int(fps_var.get())

                    # Update config (in memory)
                    config.WEBCAM_PROCESSING_WIDTH = width
                    config.WEBCAM_PROCESSING_HEIGHT = height
                    config.WEBCAM_FPS_TARGET = fps

                    messagebox.showinfo("Applied", "Settings applied! Will take effect for new webcam streams.")
                    settings_window.destroy()

                except Exception as e:
                    messagebox.showerror("Error", f"Invalid settings: {e}")

            ttk.Button(btn_frame, text="✅ Apply", command=apply_settings).pack(side=tk.RIGHT, padx=5)
            ttk.Button(btn_frame, text="❌ Cancel", command=settings_window.destroy).pack(side=tk.RIGHT)

        except Exception as e:
            ui_logger.error(f"❌ Advanced settings error: {e}")
            messagebox.showerror("Error", f"Failed to open advanced settings:\n{str(e)}")

    def update_status(self, message: str):
        """Update status message"""
        self.status_label.config(text=message)

    def validate_classroom_input(self) -> bool:
        """Validate classroom input"""
        classroom_name = self.classroom_name_var.get().strip()
        if not classroom_name:
            messagebox.showwarning("Validation Error", "Please enter a classroom name!")
            return False

        if not self.webcam_var.get():
            messagebox.showwarning("Validation Error", "Please select a webcam device!")
            return False

        return True

    def get_selected_webcam_index(self) -> Optional[int]:
        """Get selected webcam index from combo"""
        try:
            webcam_text = self.webcam_var.get()
            if webcam_text:
                return int(webcam_text.split()[1])
        except (ValueError, IndexError):
            pass
        return None

    def format_camera_stats(self) -> str:
        """Format camera statistics for display"""
        rtsp_count = len(list(self.rtsp_tree.get_children()))
        webcam_count = len(self.available_webcams)
        total_count = rtsp_count + webcam_count

        return f"📊 Total: {total_count} | RTSP: {rtsp_count} | Webcams: {webcam_count}"

    def auto_detect_optimal_settings(self):
        """Auto-detect optimal settings for webcams"""
        try:
            import psutil

            # Get system info
            cpu_count = psutil.cpu_count()
            memory = psutil.virtual_memory()

            # Suggest settings based on system
            if cpu_count >= 8 and memory.total >= 8 * 1024**3:  # 8GB+ RAM, 8+ cores
                suggested_fps = 25
                suggested_resolution = "640x480"
            elif cpu_count >= 4 and memory.total >= 4 * 1024**3:  # 4GB+ RAM, 4+ cores
                suggested_fps = 20
                suggested_resolution = "416x312"
            else:
                suggested_fps = 15
                suggested_resolution = "320x240"

            message = f"""🤖 Auto-Detected Optimal Settings:

System Information:
• CPU Cores: {cpu_count}
• RAM: {memory.total // (1024**3):.1f} GB
• Available RAM: {memory.available // (1024**3):.1f} GB

Recommended Settings:
• Processing Resolution: {suggested_resolution}
• Target FPS: {suggested_fps}
• Performance Mode: {'High' if cpu_count >= 8 else 'Balanced' if cpu_count >= 4 else 'Economy'}

Apply these settings?"""

            result = messagebox.askyesno("Auto-Optimization", message)
            if result:
                # Apply settings
                width, height = map(int, suggested_resolution.split('x'))
                config.WEBCAM_PROCESSING_WIDTH = width
                config.WEBCAM_PROCESSING_HEIGHT = height
                config.WEBCAM_FPS_TARGET = suggested_fps

                self.update_status(f"✅ Applied optimal settings: {suggested_resolution} @ {suggested_fps}fps")

        except ImportError:
            messagebox.showinfo("Auto-Optimization", "psutil not available. Please install: pip install psutil")
        except Exception as e:
            ui_logger.error(f"❌ Auto-detection error: {e}")
            messagebox.showerror("Error", f"Auto-detection failed: {e}")

    def batch_test_cameras(self):
        """Batch test all cameras"""
        self.update_status("🔍 Starting batch camera test...")

        thread = threading.Thread(target=self._batch_test_thread, daemon=True)
        thread.start()

    def _batch_test_thread(self):
        """Thread function for batch testing"""
        try:
            results = []

            # Test RTSP cameras
            for item in self.rtsp_tree.get_children():
                values = self.rtsp_tree.item(item)['values']
                camera_id = values[0]
                camera = self.camera_service.get_camera_by_id(camera_id)

                if camera:
                    success = self.camera_service.validate_camera_connection(camera)
                    results.append({
                        'name': camera.name,
                        'type': 'RTSP',
                        'success': success
                    })

            # Test webcams
            for webcam in self.available_webcams:
                success = self.camera_service.test_webcam_connection(webcam['id'])
                results.append({
                    'name': webcam['name'],
                    'type': 'Webcam',
                    'success': success
                })

            self.dialog.after(0, self._on_batch_test_complete, results)

        except Exception as e:
            ui_logger.error(f"❌ Batch test error: {e}")
            self.dialog.after(0, self._on_batch_test_error, str(e))

    def _on_batch_test_complete(self, results):
        """Handle batch test completion"""
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]

        report = f"""📊 Batch Test Results:

✅ Successful: {len(successful)}/{len(results)}
❌ Failed: {len(failed)}/{len(results)}

Failed Cameras:
"""

        for camera in failed:
            report += f"• {camera['name']} ({camera['type']})\n"

        if not failed:
            report += "None - All cameras working!"

        messagebox.showinfo("Batch Test Results", report)
        self.update_status(f"✅ Batch test complete: {len(successful)}/{len(results)} working")

    def _on_batch_test_error(self, error):
        """Handle batch test error"""
        self.update_status(f"❌ Batch test failed: {error}")
        messagebox.showerror("Batch Test Error", f"Batch test failed:\n{error}")

    def get_result(self):
        """Get dialog result"""
        return self.result, self.selected_camera

    def cleanup(self):
        """Cleanup resources"""
        ui_logger.info("🧹 Cleaning up camera selection dialog")
        try:
            if hasattr(self, 'dialog') and self.dialog:
                self.dialog.destroy()
        except Exception as e:
            ui_logger.error(f"❌ Cleanup error: {e}")

    def __del__(self):
        """Destructor"""
        self.cleanup()


# Helper function for standalone testing
def test_camera_dialog():
    """Test function for camera dialog"""
    import sys
    sys.path.append('.')

    try:
        from services.backend_api import BackendAPI
        from services.camera_service import CameraService

        root = tk.Tk()
        root.withdraw()  # Hide main window

        # Create mock services
        backend_api = BackendAPI()
        camera_service = CameraService(backend_api)

        # Show dialog
        dialog = CameraSelectionDialog(root, camera_service)
        root.wait_window(dialog.dialog)

        # Get result
        result, camera = dialog.get_result()
        print(f"Result: {result}")
        if camera:
            print(f"Selected camera: {camera.name} ({camera.camera_type})")

        root.destroy()

    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_camera_dialog()