"""
Camera selection dialog for choosing cameras from database
"""
import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Optional
import threading

from services.camera_service import CameraService, Camera
from config.config import config
from utils.logger import ui_logger


class CameraSelectionDialog:
    """Dialog để chọn camera từ danh sách"""

    def __init__(self, parent, camera_service: CameraService):
        self.parent = parent
        self.camera_service = camera_service
        self.selected_camera: Optional[Camera] = None
        self.result = None
        self.dialog = None
        self.tree = None
        self.refresh_btn = None
        self.select_btn = None
        self.test_btn = None

        self.create_dialog()
        self.load_cameras()

        ui_logger.info("📹 Camera selection dialog created")

    def create_dialog(self):
        """Tạo dialog UI"""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("📹 Chọn Camera")
        self.dialog.geometry("800x600")
        self.dialog.resizable(True, True)
        self.dialog.transient(self.parent)
        self.dialog.grab_set()

        # Center dialog
        self.center_dialog()

        # Configure grid weights
        self.dialog.grid_rowconfigure(2, weight=1)
        self.dialog.grid_columnconfigure(0, weight=1)

        # Create sections
        self.create_header()
        self.create_camera_list()
        self.create_buttons()
        self.create_status_bar()

        # Bind events
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_cancel)
        self.dialog.bind('<Return>', self.on_select)
        self.dialog.bind('<Escape>', self.on_cancel)

    def center_dialog(self):
        """Center dialog on screen"""
        self.dialog.update_idletasks()
        width = self.dialog.winfo_width()
        height = self.dialog.winfo_height()
        x = (self.dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (height // 2)
        self.dialog.geometry(f"{width}x{height}+{x}+{y}")

    def create_header(self):
        """Tạo header section"""
        header_frame = ttk.Frame(self.dialog)
        header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=20)

        # Title
        title_label = ttk.Label(
            header_frame,
            text="📹 Chọn Camera để Stream",
            font=('Arial', 16, 'bold')
        )
        title_label.pack(pady=(0, 10))

        # Description
        desc_label = ttk.Label(
            header_frame,
            text="Chọn camera từ danh sách dưới đây để bắt đầu streaming video với face recognition",
            font=('Arial', 10),
            foreground='gray'
        )
        desc_label.pack()

    def create_camera_list(self):
        """Tạo danh sách cameras"""
        list_frame = ttk.LabelFrame(self.dialog, text="📋 Danh sách Cameras", padding=10)
        list_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)

        # Create treeview với scrollbars
        tree_frame = ttk.Frame(list_frame)
        tree_frame.grid(row=0, column=0, sticky="nsew")
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        # Treeview
        columns = ('ID', 'Name', 'Location', 'RTSP URL', 'Status')
        self.tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=15)

        # Configure columns
        self.tree.heading('ID', text='ID')
        self.tree.heading('Name', text='Tên Camera')
        self.tree.heading('Location', text='Vị Trí')
        self.tree.heading('RTSP URL', text='RTSP URL')
        self.tree.heading('Status', text='Trạng Thái')

        # Column widths
        self.tree.column('ID', width=50, anchor='center')
        self.tree.column('Name', width=180, anchor='w')
        self.tree.column('Location', width=150, anchor='w')
        self.tree.column('RTSP URL', width=250, anchor='w')
        self.tree.column('Status', width=100, anchor='center')

        # Scrollbars
        v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.tree.xview)

        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        # Grid layout
        self.tree.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")

        # Bind events
        self.tree.bind('<Double-1>', self.on_select)
        self.tree.bind('<Button-3>', self.show_context_menu)  # Right-click
        self.tree.bind('<<TreeviewSelect>>', self.on_selection_change)

        # Context menu
        self.create_context_menu()

    def create_context_menu(self):
        """Tạo context menu cho treeview"""
        self.context_menu = tk.Menu(self.dialog, tearoff=0)
        self.context_menu.add_command(label="🔍 Test Connection", command=self.test_camera_connection)
        self.context_menu.add_command(label="📋 Copy RTSP URL", command=self.copy_rtsp_url)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="ℹ️ Camera Info", command=self.show_camera_info)
        self.context_menu.add_command(label="⚙️ Settings", command=self.camera_settings)

    def show_context_menu(self, event):
        """Hiển thị context menu"""
        # Select item under cursor
        item = self.tree.identify_row(event.y)
        if item:
            self.tree.selection_set(item)
            self.context_menu.post(event.x_root, event.y_root)

    def create_buttons(self):
        """Tạo button panel"""
        button_frame = ttk.Frame(self.dialog)
        button_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=20)

        # Left buttons
        left_buttons = ttk.Frame(button_frame)
        left_buttons.pack(side=tk.LEFT)

        self.refresh_btn = ttk.Button(
            left_buttons,
            text="🔄 Refresh",
            command=self.refresh_cameras,
            width=12
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
        """Tạo status bar"""
        self.status_label = ttk.Label(
            self.dialog,
            text="Ready",
            font=('Arial', 9),
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_label.grid(row=3, column=0, sticky="ew", padx=20, pady=(0, 10))

    def load_cameras(self):
        """Load cameras từ service"""
        self.update_status("Loading cameras...")
        self.refresh_btn.config(state='disabled')

        # Run in thread to avoid blocking UI
        thread = threading.Thread(target=self._load_cameras_thread, daemon=True)
        thread.start()

    def _load_cameras_thread(self):
        """Thread function để load cameras"""
        try:
            result = self.camera_service.load_cameras_from_backend()
            self.dialog.after(0, self._update_camera_list, result)
        except Exception as e:
            ui_logger.error(f"❌ Error loading cameras: {e}")
            self.dialog.after(0, self._on_load_error, str(e))

    def _update_camera_list(self, result):
        """Update camera list với data từ backend"""
        try:
            # Clear existing items
            for item in self.tree.get_children():
                self.tree.delete(item)

            if result.get('success'):
                cameras = result.get('cameras', [])

                for camera_data in cameras:
                    # Status display
                    status = "✅ Active" if camera_data.get('active') else "❌ Inactive"

                    # RTSP URL preview
                    rtsp_url = camera_data.get('rtspUrl', '')
                    rtsp_preview = (rtsp_url[:40] + "...") if len(rtsp_url) > 40 else rtsp_url

                    # Insert into treeview
                    self.tree.insert('', tk.END, values=(
                        camera_data.get('id'),
                        camera_data.get('name'),
                        camera_data.get('location'),
                        rtsp_preview,
                        status
                    ))

                count = len(cameras)
                self.update_status(f"✅ Loaded {count} cameras")
                ui_logger.info(f"📹 Loaded {count} cameras in dialog")

            else:
                message = result.get('message', 'Unknown error')
                self.update_status(f"❌ Error: {message}")
                messagebox.showerror("Error", f"Failed to load cameras:\n{message}")

        except Exception as e:
            ui_logger.error(f"❌ Error updating camera list: {e}")
            self.update_status("❌ Error updating list")

        finally:
            self.refresh_btn.config(state='normal')

    def _on_load_error(self, error_message):
        """Handle load error"""
        self.update_status(f"❌ Error: {error_message}")
        messagebox.showerror("Error", f"Failed to load cameras:\n{error_message}")
        self.refresh_btn.config(state='normal')

    def refresh_cameras(self):
        """Refresh danh sách cameras"""
        ui_logger.info("🔄 Refreshing camera list")
        self.load_cameras()

    def on_selection_change(self, event):
        """Xử lý khi selection thay đổi"""
        selection = self.tree.selection()
        has_selection = bool(selection)

        self.select_btn.config(state='normal' if has_selection else 'disabled')
        self.test_btn.config(state='normal' if has_selection else 'disabled')

        if has_selection:
            item = self.tree.item(selection[0])
            camera_name = item['values'][1]
            self.update_status(f"Selected: {camera_name}")

    def on_select(self, event=None):
        """Xử lý khi chọn camera"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Vui lòng chọn một camera!")
            return

        item = self.tree.item(selection[0])
        camera_id = item['values'][0]

        # Tìm camera object từ service
        camera = self.camera_service.get_camera_by_id(camera_id)
        if camera:
            self.selected_camera = camera
            self.result = 'ok'
            ui_logger.info(f"✅ Selected camera: {camera.name}")
            self.dialog.destroy()
        else:
            messagebox.showerror("Error", f"Camera with ID {camera_id} not found!")

    def on_cancel(self):
        """Hủy dialog"""
        self.result = 'cancel'
        self.dialog.destroy()
        ui_logger.info("❌ Camera selection canceled")

    def test_camera_connection(self):
        """Test camera connection"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Vui lòng chọn một camera để test!")
            return

        item = self.tree.item(selection[0])
        camera_id = item['values'][0]
        camera = self.camera_service.get_camera_by_id(camera_id)

        if not camera:
            messagebox.showerror("Error", "Camera not found!")
            return

        self.update_status("Testing connection...")
        self.test_btn.config(state='disabled')

        # Run test in thread
        thread = threading.Thread(
            target=self._test_connection_thread,
            args=(camera,),
            daemon=True
        )
        thread.start()

    def _test_connection_thread(self, camera: Camera):
        """Thread function để test connection"""
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

    def copy_rtsp_url(self):
        """Copy RTSP URL to clipboard"""
        selection = self.tree.selection()
        if not selection:
            return

        item = self.tree.item(selection[0])
        camera_id = item['values'][0]
        camera = self.camera_service.get_camera_by_id(camera_id)

        if camera:
            self.dialog.clipboard_clear()
            self.dialog.clipboard_append(camera.rtsp_url)
            self.update_status(f"📋 RTSP URL copied to clipboard")

    def show_camera_info(self):
        """Hiển thị thông tin chi tiết camera"""
        selection = self.tree.selection()
        if not selection:
            return

        item = self.tree.item(selection[0])
        camera_id = item['values'][0]
        camera = self.camera_service.get_camera_by_id(camera_id)

        if camera:
            info = f"""Camera Information:

ID: {camera.id}
Name: {camera.name}
Location: {camera.location}
RTSP URL: {camera.rtsp_url}
HLS URL: {camera.hls_url}
Status: {'Active' if camera.active else 'Inactive'}
Schedule: {camera.current_schedule if camera.current_schedule else 'None'}
"""
            messagebox.showinfo(f"Camera Info - {camera.name}", info)

    def camera_settings(self):
        """Mở settings cho camera"""
        selection = self.tree.selection()
        if not selection:
            return

        item = self.tree.item(selection[0])
        camera_name = item['values'][1]

        # TODO: Implement camera settings dialog
        messagebox.showinfo("Settings", f"Settings for '{camera_name}' will be implemented")

    def update_status(self, message: str):
        """Update status message"""
        self.status_label.config(text=message)

    def get_result(self):
        """Get dialog result"""
        return self.result, self.selected_camera