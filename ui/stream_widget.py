"""
ULTIMATE FIX - Camera stream widget - Force set default root window
"""
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import threading
import time
import numpy as np
from PIL import Image, ImageTk
from typing import Optional
import queue

from services.camera_service import Camera
from services.face_service import FaceRecognitionService
from config.config import config
from utils.logger import ui_logger

class CameraStreamWidget:
    """Widget hiển thị camera stream - ULTIMATE FIX for root window issues"""

    def __init__(self, parent, camera: Camera, face_service: FaceRecognitionService):
        self.parent = parent
        self.camera = camera
        self.face_service = face_service
        self.cap = None
        self.running = False
        self.video_thread = None
        self.current_frame = None
        self.frame_count = 0
        self.auto_start = True

        # CRITICAL FIX: Get and store root window reference
        self.root_window = self._get_root_window()

        # Use simple boolean flags instead of Tkinter variables
        self.recognition_enabled = True
        self.auto_attendance_enabled = True

        # Frame processing
        self.frame_queue = queue.Queue(maxsize=2)

        # Initialize UI elements
        self.status_label = None
        self.fps_label = None
        self.recognition_label = None
        self.video_label = None
        self.video_info = None
        self.main_frame = None
        self.recognition_cb = None
        self.auto_attendance_cb = None

        # Performance tracking
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0

        # Attendance tracking
        self.last_attendance_time = {}
        self.attendance_cooldown = 30

        # Create widgets
        self.create_widgets()

        ui_logger.info(f"📹 Created stream widget for camera: {camera.name}")

        # Auto start with delay
        if self.auto_start:
            self.parent.after(3000, self.start_stream)

    def _get_root_window(self):
        """Get root window and ensure it's set as default"""
        try:
            # Get root window from parent
            root = self.parent.winfo_toplevel()

            # CRITICAL FIX: Force set as default root window
            if root and root.winfo_exists():
                # This is the key fix - make sure this root is the default
                root.tk.call('wm', 'withdraw', '.')  # Hide default root if exists
                root.tk.call('wm', 'deiconify', root._w)  # Show our root

                # Set this as the default root for PhotoImage
                if not hasattr(tk, '_default_root') or tk._default_root is None:
                    tk._default_root = root

                ui_logger.info(f"✅ Root window set as default: {root}")
                return root
            else:
                ui_logger.error("❌ Cannot get valid root window")
                return None

        except Exception as e:
            ui_logger.error(f"❌ Error getting root window: {e}")
            return None

    def _ensure_root_is_default(self):
        """Ensure our root window is set as default before creating images"""
        try:
            if self.root_window and self.root_window.winfo_exists():
                # Force set as default root
                tk._default_root = self.root_window
                return True
            else:
                # Try to get root again
                self.root_window = self._get_root_window()
                return self.root_window is not None
        except Exception as e:
            ui_logger.error(f"❌ Error ensuring root is default: {e}")
            return False

    def create_widgets(self):
        """Tạo UI widgets"""
        # Main frame
        self.main_frame = ttk.Frame(self.parent)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Camera info header
        self.create_info_header()

        # Control panel
        self.create_control_panel()

        # Video display area
        self.create_video_display()

        # Status bar
        self.create_status_bar()

    def create_info_header(self):
        """Tạo header thông tin camera"""
        info_frame = ttk.Frame(self.main_frame)
        info_frame.pack(fill=tk.X, pady=(0, 10))

        title_frame = ttk.Frame(info_frame)
        title_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        camera_title = ttk.Label(
            title_frame,
            text=f"📹 {self.camera.name}",
            font=('Arial', 12, 'bold')
        )
        camera_title.pack(side=tk.LEFT)

        location_label = ttk.Label(
            title_frame,
            text=f"📍 {self.camera.location}",
            font=('Arial', 9)
        )
        location_label.pack(side=tk.LEFT, padx=(20, 0))

        info_text = f"ID: {self.camera.id} | RTSP: {self.camera.rtsp_url[:30]}..."
        info_label = ttk.Label(
            title_frame,
            text=info_text,
            font=('Arial', 8),
            foreground='gray'
        )
        info_label.pack(side=tk.LEFT, padx=(20, 0))

    def create_control_panel(self):
        """Tạo control panel"""
        control_frame = ttk.Frame(self.main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # Left controls
        left_controls = ttk.Frame(control_frame)
        left_controls.pack(side=tk.LEFT)

        self.start_btn = ttk.Button(
            left_controls,
            text="▶️ Start Stream",
            command=self.start_stream,
            width=15
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(
            left_controls,
            text="⏹️ Stop Stream",
            command=self.stop_stream,
            width=15,
            state='disabled'
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.snapshot_btn = ttk.Button(
            left_controls,
            text="📸 Snapshot",
            command=self.take_snapshot,
            width=12,
            state='disabled'
        )
        self.snapshot_btn.pack(side=tk.LEFT, padx=5)

        self.reload_btn = ttk.Button(
            left_controls,
            text="🔄 Reload Students",
            command=self.reload_student_embeddings,
            width=15
        )
        self.reload_btn.pack(side=tk.LEFT, padx=5)

        # Right controls
        right_controls = ttk.Frame(control_frame)
        right_controls.pack(side=tk.RIGHT)

        # Create checkboxes with manual state management
        self.recognition_cb = ttk.Checkbutton(
            right_controls,
            text="🔍 Face Recognition",
            command=self.toggle_recognition
        )
        self.recognition_cb.pack(side=tk.RIGHT, padx=5)
        if self.recognition_enabled:
            self.recognition_cb.state(['selected'])

        self.auto_attendance_cb = ttk.Checkbutton(
            right_controls,
            text="📝 Auto Attendance",
            command=self.toggle_auto_attendance
        )
        self.auto_attendance_cb.pack(side=tk.RIGHT, padx=5)
        if self.auto_attendance_enabled:
            self.auto_attendance_cb.state(['selected'])

        settings_btn = ttk.Button(
            right_controls,
            text="⚙️ Settings",
            command=self.open_settings,
            width=12
        )
        settings_btn.pack(side=tk.RIGHT, padx=5)

    def create_video_display(self):
        """Tạo video display area"""
        video_frame = ttk.Frame(self.main_frame)
        video_frame.pack(fill=tk.BOTH, expand=True)

        self.video_label = ttk.Label(
            video_frame,
            text="📹 Camera Preview\n\nClick 'Start Stream' to begin",
            font=('Arial', 14),
            background='black',
            foreground='white',
            anchor='center',
            justify='center'
        )
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.video_info = ttk.Label(
            video_frame,
            text="",
            font=('Arial', 8),
            background='black',
            foreground='lightgreen'
        )
        self.video_info.place(x=10, y=10)

    def create_status_bar(self):
        """Tạo status bar"""
        status_frame = ttk.Frame(self.main_frame)
        status_frame.pack(fill=tk.X, pady=(5, 0))

        self.status_label = ttk.Label(
            status_frame,
            text="Status: Ready",
            font=('Arial', 9),
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.fps_label = ttk.Label(
            status_frame,
            text="FPS: 0",
            font=('Arial', 9),
            relief=tk.SUNKEN,
            anchor=tk.E,
            width=10
        )
        self.fps_label.pack(side=tk.RIGHT, padx=(5, 0))

        self.recognition_label = ttk.Label(
            status_frame,
            text="Recognized: 0",
            font=('Arial', 9),
            relief=tk.SUNKEN,
            anchor=tk.E,
            width=15
        )
        self.recognition_label.pack(side=tk.RIGHT, padx=(5, 0))

    def toggle_recognition(self):
        """Toggle face recognition"""
        try:
            if self.recognition_cb:
                state = self.recognition_cb.instate(['selected'])
                self.recognition_enabled = state
            else:
                self.recognition_enabled = not self.recognition_enabled

            status = "enabled" if self.recognition_enabled else "disabled"
            self.update_status(f"Face recognition {status}")
            ui_logger.info(f"🔍 Face recognition {status}")

        except Exception as e:
            ui_logger.error(f"❌ Error toggling recognition: {e}")
            self.recognition_enabled = not self.recognition_enabled

    def toggle_auto_attendance(self):
        """Toggle auto attendance"""
        try:
            if self.auto_attendance_cb:
                state = self.auto_attendance_cb.instate(['selected'])
                self.auto_attendance_enabled = state
            else:
                self.auto_attendance_enabled = not self.auto_attendance_enabled

            status = "enabled" if self.auto_attendance_enabled else "disabled"
            self.update_status(f"Auto attendance {status}")
            ui_logger.info(f"📝 Auto attendance {status}")

        except Exception as e:
            ui_logger.error(f"❌ Error toggling auto attendance: {e}")
            self.auto_attendance_enabled = not self.auto_attendance_enabled

    def should_record_attendance(self) -> bool:
        """Check if should record attendance"""
        return self.auto_attendance_enabled

    def start_stream(self):
        """Bắt đầu stream camera - ULTIMATE FIX"""
        if self.running:
            return

        try:
            self.update_status("Connecting to camera...")
            ui_logger.info(f"🎬 Starting stream for camera: {self.camera.name}")

            # CRITICAL FIX: Ensure root window is ready and set as default
            if not self._ensure_root_is_default():
                ui_logger.error("❌ Root window not ready")
                self.show_error_display("Root window not ready")
                return

            # Test PhotoImage creation BEFORE starting stream
            if not self._test_photoimage_creation():
                ui_logger.error("❌ PhotoImage creation test failed")
                self.show_error_display("PhotoImage creation failed")
                return

            # Create video capture
            self.cap = cv2.VideoCapture(self.camera.rtsp_url)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            if not self.cap.isOpened():
                raise Exception(f"Cannot open camera stream: {self.camera.rtsp_url}")

            # Test frame reading
            ret, frame = self.cap.read()
            if not ret:
                raise Exception("Cannot read frame from camera")

            ui_logger.info(f"📺 Camera resolution: {frame.shape[1]}x{frame.shape[0]}")

            self.running = True
            self.update_controls_state(streaming=True)
            self.clear_error_display()

            # Start video thread
            self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
            self.video_thread.start()

            # Start UI update loop
            self.schedule_ui_update()

            self.update_status("Streaming...")
            ui_logger.info(f"✅ Started streaming camera: {self.camera.name}")

        except Exception as e:
            ui_logger.error(f"❌ Error starting stream: {e}")
            self.show_error_display(f"Stream Error: {str(e)}")
            self.update_status(f"Error: {str(e)}")

            if self.cap:
                self.cap.release()
                self.cap = None

    def _test_photoimage_creation(self):
        """Test PhotoImage creation with proper root window setup"""
        max_attempts = 5

        for attempt in range(max_attempts):
            try:
                # Ensure root is default
                if not self._ensure_root_is_default():
                    time.sleep(0.2)
                    continue

                # Create test image
                test_image = Image.new('RGB', (10, 10), color='black')

                # CRITICAL FIX: Create PhotoImage with explicit root parameter
                if self.root_window and self.root_window.winfo_exists():
                    test_photo = ImageTk.PhotoImage(test_image, master=self.root_window)
                else:
                    test_photo = ImageTk.PhotoImage(test_image)

                # Clean up
                del test_photo
                del test_image

                ui_logger.info(f"✅ PhotoImage test successful on attempt {attempt + 1}")
                return True

            except Exception as e:
                ui_logger.warning(f"PhotoImage test failed (attempt {attempt + 1}): {e}")
                if attempt < max_attempts - 1:
                    time.sleep(0.5)  # Wait longer between attempts
                    continue
                else:
                    ui_logger.error(f"❌ PhotoImage test failed after {max_attempts} attempts")
                    return False

        return False

    def clear_error_display(self):
        """Clear error display"""
        try:
            if self.video_label and self.video_label.winfo_exists():
                self.video_label.config(
                    text="📹 Initializing camera...",
                    background='black',
                    foreground='white'
                )
        except Exception as e:
            ui_logger.error(f"Error clearing error display: {e}")

    def show_error_display(self, error_message: str):
        """Show error in video display area"""
        try:
            if self.video_label and self.video_label.winfo_exists():
                self.video_label.config(
                    image='',
                    text=f"❌ {error_message}",
                    background='darkred',
                    foreground='white'
                )
                self.video_label.image = None
        except Exception as e:
            ui_logger.error(f"Error showing error display: {e}")

    def stop_stream(self):
        """Dừng stream camera"""
        self.running = False

        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=2)

        if self.cap:
            self.cap.release()
            self.cap = None

        # Clear frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        self.update_controls_state(streaming=False)
        self.update_status("Stopped")
        self.clear_video_display()

        ui_logger.info(f"⏹️ Stopped streaming camera: {self.camera.name}")

    def clear_video_display(self):
        """Clear video display safely"""
        try:
            if self.video_label and self.video_label.winfo_exists():
                self.video_label.config(
                    image='',
                    text="📹 Camera Preview\n\nClick 'Start Stream' to begin",
                    background='black',
                    foreground='white'
                )
                self.video_label.image = None

            if self.video_info and self.video_info.winfo_exists():
                self.video_info.config(text="")

            if self.fps_label and self.fps_label.winfo_exists():
                self.fps_label.config(text="FPS: 0")

            if self.recognition_label and self.recognition_label.winfo_exists():
                self.recognition_label.config(text="Recognized: 0")

        except Exception as e:
            ui_logger.error(f"❌ Error clearing display: {e}")

    def video_loop(self):
        """Main video processing loop"""
        ui_logger.info(f"🎥 Video loop started for camera: {self.camera.name}")
        recognition_count = 0

        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    ui_logger.warning("Failed to read frame")
                    break

                self.current_frame = frame.copy()
                display_frame = cv2.resize(frame, (640, 480))

                # Face recognition
                if (self.recognition_enabled and
                    self.frame_count % config.FRAME_PROCESSING_INTERVAL == 0):

                    try:
                        faces = self.face_service.recognize_faces(display_frame)
                        if faces:
                            recognition_count += len(faces)
                            self.draw_face_annotations(display_frame, faces)

                            if self.should_record_attendance():
                                self.handle_attendance(faces)

                        self.parent.after(0, self._update_recognition_count, recognition_count)

                    except Exception as e:
                        ui_logger.error(f"❌ Face recognition error: {e}")

                # Queue frame for display
                try:
                    frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

                    if not self.frame_queue.full():
                        self.frame_queue.put_nowait({
                            'frame': frame_rgb,
                            'frame_count': self.frame_count,
                            'recognition_enabled': self.recognition_enabled
                        })
                except queue.Full:
                    pass
                except Exception as e:
                    ui_logger.error(f"Error queuing frame: {e}")

                self.update_fps()
                self.frame_count += 1
                time.sleep(0.033)

            except Exception as e:
                ui_logger.error(f"❌ Error in video loop: {e}")
                time.sleep(0.1)
                continue

        ui_logger.info(f"🎬 Video loop ended for camera: {self.camera.name}")

    def schedule_ui_update(self):
        """Schedule UI update from main thread"""
        if not self.running:
            return

        try:
            if not self.frame_queue.empty():
                frame_data = self.frame_queue.get_nowait()
                self.update_video_display_safe(frame_data)

        except queue.Empty:
            pass
        except Exception as e:
            ui_logger.error(f"Error in UI update: {e}")

        if self.running:
            self.parent.after(33, self.schedule_ui_update)

    def update_video_display_safe(self, frame_data):
        """Update video display safely - ULTIMATE FIX"""
        try:
            if not self.video_label or not self.video_label.winfo_exists():
                return

            frame_rgb = frame_data['frame']
            frame_count = frame_data['frame_count']
            recognition_enabled = frame_data['recognition_enabled']

            # CRITICAL FIX: Ensure root is default before creating PhotoImage
            if not self._ensure_root_is_default():
                ui_logger.warning("Root window not available for PhotoImage")
                return

            # Create PhotoImage with explicit master parameter
            try:
                frame_pil = Image.fromarray(frame_rgb)

                # ULTIMATE FIX: Create PhotoImage with explicit master
                if self.root_window and self.root_window.winfo_exists():
                    frame_tk = ImageTk.PhotoImage(frame_pil, master=self.root_window)
                else:
                    frame_tk = ImageTk.PhotoImage(frame_pil)

                # Update video label
                self.video_label.config(image=frame_tk, text='')
                self.video_label.image = frame_tk

            except Exception as e:
                ui_logger.error(f"❌ PhotoImage creation failed: {e}")
                self.show_error_display(f"Display Error: {str(e)}")
                return

            # Update video info
            if self.video_info and self.video_info.winfo_exists():
                info_text = f"Frame: {frame_count} | Recognition: {'ON' if recognition_enabled else 'OFF'}"
                self.video_info.config(text=info_text)

        except Exception as e:
            ui_logger.error(f"❌ Error updating video display: {e}")
            self.show_error_display(f"Display Error: {str(e)}")

    def draw_face_annotations(self, frame: np.ndarray, faces: list):
        """Draw face annotations"""
        for face in faces:
            try:
                bbox = face.get('bbox', [0, 0, 0, 0])
                if not bbox or len(bbox) < 4:
                    continue

                h, w = frame.shape[:2]
                bbox = [
                    max(0, min(int(bbox[0]), w)),
                    max(0, min(int(bbox[1]), h)),
                    max(0, min(int(bbox[2]), w)),
                    max(0, min(int(bbox[3]), h))
                ]

                if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                    continue

                name = face.get('name', 'Unknown')
                confidence = face.get('confidence', 0.0)
                is_known = face.get('is_known', False)

                color = (0, 255, 0) if is_known else (0, 0, 255)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

                label = f"{name} - {confidence:.2f}"
                cv2.putText(frame, label, (bbox[0], bbox[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            except Exception as e:
                ui_logger.error(f"❌ Error drawing annotation: {e}")
                continue

    def handle_attendance(self, faces):
        """Handle attendance recording"""
        current_time = time.time()

        for face in faces:
            try:
                if not face.get('is_known') or not face.get('student_id'):
                    continue

                student_id = face['student_id']

                if (student_id in self.last_attendance_time and
                    current_time - self.last_attendance_time[student_id] < self.attendance_cooldown):
                    continue

                result = self.face_service.record_attendance(student_id, self.camera.id)

                if result['success']:
                    self.last_attendance_time[student_id] = current_time
                    ui_logger.info(f"✅ Attendance recorded for {student_id}")
                    self.parent.after(0, lambda: self.update_status(f"✅ Attendance: {face['name']}"))

            except Exception as e:
                ui_logger.error(f"❌ Error handling attendance: {e}")
                continue

    def update_fps(self):
        """Update FPS calculation"""
        self.fps_counter += 1
        current_time = time.time()

        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = current_time
            self.parent.after(0, self._update_fps_display, self.current_fps)

    def _update_fps_display(self, fps):
        """Update FPS display safely"""
        try:
            if self.fps_label and self.fps_label.winfo_exists():
                self.fps_label.config(text=f"FPS: {fps}")
        except Exception as e:
            ui_logger.error(f"❌ Error updating FPS: {e}")

    def _update_recognition_count(self, count):
        """Update recognition count safely"""
        try:
            if self.recognition_label and self.recognition_label.winfo_exists():
                self.recognition_label.config(text=f"Recognized: {count}")
        except Exception as e:
            ui_logger.error(f"❌ Error updating recognition count: {e}")

    def update_controls_state(self, streaming: bool):
        """Update control buttons state"""
        try:
            if streaming:
                if self.start_btn and self.start_btn.winfo_exists():
                    self.start_btn.config(state='disabled')
                if self.stop_btn and self.stop_btn.winfo_exists():
                    self.stop_btn.config(state='normal')
                if self.snapshot_btn and self.snapshot_btn.winfo_exists():
                    self.snapshot_btn.config(state='normal')
            else:
                if self.start_btn and self.start_btn.winfo_exists():
                    self.start_btn.config(state='normal')
                if self.stop_btn and self.stop_btn.winfo_exists():
                    self.stop_btn.config(state='disabled')
                if self.snapshot_btn and self.snapshot_btn.winfo_exists():
                    self.snapshot_btn.config(state='disabled')
        except Exception as e:
            ui_logger.error(f"❌ Error updating controls: {e}")

    def update_status(self, message: str):
        """Update status message safely"""
        try:
            if self.status_label and self.status_label.winfo_exists():
                self.status_label.config(text=f"Status: {message}")
            else:
                ui_logger.info(f"📹 {self.camera.name}: {message}")
        except Exception as e:
            ui_logger.error(f"❌ Error updating status: {e}")

    def take_snapshot(self):
        """Take snapshot"""
        if not self.running or self.current_frame is None:
            messagebox.showwarning("Warning", "No active stream to capture!")
            return

        try:
            from tkinter import filedialog
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_{self.camera.name}_{timestamp}.jpg"

            filepath = filedialog.asksaveasfilename(
                defaultextension=".jpg",
                filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")],
                initialname=filename
            )

            if filepath:
                cv2.imwrite(filepath, self.current_frame)
                self.update_status(f"Snapshot saved: {filepath}")
                ui_logger.info(f"📸 Snapshot saved: {filepath}")

        except Exception as e:
            ui_logger.error(f"❌ Error taking snapshot: {e}")
            messagebox.showerror("Error", f"Failed to save snapshot:\n{str(e)}")

    def reload_student_embeddings(self):
        """Reload student embeddings"""
        self.update_status("Reloading student embeddings...")
        thread = threading.Thread(target=self._reload_embeddings_thread, daemon=True)
        thread.start()

    def _reload_embeddings_thread(self):
        """Thread function to reload embeddings"""
        try:
            result = self.face_service.load_student_embeddings()
            self.parent.after(0, self._on_embeddings_reloaded, result)
        except Exception as e:
            self.parent.after(0, self._on_embeddings_reload_error, str(e))

    def _on_embeddings_reloaded(self, result):
        """Handle embeddings reloaded"""
        if result['success']:
            self.update_status(f"✅ Reloaded {result['count']} student embeddings")
            messagebox.showinfo("Success", f"✅ Reloaded {result['count']} student embeddings")
        else:
            self.update_status(f"❌ Failed to reload embeddings: {result['message']}")
            messagebox.showerror("Error", f"Failed to reload embeddings:\n{result['message']}")

    def _on_embeddings_reload_error(self, error):
        """Handle reload error"""
        self.update_status(f"❌ Reload error: {error}")
        messagebox.showerror("Error", f"Error reloading embeddings:\n{error}")

    def open_settings(self):
        """Open settings"""
        messagebox.showinfo("Settings", f"Settings for camera '{self.camera.name}' will be implemented")

    def cleanup(self):
        """Cleanup resources"""
        if self.running:
            self.stop_stream()
        ui_logger.info(f"🧹 Cleaned up stream widget for camera: {self.camera.name}")
