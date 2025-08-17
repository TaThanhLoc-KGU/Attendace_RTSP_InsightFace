"""
FIXED Camera stream widget - Based on original structure with fixes
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
    """Widget hiển thị camera stream - FIXED based on original structure"""

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

        # CRITICAL FIX: Get and store root window reference properly
        self.root_window = self._get_root_window()

        # Control flags
        self.recognition_enabled = True
        self.auto_attendance_enabled = True

        # Frame processing with FIXED queue management
        self.frame_queue = queue.Queue(maxsize=3)  # Slightly larger queue

        # Initialize UI elements
        self.status_label = None
        self.fps_label = None
        self.recognition_label = None
        self.video_label = None
        self.video_info = None
        self.main_frame = None
        self.recognition_cb = None
        self.auto_attendance_cb = None
        self.start_btn = None
        self.stop_btn = None
        self.snapshot_btn = None

        # Performance tracking
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0

        # Attendance tracking
        self.last_attendance_time = {}
        self.attendance_cooldown = 30

        # Create widgets first
        self.create_widgets()

        ui_logger.info(f"📹 Created FIXED stream widget for camera: {camera.name}")

        # FIXED: Auto start with proper delay and root check
        if self.auto_start:
            self.parent.after(2000, self._safe_auto_start)

    def _get_root_window(self):
        """Get root window and ensure it's properly set - FIXED"""
        try:
            # Get root window from parent
            root = self.parent.winfo_toplevel()

            if root and root.winfo_exists():
                # CRITICAL FIX: Ensure this root is ready for PhotoImage
                # Test PhotoImage creation capability
                try:
                    test_img = Image.new('RGB', (1, 1), color='black')
                    test_photo = ImageTk.PhotoImage(test_img, master=root)
                    del test_photo, test_img  # Clean up

                    ui_logger.info(f"✅ Root window ready for PhotoImage: {root}")
                    return root
                except Exception as e:
                    ui_logger.warning(f"⚠️ Root window not ready for PhotoImage: {e}")
                    return None
            else:
                ui_logger.error("❌ Cannot get valid root window")
                return None

        except Exception as e:
            ui_logger.error(f"❌ Error getting root window: {e}")
            return None

    def _safe_auto_start(self):
        """Safe auto start with root window verification"""
        try:
            # Re-verify root window before starting
            if not self.root_window or not self.root_window.winfo_exists():
                self.root_window = self._get_root_window()

            if self.root_window:
                ui_logger.info("🚀 Auto-starting camera stream...")
                self.start_stream()
            else:
                ui_logger.error("❌ Cannot auto-start: root window not ready")
                self.show_error_display("UI not ready for camera display")
        except Exception as e:
            ui_logger.error(f"❌ Auto-start error: {e}")

    def create_widgets(self):
        """Create UI widgets - ORIGINAL structure with fixes"""
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
        """Create camera info header"""
        info_frame = ttk.Frame(self.main_frame)
        info_frame.pack(fill=tk.X, pady=(0, 10))

        title_frame = ttk.Frame(info_frame)
        title_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Camera icon based on type
        camera_icon = "🎥" if self.camera.is_webcam() else "📹"

        camera_title = ttk.Label(
            title_frame,
            text=f"{camera_icon} {self.camera.name}",
            font=('Arial', 12, 'bold')
        )
        camera_title.pack(side=tk.LEFT)

        location_label = ttk.Label(
            title_frame,
            text=f"📍 {self.camera.location}",
            font=('Arial', 9)
        )
        location_label.pack(side=tk.LEFT, padx=(20, 0))

        # Camera info
        info_text = f"ID: {self.camera.id} | Type: {self.camera.camera_type.upper()}"
        if self.camera.is_webcam():
            info_text += f" | Device: {self.camera.device_index}"

        info_label = ttk.Label(
            title_frame,
            text=info_text,
            font=('Arial', 8),
            foreground='gray'
        )
        info_label.pack(side=tk.LEFT, padx=(20, 0))

    def create_control_panel(self):
        """Create control panel"""
        control_frame = ttk.LabelFrame(self.main_frame, text="🎛️ Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # Left controls
        left_controls = ttk.Frame(control_frame)
        left_controls.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Stream controls
        stream_controls = ttk.Frame(left_controls)
        stream_controls.pack(side=tk.LEFT)

        self.start_btn = ttk.Button(
            stream_controls,
            text="▶️ Start",
            command=self.start_stream,
            width=10
        )
        self.start_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.stop_btn = ttk.Button(
            stream_controls,
            text="⏹️ Stop",
            command=self.stop_stream,
            width=10,
            state='disabled'
        )
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.snapshot_btn = ttk.Button(
            stream_controls,
            text="📸 Snapshot",
            command=self.take_snapshot,
            width=12,
            state='disabled'
        )
        self.snapshot_btn.pack(side=tk.LEFT, padx=(0, 10))

        # Recognition controls
        recognition_frame = ttk.Frame(left_controls)
        recognition_frame.pack(side=tk.LEFT, padx=(20, 0))

        self.recognition_cb = ttk.Checkbutton(
            recognition_frame,
            text="🎯 Face Recognition",
            command=self.toggle_recognition
        )
        self.recognition_cb.pack(side=tk.LEFT)
        self.recognition_cb.state(['selected'])  # Default enabled

        self.auto_attendance_cb = ttk.Checkbutton(
            recognition_frame,
            text="📝 Auto Attendance",
            command=self.toggle_auto_attendance
        )
        self.auto_attendance_cb.pack(side=tk.LEFT, padx=(20, 0))
        self.auto_attendance_cb.state(['selected'])  # Default enabled

        # Right info panel
        right_info = ttk.Frame(control_frame)
        right_info.pack(side=tk.RIGHT)

        # Performance info
        self.fps_label = ttk.Label(
            right_info,
            text="📊 FPS: 0",
            font=('Arial', 9),
            foreground='blue'
        )
        self.fps_label.pack(side=tk.RIGHT, padx=(10, 0))

        self.recognition_label = ttk.Label(
            right_info,
            text="🎯 Recognized: 0",
            font=('Arial', 9),
            foreground='green'
        )
        self.recognition_label.pack(side=tk.RIGHT, padx=(10, 0))

    def create_video_display(self):
        """Create video display area - FIXED"""
        video_frame = ttk.LabelFrame(
            self.main_frame,
            text=f"📺 {self.camera.camera_type.upper()} Video Stream",
            padding=10
        )
        video_frame.pack(fill=tk.BOTH, expand=True)

        # CRITICAL FIX: Create video label with proper initialization
        self.video_label = ttk.Label(
            video_frame,
            text="📷 Camera Preview\n\nClick 'Start' to begin streaming",
            font=('Arial', 14),
            anchor=tk.CENTER,
            background='black',
            foreground='white'
        )
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Video info label
        self.video_info = ttk.Label(
            video_frame,
            text="",
            font=('Arial', 9),
            foreground='gray'
        )
        self.video_info.pack(side=tk.BOTTOM, pady=(5, 0))

    def create_status_bar(self):
        """Create status bar"""
        self.status_label = ttk.Label(
            self.main_frame,
            text="⏳ Ready to start",
            font=('Arial', 9),
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_label.pack(fill=tk.X, pady=(10, 0))

    def start_stream(self):
        """Start camera stream - FIXED VERSION"""
        if self.running:
            return

        try:
            self.update_status("Connecting to camera...")
            ui_logger.info(f"🎬 Starting stream for camera: {self.camera.name}")

            # CRITICAL FIX: Ensure root window is ready
            if not self.root_window or not self.root_window.winfo_exists():
                self.root_window = self._get_root_window()

            if not self.root_window:
                raise Exception("UI not ready for video display")

            # Clear any previous error display
            self.clear_error_display()

            # Create video capture with proper source
            capture_source = self.camera.get_capture_url()
            ui_logger.info(f"📺 Opening camera source: {capture_source}")

            self.cap = cv2.VideoCapture(capture_source)

            # FIXED: Camera configuration
            if not self.cap.isOpened():
                raise Exception(f"Cannot open camera: {capture_source}")

            # Configure camera settings
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if self.camera.is_webcam():
                # Webcam specific settings
                self.cap.set(cv2.CAP_PROP_FPS, config.WEBCAM_FPS_TARGET)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.DISPLAY_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.DISPLAY_HEIGHT)
                # Use MJPEG for better performance
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            else:
                # RTSP settings
                self.cap.set(cv2.CAP_PROP_FPS, 25)

            # Test frame reading
            ret, frame = self.cap.read()
            if not ret:
                raise Exception("Cannot read frame from camera")
            frame = self.face_service.process_frame(frame)

            ui_logger.info(f"📺 Camera resolution: {frame.shape[1]}x{frame.shape[0]}")

            # Start video processing
            self.running = True
            self.update_controls_state(streaming=True)

            # Start video thread
            self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
            self.video_thread.start()

            # Start UI update loop
            self.schedule_ui_update()

            self.update_status("✅ Streaming...")
            ui_logger.info(f"✅ Started streaming camera: {self.camera.name}")

        except Exception as e:
            ui_logger.error(f"❌ Error starting stream: {e}")
            self.show_error_display(f"Stream Error: {str(e)}")
            self.update_status(f"❌ Error: {str(e)}")

            if self.cap:
                self.cap.release()
                self.cap = None

    def video_loop(self):
        """Main video processing loop - FIXED"""
        ui_logger.info(f"🎥 Video loop started for camera: {self.camera.name}")
        recognition_count = 0
        consecutive_failures = 0
        max_failures = 10

        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    consecutive_failures += 1
                    ui_logger.warning(f"Failed to read frame (consecutive: {consecutive_failures})")

                    if consecutive_failures >= max_failures:
                        ui_logger.error("Too many consecutive failures, stopping video loop")
                        break

                    time.sleep(0.1)
                    continue

                # Reset failure counter on success
                consecutive_failures = 0

                # Store current frame
                self.current_frame = frame.copy()

                # Resize for display
                display_frame = cv2.resize(frame, (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT))

                # Face recognition with throttling
                if (self.recognition_enabled and
                    self.frame_count % config.FRAME_PROCESSING_INTERVAL == 0):

                    try:
                        faces = self.face_service.recognize_faces_optimized(display_frame)
                        if faces:
                            recognition_count += len(faces)
                            self.draw_face_annotations(display_frame, faces)

                            if self.should_record_attendance():
                                self.handle_attendance(faces)

                        # Update recognition count in UI thread
                        self.parent.after(0, self._update_recognition_count, recognition_count)

                    except Exception as e:
                        ui_logger.error(f"❌ Face recognition error: {e}")

                # FIXED: Queue frame for display with proper format
                try:
                    frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

                    # Clear old frames if queue is full
                    while self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            break

                    # Queue new frame
                    frame_data = {
                        'frame': frame_rgb,
                        'frame_count': self.frame_count,
                        'recognition_enabled': self.recognition_enabled
                    }
                    self.frame_queue.put_nowait(frame_data)

                except queue.Full:
                    pass  # Skip frame if queue is still full
                except Exception as e:
                    ui_logger.error(f"Error queuing frame: {e}")

                # Update performance metrics
                self.update_fps()
                self.frame_count += 1

                # Frame rate control
                time.sleep(0.033)  # ~30 FPS

            except Exception as e:
                consecutive_failures += 1
                ui_logger.error(f"❌ Error in video loop: {e}")

                if consecutive_failures >= max_failures or not self.running:
                    break

                time.sleep(0.1)

        ui_logger.info(f"🎬 Video loop ended for camera: {self.camera.name}")

    def schedule_ui_update(self):
        """Schedule UI update from main thread - FIXED"""
        if not self.running:
            return

        try:
            # FIXED: Get frame from queue safely
            try:
                if not self.frame_queue.empty():
                    frame_data = self.frame_queue.get_nowait()
                    self.update_video_display_safe(frame_data)
            except queue.Empty:
                pass
            except Exception as e:
                ui_logger.error(f"Error in UI update: {e}")

            # Schedule next update
            if self.running:
                self.parent.after(33, self.schedule_ui_update)  # ~30 FPS UI updates

        except Exception as e:
            ui_logger.error(f"❌ Error in schedule_ui_update: {e}")
            if self.running:
                self.parent.after(100, self.schedule_ui_update)

    def update_video_display_safe(self, frame_data):
        """Update video display safely - ULTIMATE FIX"""
        try:
            if not self.video_label or not self.video_label.winfo_exists():
                return

            frame_rgb = frame_data['frame']
            frame_count = frame_data['frame_count']
            recognition_enabled = frame_data['recognition_enabled']

            # CRITICAL FIX: Ensure root window is available for PhotoImage creation
            if not self.root_window or not self.root_window.winfo_exists():
                ui_logger.warning("Root window not available for PhotoImage")
                return

            # Create PhotoImage with FIXED error handling
            try:
                frame_pil = Image.fromarray(frame_rgb)

                # ULTIMATE FIX: Create PhotoImage with explicit master
                frame_tk = ImageTk.PhotoImage(frame_pil, master=self.root_window)

                # Update video label
                if self.video_label.winfo_exists():
                    self.video_label.config(image=frame_tk, text='')
                    self.video_label.image = frame_tk  # Keep reference

            except Exception as e:
                ui_logger.error(f"❌ PhotoImage creation failed: {e}")
                # Don't show error display for individual frame failures
                return

            # Update video info
            if self.video_info and self.video_info.winfo_exists():
                info_text = f"Frame: {frame_count} | Recognition: {'ON' if recognition_enabled else 'OFF'}"
                self.video_info.config(text=info_text)

        except Exception as e:
            ui_logger.error(f"❌ Error updating video display: {e}")

    def draw_face_annotations(self, frame: np.ndarray, faces: list):
        """Draw face annotations on frame"""

        for face in faces:
            try:
                bbox = face.get('bbox', [0, 0, 100, 100])
                name = face.get('name', 'Unknown')
                similarity = face.get('similarity', 0.0)

                # Draw bounding box
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label
                label = f"{name} ({similarity:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            except Exception as e:
                ui_logger.error(f"Error drawing face annotation: {e}")

    def handle_attendance(self, faces: list):
        """Handle attendance recording"""
        current_time = time.time()

        for face in faces:
            try:
                student_id = face.get('student_id')
                if not student_id:
                    continue

                # Check cooldown
                last_time = self.last_attendance_time.get(student_id, 0)
                if current_time - last_time < self.attendance_cooldown:
                    continue

                # Record attendance
                attendance_data = {
                    'studentId': student_id,
                    'cameraId': self.camera.id,
                    'timestamp': current_time
                }

                # TODO: Send to backend
                self.last_attendance_time[student_id] = current_time
                ui_logger.info(f"📝 Attendance recorded for student {student_id}")

            except Exception as e:
                ui_logger.error(f"Error recording attendance: {e}")

    def stop_stream(self):
        """Stop camera stream"""
        ui_logger.info(f"⏹️ Stopping stream for camera: {self.camera.name}")

        self.running = False

        # Wait for video thread to finish
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=2.0)

        # Release camera
        if self.cap:
            self.cap.release()
            self.cap = None

        # Clear frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        # Update UI
        self.update_controls_state(streaming=False)
        self.update_status("⏹️ Stopped")
        self.clear_video_display()

        ui_logger.info(f"✅ Stopped streaming camera: {self.camera.name}")

    def clear_video_display(self):
        """Clear video display safely"""
        try:
            if self.video_label and self.video_label.winfo_exists():
                self.video_label.config(
                    image='',
                    text="📹 Camera Preview\n\nClick 'Start' to begin streaming",
                    background='black',
                    foreground='white'
                )
                self.video_label.image = None

            if self.video_info and self.video_info.winfo_exists():
                self.video_info.config(text="")

        except Exception as e:
            ui_logger.error(f"❌ Error clearing display: {e}")

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

    def update_fps(self):
        """Update FPS calculation"""
        self.fps_counter += 1
        current_time = time.time()

        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = current_time

            # Update FPS display in UI thread
            self.parent.after(0, self._update_fps_display, self.current_fps)

    def _update_fps_display(self, fps):
        """Update FPS display safely"""
        try:
            if self.fps_label and self.fps_label.winfo_exists():
                self.fps_label.config(text=f"📊 FPS: {fps:.1f}")
        except Exception as e:
            ui_logger.error(f"❌ Error updating FPS: {e}")

    def _update_recognition_count(self, count):
        """Update recognition count safely"""
        try:
            if self.recognition_label and self.recognition_label.winfo_exists():
                self.recognition_label.config(text=f"🎯 Recognized: {count}")
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
                timestamp = time.strftime('%H:%M:%S')
                self.status_label.config(text=f"⏰ {timestamp} | {message}")
            else:
                ui_logger.info(f"📹 {self.camera.name}: {message}")
        except Exception as e:
            ui_logger.error(f"❌ Error updating status: {e}")

    def toggle_recognition(self):
        """Toggle face recognition"""
        try:
            if self.recognition_cb:
                state = self.recognition_cb.instate(['selected'])
                self.recognition_enabled = state
            else:
                self.recognition_enabled = not self.recognition_enabled

            status = "enabled" if self.recognition_enabled else "disabled"
            self.update_status(f"🎯 Face recognition {status}")
            ui_logger.info(f"🎯 Face recognition {status}")

        except Exception as e:
            ui_logger.error(f"❌ Error toggling recognition: {e}")

    def toggle_auto_attendance(self):
        """Toggle auto attendance"""
        try:
            if self.auto_attendance_cb:
                state = self.auto_attendance_cb.instate(['selected'])
                self.auto_attendance_enabled = state
            else:
                self.auto_attendance_enabled = not self.auto_attendance_enabled

            status = "enabled" if self.auto_attendance_enabled else "disabled"
            self.update_status(f"📝 Auto attendance {status}")
            ui_logger.info(f"📝 Auto attendance {status}")

        except Exception as e:
            ui_logger.error(f"❌ Error toggling auto attendance: {e}")

    def should_record_attendance(self) -> bool:
        """Check if should record attendance"""
        return self.auto_attendance_enabled

    def take_snapshot(self):
        """Take snapshot of current frame"""
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
                self.update_status(f"📸 Snapshot saved: {filepath}")
                ui_logger.info(f"📸 Snapshot saved: {filepath}")

        except Exception as e:
            ui_logger.error(f"❌ Error taking snapshot: {e}")
            messagebox.showerror("Error", f"Failed to save snapshot:\n{str(e)}")

    def cleanup(self):
        """Cleanup resources"""
        ui_logger.info(f"🧹 Cleaning up stream widget for camera: {self.camera.name}")
        if self.running:
            self.stop_stream()