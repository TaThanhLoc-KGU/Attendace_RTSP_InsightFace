"""
MINIMAL FIX for stream_widget.py - Based on original file structure
Chỉ sửa những chỗ cần thiết để fix tracking display
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
    """Widget hiển thị camera stream - MINIMAL FIX based on original"""

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

        # Get root window reference
        self.root_window = self._get_root_window()

        # Control flags - FIXED: Don't create BooleanVar here
        self.recognition_enabled = True
        self.auto_attendance_enabled = True

        # Frame processing
        self.frame_queue = queue.Queue(maxsize=3)

        # Initialize UI elements to None first
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

        # Control variables - FIXED: Create after widgets
        self.recognition_var = None
        self.auto_attendance_var = None

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
            self.parent.after(2000, self._safe_auto_start)

    def _get_root_window(self):
        """Get root window safely"""
        try:
            root = self.parent.winfo_toplevel()
            if root and root.winfo_exists():
                # Test PhotoImage capability
                test_img = Image.new('RGB', (1, 1), color='black')
                test_photo = ImageTk.PhotoImage(test_img, master=root)
                test_photo = None  # Clean up
                return root
            return None
        except Exception as e:
            ui_logger.error(f"❌ Root window error: {e}")
            return None

    def create_widgets(self):
        """Create UI widgets - ORIGINAL structure"""
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
            info_text += f" | Device: {self.camera.rtsp_url}"
        else:
            # Mask sensitive URL info
            info_text += f" | RTSP: {self.camera.rtsp_url[:30]}..."

        info_label = ttk.Label(
            info_frame,
            text=info_text,
            font=('Arial', 8),
            foreground='gray'
        )
        info_label.pack(side=tk.RIGHT)

    def create_control_panel(self):
        """Create control panel"""
        control_frame = ttk.LabelFrame(self.main_frame, text="📋 Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # Left side - Stream controls
        left_frame = ttk.Frame(control_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.start_btn = ttk.Button(left_frame, text="▶️ Start Stream", command=self.start_stream)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.stop_btn = ttk.Button(left_frame, text="⏹️ Stop Stream", command=self.stop_stream, state='disabled')
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.snapshot_btn = ttk.Button(left_frame, text="📸 Snapshot", command=self.take_snapshot, state='disabled')
        self.snapshot_btn.pack(side=tk.LEFT, padx=(0, 5))

        # Right side - Recognition controls
        right_frame = ttk.Frame(control_frame)
        right_frame.pack(side=tk.RIGHT)

        # FIXED: Create BooleanVar after parent is ready
        try:
            self.recognition_var = tk.BooleanVar(value=True)
            self.auto_attendance_var = tk.BooleanVar(value=True)
        except Exception as e:
            ui_logger.warning(f"⚠️ Cannot create BooleanVar yet: {e}")
            # Will create later when needed

        # Recognition checkbox
        self.recognition_cb = ttk.Checkbutton(
            right_frame,
            text="🎯 Face Recognition",
            variable=self.recognition_var if self.recognition_var else None,
            command=self.toggle_recognition
        )
        self.recognition_cb.pack(side=tk.LEFT, padx=(0, 10))

        # Auto attendance checkbox
        self.auto_attendance_cb = ttk.Checkbutton(
            right_frame,
            text="📝 Auto Attendance",
            variable=self.auto_attendance_var if self.auto_attendance_var else None,
            command=self.toggle_auto_attendance
        )
        self.auto_attendance_cb.pack(side=tk.LEFT)

        # Status labels
        self.fps_label = ttk.Label(
            control_frame,
            text="📊 FPS: 0.0",
            font=('Arial', 9),
            foreground='blue'
        )
        self.fps_label.pack(side=tk.BOTTOM, anchor=tk.W, pady=(5, 0))

        self.recognition_label = ttk.Label(
            control_frame,
            text="🎯 Recognition: Ready",
            font=('Arial', 9),
            foreground='green'
        )
        self.recognition_label.pack(side=tk.BOTTOM, anchor=tk.E, pady=(5, 0))

    def create_video_display(self):
        """Create video display area"""
        video_frame = ttk.LabelFrame(
            self.main_frame,
            text=f"📺 {self.camera.camera_type.upper()} Video Stream",
            padding=10
        )
        video_frame.pack(fill=tk.BOTH, expand=True)

        # Video label
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

    def _safe_auto_start(self):
        """Safe auto start"""
        try:
            # Ensure BooleanVar are created
            self._ensure_control_vars()

            if not self.running:
                ui_logger.info(f"🚀 Auto-starting camera: {self.camera.name}")
                self.start_stream()
        except Exception as e:
            ui_logger.error(f"❌ Auto-start failed: {e}")

    def _ensure_control_vars(self):
        """Ensure control variables are created"""
        try:
            if not self.recognition_var:
                self.recognition_var = tk.BooleanVar(value=True)
                if self.recognition_cb:
                    self.recognition_cb.config(variable=self.recognition_var)

            if not self.auto_attendance_var:
                self.auto_attendance_var = tk.BooleanVar(value=True)
                if self.auto_attendance_cb:
                    self.auto_attendance_cb.config(variable=self.auto_attendance_var)
        except Exception as e:
            ui_logger.error(f"❌ Error creating control vars: {e}")

    def start_stream(self):
        """Start camera stream"""
        if self.running:
            return

        try:
            self.update_status("Connecting to camera...")
            ui_logger.info(f"🎬 Starting stream for camera: {self.camera.name}")

            # Ensure root window is ready
            if not self.root_window or not self.root_window.winfo_exists():
                self.root_window = self._get_root_window()

            if not self.root_window:
                raise Exception("UI not ready for video display")

            # Clear any previous error display
            self.clear_error_display()

            # Create video capture
            capture_source = self.camera.get_capture_url()
            ui_logger.info(f"📺 Opening camera source: {capture_source}")

            self.cap = cv2.VideoCapture(capture_source)

            if not self.cap.isOpened():
                raise Exception(f"Cannot open camera: {capture_source}")

            # Configure camera settings
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if self.camera.is_webcam():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.DISPLAY_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.DISPLAY_HEIGHT)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            else:
                self.cap.set(cv2.CAP_PROP_FPS, 25)

            # Test frame reading
            ret, frame = self.cap.read()
            if not ret:
                raise Exception("Cannot read frame from camera")

            ui_logger.info(f"📺 Camera resolution: {frame.shape[1]}x{frame.shape[0]}")

            # Start video processing
            self.running = True
            self.update_controls_state(streaming=True)

            # Start video thread - FIXED VERSION
            self.video_thread = threading.Thread(target=self.video_loop_fixed, daemon=True)
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

    def video_loop_fixed(self):
        """FIXED video loop - only use process_frame"""
        ui_logger.info(f"🎥 FIXED video loop started for camera: {self.camera.name}")
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

                # Reset failure counter
                consecutive_failures = 0

                # Store current frame
                self.current_frame = frame.copy()

                # Resize for display
                display_frame = cv2.resize(frame, (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT))

                # 🎯 CRITICAL FIX: Only use process_frame - it does everything!
                if self.recognition_enabled:
                    try:
                        # process_frame already draws bounding boxes and names!
                        processed_frame = self.face_service.process_frame(display_frame)
                        display_frame = processed_frame

                        # Update recognition count (approximate)
                        self.parent.after(0, self._update_recognition_count, self.frame_count // 30)

                    except Exception as e:
                        ui_logger.error(f"❌ Face processing error: {e}")

                # Convert to RGB for display
                try:
                    frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

                    # Clear old frames
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
                    pass  # Skip frame
                except Exception as e:
                    ui_logger.error(f"Error queuing frame: {e}")

                # Update counters
                self.update_fps()
                self.frame_count += 1

                # Frame rate control
                time.sleep(0.033)  # ~30 FPS

            except Exception as e:
                consecutive_failures += 1
                ui_logger.error(f"❌ Video loop error: {e}")

                if consecutive_failures >= max_failures or not self.running:
                    break

                time.sleep(0.1)

        ui_logger.info(f"🎬 Video loop ended for camera: {self.camera.name}")

    def schedule_ui_update(self):
        """Schedule UI updates"""
        if not self.running:
            return

        try:
            # Update video display
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
                self.parent.after(33, self.schedule_ui_update)  # ~30 FPS

        except Exception as e:
            ui_logger.error(f"❌ Error in schedule_ui_update: {e}")
            if self.running:
                self.parent.after(100, self.schedule_ui_update)

    def update_video_display_safe(self, frame_data):
        """Update video display safely"""
        try:
            if not self.video_label or not self.video_label.winfo_exists():
                return

            frame_rgb = frame_data['frame']
            frame_count = frame_data['frame_count']
            recognition_enabled = frame_data['recognition_enabled']

            # Check root window
            if not self.root_window or not self.root_window.winfo_exists():
                ui_logger.warning("Root window not available")
                return

            # Create PhotoImage
            try:
                frame_pil = Image.fromarray(frame_rgb)
                frame_tk = ImageTk.PhotoImage(frame_pil, master=self.root_window)

                # Update video label
                if self.video_label.winfo_exists():
                    self.video_label.config(image=frame_tk, text='')
                    self.video_label.image = frame_tk  # Keep reference

            except Exception as e:
                ui_logger.error(f"❌ PhotoImage error: {e}")
                return

            # Update video info
            if self.video_info and self.video_info.winfo_exists():
                info_text = f"Frame: {frame_count} | Recognition: {'ON' if recognition_enabled else 'OFF'} | FPS: {self.current_fps:.1f}"
                self.video_info.config(text=info_text)

        except Exception as e:
            ui_logger.error(f"❌ Display update error: {e}")

    def stop_stream(self):
        """Stop camera stream"""
        if not self.running:
            return

        ui_logger.info(f"⏹️ Stopping stream: {self.camera.name}")
        self.running = False

        # Wait for thread to finish
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=1.0)

        # Release camera
        if self.cap:
            self.cap.release()
            self.cap = None

        # Update UI
        self.update_controls_state(streaming=False)
        self.show_default_display()
        self.update_status("⏹️ Stopped")

        ui_logger.info(f"✅ Stream stopped: {self.camera.name}")

    def toggle_recognition(self):
        """Toggle face recognition"""
        try:
            self._ensure_control_vars()
            if self.recognition_var:
                self.recognition_enabled = self.recognition_var.get()
            else:
                self.recognition_enabled = not self.recognition_enabled

            status = "enabled" if self.recognition_enabled else "disabled"
            self.update_status(f"Recognition {status}")
            ui_logger.info(f"🎯 Recognition {status} for {self.camera.name}")
        except Exception as e:
            ui_logger.error(f"❌ Toggle recognition error: {e}")

    def toggle_auto_attendance(self):
        """Toggle auto attendance"""
        try:
            self._ensure_control_vars()
            if self.auto_attendance_var:
                self.auto_attendance_enabled = self.auto_attendance_var.get()
            else:
                self.auto_attendance_enabled = not self.auto_attendance_enabled

            status = "enabled" if self.auto_attendance_enabled else "disabled"
            self.update_status(f"Auto attendance {status}")
            ui_logger.info(f"📝 Auto attendance {status} for {self.camera.name}")
        except Exception as e:
            ui_logger.error(f"❌ Toggle attendance error: {e}")

    def take_snapshot(self):
        """Take snapshot"""
        if self.current_frame is not None:
            try:
                import os
                from datetime import datetime

                # Create snapshots directory
                snapshot_dir = "data/snapshots"
                os.makedirs(snapshot_dir, exist_ok=True)

                # Generate filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.camera.name}_{timestamp}.jpg"
                filepath = os.path.join(snapshot_dir, filename)

                # Save image
                cv2.imwrite(filepath, self.current_frame)

                self.update_status(f"📸 Snapshot saved: {filename}")
                ui_logger.info(f"📸 Snapshot saved: {filepath}")

                messagebox.showinfo("Snapshot", f"Snapshot saved successfully!\n\n{filepath}")

            except Exception as e:
                error_msg = f"Failed to save snapshot: {e}"
                self.update_status(f"❌ {error_msg}")
                messagebox.showerror("Error", error_msg)
        else:
            messagebox.showwarning("Warning", "No frame available for snapshot")

    def update_controls_state(self, streaming: bool):
        """Update control button states"""
        try:
            if streaming:
                if self.start_btn:
                    self.start_btn.config(state='disabled')
                if self.stop_btn:
                    self.stop_btn.config(state='normal')
                if self.snapshot_btn:
                    self.snapshot_btn.config(state='normal')
            else:
                if self.start_btn:
                    self.start_btn.config(state='normal')
                if self.stop_btn:
                    self.stop_btn.config(state='disabled')
                if self.snapshot_btn:
                    self.snapshot_btn.config(state='disabled')
        except Exception as e:
            ui_logger.error(f"❌ Control state error: {e}")

    def show_default_display(self):
        """Show default display"""
        try:
            if self.video_label and self.video_label.winfo_exists():
                self.video_label.config(
                    image='',
                    text="📹 Camera Ready\n\nClick 'Start' to begin streaming",
                    background='black',
                    foreground='white'
                )
                self.video_label.image = None
        except Exception as e:
            ui_logger.error(f"❌ Default display error: {e}")

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

    def update_status(self, status: str):
        """Update status label"""
        try:
            if self.status_label and self.status_label.winfo_exists():
                self.status_label.config(text=status)
        except Exception as e:
            ui_logger.error(f"❌ Status update error: {e}")

    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()

        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = current_time

            # Update FPS label in UI thread
            if self.fps_label and self.fps_label.winfo_exists():
                self.parent.after(0, self._update_fps_display)

    def _update_fps_display(self):
        """Update FPS display safely"""
        try:
            if self.fps_label and self.fps_label.winfo_exists():
                self.fps_label.config(text=f"📊 FPS: {self.current_fps:.1f}")
        except Exception as e:
            ui_logger.error(f"❌ FPS update error: {e}")

    def _update_recognition_count(self, count):
        """Update recognition count safely"""
        try:
            if self.recognition_label and self.recognition_label.winfo_exists():
                self.recognition_label.config(text=f"🎯 Recognition: {count} frames processed")
        except Exception as e:
            ui_logger.error(f"❌ Recognition update error: {e}")

    def cleanup(self):
        """Cleanup resources"""
        ui_logger.info(f"🧹 Cleaning up stream widget: {self.camera.name}")

        try:
            self.stop_stream()

            # Clear frame queue
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break

        except Exception as e:
            ui_logger.error(f"❌ Cleanup error: {e}")