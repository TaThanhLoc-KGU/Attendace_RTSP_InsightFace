"""
Camera stream widget for displaying video with face recognition
"""
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import threading
import time
import numpy as np
from PIL import Image, ImageTk
from typing import Optional

from services.camera_service import Camera
from services.face_service import FaceRecognitionService
from config.config import config
from utils.logger import ui_logger


class CameraStreamWidget:
    """Widget hiển thị camera stream với face recognition"""

    def __init__(self, parent, camera: Camera, face_service: FaceRecognitionService):
        self.parent = parent
        self.camera = camera
        self.face_service = face_service
        self.cap = None
        self.running = False
        self.video_thread = None
        self.recognition_enabled = True
        self.current_frame = None
        self.frame_count = 0

        # Performance tracking
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0

        self.create_widgets()
        ui_logger.info(f"📹 Created stream widget for camera: {camera.name}")

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

        # Camera name và location
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

        # Camera info
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

        # Right controls
        right_controls = ttk.Frame(control_frame)
        right_controls.pack(side=tk.RIGHT)

        # Face recognition toggle
        self.recognition_var = tk.BooleanVar(value=True)
        self.recognition_cb = ttk.Checkbutton(
            right_controls,
            text="🔍 Face Recognition",
            variable=self.recognition_var,
            command=self.toggle_recognition
        )
        self.recognition_cb.pack(side=tk.RIGHT, padx=5)

        # Settings button
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

        # Video label với border
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

        # Video info overlay
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

        # FPS display
        self.fps_label = ttk.Label(
            status_frame,
            text="FPS: 0",
            font=('Arial', 9),
            relief=tk.SUNKEN,
            anchor=tk.E,
            width=10
        )
        self.fps_label.pack(side=tk.RIGHT, padx=(5, 0))

    def start_stream(self):
        """Bắt đầu stream camera"""
        if self.running:
            return

        try:
            self.update_status("Connecting to camera...")
            ui_logger.info(f"🎬 Starting stream for camera: {self.camera.name}")

            # Create video capture
            self.cap = cv2.VideoCapture(self.camera.rtsp_url)

            # Set capture properties
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

            # Start video thread
            self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
            self.video_thread.start()

            self.update_status("Streaming...")
            ui_logger.info(f"✅ Started streaming camera: {self.camera.name}")

        except Exception as e:
            ui_logger.error(f"❌ Error starting stream: {e}")
            self.update_status(f"Error: {str(e)}")
            messagebox.showerror("Stream Error", f"Cannot start camera stream:\n{str(e)}")

            if self.cap:
                self.cap.release()
                self.cap = None

    def stop_stream(self):
        """Dừng stream camera"""
        self.running = False

        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=2)

        if self.cap:
            self.cap.release()
            self.cap = None

        self.update_controls_state(streaming=False)
        self.update_status("Stopped")

        # Clear video display
        self.video_label.config(
            image='',
            text="📹 Camera Preview\n\nClick 'Start Stream' to begin"
        )
        self.video_info.config(text="")
        self.fps_label.config(text="FPS: 0")

        ui_logger.info(f"⏹️ Stopped streaming camera: {self.camera.name}")

    def take_snapshot(self):
        """Chụp ảnh snapshot"""
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

    def toggle_recognition(self):
        """Bật/tắt face recognition"""
        self.recognition_enabled = self.recognition_var.get()
        status = "enabled" if self.recognition_enabled else "disabled"
        self.update_status(f"Face recognition {status}")
        ui_logger.info(f"🔍 Face recognition {status} for camera: {self.camera.name}")

    def open_settings(self):
        """Mở settings cho camera"""
        # TODO: Implement camera settings dialog
        messagebox.showinfo("Settings", f"Settings for camera '{self.camera.name}' will be implemented")

    def video_loop(self):
        """Main video processing loop"""
        ui_logger.info(f"🎥 Video loop started for camera: {self.camera.name}")

        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    ui_logger.warning("Failed to read frame")
                    break

                self.current_frame = frame.copy()

                # Resize frame for display
                display_frame = cv2.resize(frame, (640, 480))

                # Face recognition every N frames
                if (self.recognition_enabled and
                        self.frame_count % config.FRAME_PROCESSING_INTERVAL == 0):
                    faces = self.face_service.recognize_faces(display_frame)
                    self.draw_face_annotations(display_frame, faces)

                # Convert cho Tkinter display
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_tk = ImageTk.PhotoImage(frame_pil)

                # Update UI (thread-safe)
                self.parent.after(0, self.update_video_display, frame_tk)

                # Update FPS
                self.update_fps()

                self.frame_count += 1
                time.sleep(0.033)  # ~30 FPS

            except Exception as e:
                ui_logger.error(f"❌ Error in video loop: {e}")
                break

        ui_logger.info(f"🎬 Video loop ended for camera: {self.camera.name}")
        self.parent.after(0, self.stop_stream)

    def draw_face_annotations(self, frame: np.ndarray, faces: list):
        """Vẽ annotations cho detected faces"""
        for face in faces:
            bbox = face['bbox']
            name = face['name']
            confidence = face['confidence']
            is_known = face['is_known']

            # Màu sắc theo trạng thái
            color = (0, 255, 0) if is_known else (0, 0, 255)  # Green for known, Red for unknown

            # Draw bounding box
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            # Draw label background
            label = f"{name} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

            cv2.rectangle(frame,
                          (bbox[0], bbox[1] - label_size[1] - 10),
                          (bbox[0] + label_size[0], bbox[1]),
                          color, -1)

            # Draw label text
            cv2.putText(frame, label, (bbox[0], bbox[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Draw additional info
            if 'age' in face and face['age'] > 0:
                age_gender = f"Age: {face['age']}, {'M' if face['gender'] == 0 else 'F'}"
                cv2.putText(frame, age_gender, (bbox[0], bbox[3] + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def update_video_display(self, frame_tk):
        """Update video display (thread-safe)"""
        try:
            self.video_label.config(image=frame_tk, text='')
            self.video_label.image = frame_tk  # Keep reference

            # Update video info
            info_text = f"Frame: {self.frame_count} | Recognition: {'ON' if self.recognition_enabled else 'OFF'}"
            self.video_info.config(text=info_text)

        except Exception as e:
            ui_logger.error(f"❌ Error updating video display: {e}")

    def update_fps(self):
        """Update FPS calculation"""
        self.fps_counter += 1
        current_time = time.time()

        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = current_time

            # Update FPS label (thread-safe)
            self.parent.after(0, lambda: self.fps_label.config(text=f"FPS: {self.current_fps}"))

    def update_controls_state(self, streaming: bool):
        """Update control buttons state"""
        if streaming:
            self.start_btn.config(state='disabled')
            self.stop_btn.config(state='normal')
            self.snapshot_btn.config(state='normal')
        else:
            self.start_btn.config(state='normal')
            self.stop_btn.config(state='disabled')
            self.snapshot_btn.config(state='disabled')

    def update_status(self, message: str):
        """Update status message"""
        self.status_label.config(text=f"Status: {message}")

    def cleanup(self):
        """Cleanup resources"""
        if self.running:
            self.stop_stream()
        ui_logger.info(f"🧹 Cleaned up stream widget for camera: {self.camera.name}")