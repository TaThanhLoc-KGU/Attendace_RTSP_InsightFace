"""
Advanced Camera Stream Widget với full InsightFace integration
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
from concurrent.futures import ThreadPoolExecutor

from services.camera_service import Camera
from services.advanced_face_service import AdvancedFaceService
from config.config import config
from utils.logger import ui_logger


class AdvancedCameraStreamWidget:
    """Advanced Camera Stream Widget với full InsightFace features"""

    def __init__(self, parent, camera: Camera, face_service: AdvancedFaceService):
        self.parent = parent
        self.camera = camera
        self.face_service = face_service
        self.cap = None
        self.running = False
        self.video_thread = None
        self.current_frame = None
        self.frame_count = 0
        self.auto_start = True

        # Performance settings
        self.target_fps = 30
        self.display_scale = 0.8

        # Root window management
        self.root_window = self._get_root_window()

        # State management
        self.recognition_enabled = True
        self.tracking_enabled = True
        self.show_age_gender = True
        self.show_quality_metrics = True

        # Frame processing
        self.frame_queue = queue.Queue(maxsize=3)

        # UI elements
        self.status_label = None
        self.fps_label = None
        self.tracks_label = None
        self.recognition_label = None
        self.video_label = None
        self.video_info = None
        self.main_frame = None

        # Controls
        self.recognition_cb = None
        self.tracking_cb = None
        self.age_gender_cb = None
        self.quality_cb = None
        self.threshold_var = None

        # Performance tracking
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0

        # Create widgets
        self.create_widgets()

        ui_logger.info(f"🚀 Created ADVANCED stream widget for camera: {camera.name}")

        # Auto start
        if self.auto_start:
            self.parent.after(2000, self.start_stream)

    def _get_root_window(self):
        """Get root window and ensure it's set as default"""
        try:
            root = self.parent.winfo_toplevel()

            if root and root.winfo_exists():
                root.tk.call('wm', 'withdraw', '.')
                root.tk.call('wm', 'deiconify', root._w)

                if not hasattr(tk, '_default_root') or tk._default_root is None:
                    tk._default_root = root

                return root
            else:
                return None

        except Exception as e:
            ui_logger.error(f"❌ Error getting root window: {e}")
            return None

    def _ensure_root_is_default(self):
        """Ensure our root window is set as default"""
        try:
            if self.root_window and self.root_window.winfo_exists():
                tk._default_root = self.root_window
                return True
            else:
                self.root_window = self._get_root_window()
                return self.root_window is not None
        except Exception as e:
            ui_logger.error(f"❌ Error ensuring root is default: {e}")
            return False

    def create_widgets(self):
        """Tạo UI widgets với advanced controls"""
        # Main frame
        self.main_frame = ttk.Frame(self.parent)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Camera info header
        self.create_info_header()

        # Advanced control panel
        self.create_advanced_control_panel()

        # Video display area
        self.create_video_display()

        # Advanced status bar
        self.create_advanced_status_bar()

    def create_info_header(self):
        """Tạo header với advanced info"""
        info_frame = ttk.Frame(self.main_frame)
        info_frame.pack(fill=tk.X, pady=(0, 10))

        title_frame = ttk.Frame(info_frame)
        title_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        camera_title = ttk.Label(
            title_frame,
            text=f"🧠 {self.camera.name} (ADVANCED AI)",
            font=('Arial', 12, 'bold'),
            foreground='purple'
        )
        camera_title.pack(side=tk.LEFT)

        location_label = ttk.Label(
            title_frame,
            text=f"📍 {self.camera.location}",
            font=('Arial', 9)
        )
        location_label.pack(side=tk.LEFT, padx=(20, 0))

        # AI features info
        ai_label = ttk.Label(
            title_frame,
            text="🎯 Detection • 👤 Recognition • 📊 Tracking • 👥 Age/Gender • 💎 Quality",
            font=('Arial', 8),
            foreground='darkblue'
        )
        ai_label.pack(side=tk.LEFT, padx=(20, 0))

    def create_advanced_control_panel(self):
        """Tạo advanced control panel"""
        control_frame = ttk.Frame(self.main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # Left controls
        left_controls = ttk.Frame(control_frame)
        left_controls.pack(side=tk.LEFT)

        self.start_btn = ttk.Button(
            left_controls,
            text="🧠 Start Advanced AI",
            command=self.start_stream,
            width=20
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

        # Threshold control
        threshold_frame = ttk.Frame(left_controls)
        threshold_frame.pack(side=tk.LEFT, padx=10)

        ttk.Label(threshold_frame, text="Threshold:", font=('Arial', 8)).pack(side=tk.LEFT)
        self.threshold_var = tk.StringVar(value="0.5")
        threshold_scale = ttk.Scale(
            threshold_frame,
            from_=0.1,
            to=0.9,
            variable=self.threshold_var,
            orient=tk.HORIZONTAL,
            length=100,
            command=self.on_threshold_changed
        )
        threshold_scale.pack(side=tk.LEFT, padx=5)

        self.threshold_label = ttk.Label(threshold_frame, text="0.5", font=('Arial', 8))
        self.threshold_label.pack(side=tk.LEFT)

        # Right controls - Advanced features
        right_controls = ttk.Frame(control_frame)
        right_controls.pack(side=tk.RIGHT)

        # Feature toggles
        features_frame = ttk.Frame(right_controls)
        features_frame.pack(side=tk.RIGHT)

        self.recognition_cb = ttk.Checkbutton(
            features_frame,
            text="🔍 Recognition",
            command=self.toggle_recognition
        )
        self.recognition_cb.pack(side=tk.LEFT, padx=3)
        if self.recognition_enabled:
            self.recognition_cb.state(['selected'])

        self.tracking_cb = ttk.Checkbutton(
            features_frame,
            text="📊 Tracking",
            command=self.toggle_tracking
        )
        self.tracking_cb.pack(side=tk.LEFT, padx=3)
        if self.tracking_enabled:
            self.tracking_cb.state(['selected'])

        self.age_gender_cb = ttk.Checkbutton(
            features_frame,
            text="👥 Age/Gender",
            command=self.toggle_age_gender
        )
        self.age_gender_cb.pack(side=tk.LEFT, padx=3)
        if self.show_age_gender:
            self.age_gender_cb.state(['selected'])

        self.quality_cb = ttk.Checkbutton(
            features_frame,
            text="💎 Quality",
            command=self.toggle_quality
        )
        self.quality_cb.pack(side=tk.LEFT, padx=3)
        if self.show_quality_metrics:
            self.quality_cb.state(['selected'])

    def create_video_display(self):
        """Tạo video display area"""
        video_frame = ttk.Frame(self.main_frame)
        video_frame.pack(fill=tk.BOTH, expand=True)

        self.video_label = ttk.Label(
            video_frame,
            text="🧠 ADVANCED AI Camera Preview\n\n🎯 Detection + 👤 Recognition + 📊 Tracking\n👥 Age/Gender + 💎 Quality Analysis\n\nClick 'Start Advanced AI' to begin",
            font=('Arial', 14),
            background='black',
            foreground='cyan',
            anchor='center',
            justify='center'
        )
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.video_info = ttk.Label(
            video_frame,
            text="",
            font=('Arial', 8),
            background='black',
            foreground='yellow'
        )
        self.video_info.place(x=10, y=10)

    def create_advanced_status_bar(self):
        """Tạo advanced status bar"""
        status_frame = ttk.Frame(self.main_frame)
        status_frame.pack(fill=tk.X, pady=(5, 0))

        self.status_label = ttk.Label(
            status_frame,
            text="Status: Ready for ADVANCED AI",
            font=('Arial', 9),
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Advanced metrics
        self.fps_label = ttk.Label(
            status_frame,
            text="FPS: 0",
            font=('Arial', 9, 'bold'),
            relief=tk.SUNKEN,
            anchor=tk.E,
            width=10,
            foreground='blue'
        )
        self.fps_label.pack(side=tk.RIGHT, padx=(5, 0))

        self.tracks_label = ttk.Label(
            status_frame,
            text="Tracks: 0",
            font=('Arial', 9),
            relief=tk.SUNKEN,
            anchor=tk.E,
            width=12,
            foreground='green'
        )
        self.tracks_label.pack(side=tk.RIGHT, padx=(5, 0))

        self.recognition_label = ttk.Label(
            status_frame,
            text="Known: 0",
            font=('Arial', 9),
            relief=tk.SUNKEN,
            anchor=tk.E,
            width=12,
            foreground='orange'
        )
        self.recognition_label.pack(side=tk.RIGHT, padx=(5, 0))

    def on_threshold_changed(self, value):
        """Handle threshold change"""
        try:
            threshold = float(value)
            self.threshold_label.config(text=f"{threshold:.2f}")
            self.face_service.update_threshold(threshold)
            self.update_status(f"Recognition threshold: {threshold:.2f}")
        except Exception as e:
            ui_logger.error(f"Error updating threshold: {e}")

    def toggle_recognition(self):
        """Toggle recognition"""
        try:
            if self.recognition_cb:
                self.recognition_enabled = self.recognition_cb.instate(['selected'])
            else:
                self.recognition_enabled = not self.recognition_enabled

            status = "enabled" if self.recognition_enabled else "disabled"
            self.update_status(f"Recognition {status}")
            ui_logger.info(f"🔍 Recognition {status}")

        except Exception as e:
            ui_logger.error(f"❌ Error toggling recognition: {e}")

    def toggle_tracking(self):
        """Toggle tracking"""
        try:
            if self.tracking_cb:
                self.tracking_enabled = self.tracking_cb.instate(['selected'])
            else:
                self.tracking_enabled = not self.tracking_enabled

            status = "enabled" if self.tracking_enabled else "disabled"
            self.update_status(f"Tracking {status}")
            ui_logger.info(f"📊 Tracking {status}")

        except Exception as e:
            ui_logger.error(f"❌ Error toggling tracking: {e}")

    def toggle_age_gender(self):
        """Toggle age/gender display"""
        try:
            if self.age_gender_cb:
                self.show_age_gender = self.age_gender_cb.instate(['selected'])
            else:
                self.show_age_gender = not self.show_age_gender

            status = "enabled" if self.show_age_gender else "disabled"
            self.update_status(f"Age/Gender display {status}")

        except Exception as e:
            ui_logger.error(f"❌ Error toggling age/gender: {e}")

    def toggle_quality(self):
        """Toggle quality metrics display"""
        try:
            if self.quality_cb:
                self.show_quality_metrics = self.quality_cb.instate(['selected'])
            else:
                self.show_quality_metrics = not self.show_quality_metrics

            status = "enabled" if self.show_quality_metrics else "disabled"
            self.update_status(f"Quality metrics {status}")

        except Exception as e:
            ui_logger.error(f"❌ Error toggling quality: {e}")

    def start_stream(self):
        """Start advanced AI stream"""
        if self.running:
            return

        try:
            self.update_status("🧠 Starting ADVANCED AI stream...")
            ui_logger.info(f"🧠 Starting ADVANCED AI stream for camera: {self.camera.name}")

            # Ensure root window is ready
            if not self._ensure_root_is_default():
                self.show_error_display("Root window not ready")
                return

            # Test PhotoImage creation
            if not self._test_photoimage_creation():
                self.show_error_display("PhotoImage creation failed")
                return

            # Create video capture
            self.cap = cv2.VideoCapture(self.camera.rtsp_url)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

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
            self.video_thread = threading.Thread(target=self.advanced_video_loop, daemon=True)
            self.video_thread.start()

            # Start UI update loop
            self.schedule_ui_update()

            self.update_status(f"🧠 ADVANCED AI streaming with full InsightFace features...")
            ui_logger.info(f"✅ Started ADVANCED AI streaming camera: {self.camera.name}")

        except Exception as e:
            ui_logger.error(f"❌ Error starting ADVANCED AI stream: {e}")
            self.show_error_display(f"Stream Error: {str(e)}")
            self.update_status(f"Error: {str(e)}")

            if self.cap:
                self.cap.release()
                self.cap = None

    def advanced_video_loop(self):
        """Advanced video processing loop với full InsightFace"""
        ui_logger.info(f"🧠 ADVANCED AI video loop started for camera: {self.camera.name}")

        frame_time = 1.0 / self.target_fps

        while self.running:
            loop_start = time.time()

            try:
                ret, frame = self.cap.read()
                if not ret:
                    ui_logger.warning("Failed to read frame")
                    time.sleep(0.001)
                    continue

                self.current_frame = frame.copy()

                # Scale for display
                display_height = int(frame.shape[0] * self.display_scale)
                display_width = int(frame.shape[1] * self.display_scale)
                display_frame = cv2.resize(frame, (display_width, display_height))

                # Advanced face processing
                if self.recognition_enabled or self.tracking_enabled:
                    if self.tracking_enabled:
                        # Use advanced tracking
                        faces = self.face_service.track_faces_advanced(display_frame)
                    else:
                        # Use basic analysis
                        faces = self.face_service.analyze_faces_advanced(display_frame)

                    # Draw advanced annotations
                    self.draw_advanced_annotations(display_frame, faces)

                    # Update UI metrics
                    self.parent.after(0, self._update_advanced_metrics, faces)

                # Queue frame for display
                try:
                    frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

                    if not self.frame_queue.full():
                        self.frame_queue.put_nowait({
                            'frame': frame_rgb,
                            'frame_count': self.frame_count,
                            'timestamp': time.time()
                        })
                except queue.Full:
                    pass

                self.frame_count += 1

                # Frame rate control
                elapsed = time.time() - loop_start
                sleep_time = max(0, frame_time - elapsed)

                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                ui_logger.error(f"❌ Error in ADVANCED AI video loop: {e}")
                time.sleep(0.001)
                continue

        ui_logger.info(f"🧠 ADVANCED AI video loop ended for camera: {self.camera.name}")

    def draw_advanced_annotations(self, frame: np.ndarray, faces: list):
        """Draw advanced annotations với full InsightFace features"""
        for face in faces:
            try:
                bbox = face.get('bbox', [0, 0, 0, 0])
                if not bbox or len(bbox) < 4:
                    continue

                h, w = frame.shape[:2]
                x1, y1, x2, y2 = [
                    max(0, min(int(bbox[0]), w)),
                    max(0, min(int(bbox[1]), h)),
                    max(0, min(int(bbox[2]), w)),
                    max(0, min(int(bbox[3]), h))
                ]

                if x1 >= x2 or y1 >= y2:
                    continue

                # Colors based on recognition status
                is_known = face.get('is_known', False)
                color = (0, 255, 0) if is_known else (0, 165, 255)  # Green for known, Orange for unknown
                thickness = 2

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                # Prepare labels
                labels = []

                # Identity
                identity = face.get('identity', 'Unknown')
                confidence = face.get('confidence', 0.0)
                labels.append(f"{identity} ({confidence:.2f})")

                # Age and Gender (if enabled)
                if self.show_age_gender:
                    age = face.get('age', 0)
                    gender_text = face.get('gender_text', 'Unknown')
                    labels.append(f"{gender_text}, {age}y")

                # Track info (if tracking enabled)
                if self.tracking_enabled and 'track_id' in face:
                    track_id = face['track_id']
                    track_length = face.get('track_length', 0)
                    labels.append(f"Track #{track_id} ({track_length})")

                # Quality metrics (if enabled)
                if self.show_quality_metrics:
                    quality_score = face.get('quality_score', 0.0)
                    blur_score = face.get('blur_score', 0.0)
                    labels.append(f"Q:{quality_score:.2f} B:{blur_score:.0f}")

                # Draw labels
                y_offset = y1 - 10
                for i, label in enumerate(labels):
                    if y_offset < 20:
                        y_offset = y2 + 20 + (i * 20)

                    # Background for text
                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y_offset - text_height - 5), (x1 + text_width, y_offset + 5), color, -1)

                    # Text
                    cv2.putText(frame, label, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y_offset -= 25

                # Draw landmarks (if available)
                landmarks = face.get('landmarks', [])
                if landmarks and len(landmarks) >= 5:
                    for point in landmarks:
                        if len(point) >= 2:
                            cv2.circle(frame, (int(point[0]), int(point[1])), 2, (255, 255, 0), -1)

            except Exception as e:
                ui_logger.error(f"❌ Error drawing advanced annotation: {e}")
                continue

    def schedule_ui_update(self):
        """Schedule UI updates"""
        if not self.running:
            return

        try:
            if not self.frame_queue.empty():
                frame_data = self.frame_queue.get_nowait()
                self.update_video_display_advanced(frame_data)

        except queue.Empty:
            pass
        except Exception as e:
            ui_logger.error(f"Error in UI update: {e}")

        if self.running:
            self.parent.after(33, self.schedule_ui_update)  # ~30 FPS UI updates

    def update_video_display_advanced(self, frame_data):
        """Update video display with advanced features"""
        try:
            if not self.video_label or not self.video_label.winfo_exists():
                return

            frame_rgb = frame_data['frame']
            frame_count = frame_data['frame_count']

            # Ensure root is default
            if not self._ensure_root_is_default():
                return

            # Create PhotoImage
            try:
                frame_pil = Image.fromarray(frame_rgb)

                if self.root_window and self.root_window.winfo_exists():
                    frame_tk = ImageTk.PhotoImage(frame_pil, master=self.root_window)
                else:
                    frame_tk = ImageTk.PhotoImage(frame_pil)

                # Update video label
                self.video_label.config(image=frame_tk, text='')
                self.video_label.image = frame_tk

            except Exception as e:
                ui_logger.error(f"❌ PhotoImage creation failed: {e}")
                return

            # Update video info
            if self.video_info and self.video_info.winfo_exists():
                features = []
                if self.recognition_enabled:
                    features.append("🔍Recognition")
                if self.tracking_enabled:
                    features.append("📊Tracking")
                if self.show_age_gender:
                    features.append("👥Age/Gender")
                if self.show_quality_metrics:
                    features.append("💎Quality")

                info_text = f"Frame: {frame_count} | Features: {' '.join(features)}"
                self.video_info.config(text=info_text)

            # Update FPS
            self.update_fps()

        except Exception as e:
            ui_logger.error(f"❌ Error in advanced video display: {e}")

    def _update_advanced_metrics(self, faces):
        """Update advanced metrics display"""
        try:
            # Count known vs unknown faces
            known_faces = sum(1 for face in faces if face.get('is_known', False))

            # Get active tracks info
            if self.tracking_enabled:
                tracks_info = self.face_service.get_active_tracks_info()
                active_tracks = len(tracks_info)
            else:
                active_tracks = len(faces)

            # Update labels
            if self.recognition_label and self.recognition_label.winfo_exists():
                self.recognition_label.config(text=f"Known: {known_faces}")

            if self.tracks_label and self.tracks_label.winfo_exists():
                self.tracks_label.config(text=f"Tracks: {active_tracks}")

        except Exception as e:
            ui_logger.error(f"❌ Error updating advanced metrics: {e}")

    def update_fps(self):
        """Update FPS calculation"""
        self.fps_counter += 1
        current_time = time.time()

        if current_time - self.last_fps_time >= 0.5:
            self.current_fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = current_time

            self.parent.after(0, self._update_fps_display, self.current_fps)

    def _update_fps_display(self, fps):
        """Update FPS display"""
        try:
            if self.fps_label and self.fps_label.winfo_exists():
                color = 'green' if fps >= 25 else 'orange' if fps >= 15 else 'red'
                self.fps_label.config(text=f"FPS: {fps:.1f}", foreground=color)
        except Exception as e:
            ui_logger.error(f"❌ Error updating FPS display: {e}")

    def _test_photoimage_creation(self):
        """Test PhotoImage creation"""
        try:
            if not self._ensure_root_is_default():
                return False

            test_image = Image.new('RGB', (10, 10), color='black')

            if self.root_window and self.root_window.winfo_exists():
                test_photo = ImageTk.PhotoImage(test_image, master=self.root_window)
            else:
                test_photo = ImageTk.PhotoImage(test_image)

            del test_photo
            del test_image

            return True

        except Exception as e:
            ui_logger.error(f"❌ PhotoImage test failed: {e}")
            return False

    def clear_error_display(self):
        """Clear error display"""
        try:
            if self.video_label and self.video_label.winfo_exists():
                self.video_label.config(
                    text="🧠 Initializing ADVANCED AI...",
                    background='black',
                    foreground='cyan'
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
        """Stop advanced AI stream"""
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

        ui_logger.info(f"⏹️ Stopped ADVANCED AI streaming camera: {self.camera.name}")

    def clear_video_display(self):
        """Clear video display"""
        try:
            if self.video_label and self.video_label.winfo_exists():
                self.video_label.config(
                    image='',
                    text="🧠 ADVANCED AI Camera Preview\n\n🎯 Detection + 👤 Recognition + 📊 Tracking\n👥 Age/Gender + 💎 Quality Analysis\n\nClick 'Start Advanced AI' to begin",
                    background='black',
                    foreground='cyan'
                )
                self.video_label.image = None

            if self.video_info and self.video_info.winfo_exists():
                self.video_info.config(text="")

            if self.fps_label and self.fps_label.winfo_exists():
                self.fps_label.config(text="FPS: 0", foreground='blue')

            if self.tracks_label and self.tracks_label.winfo_exists():
                self.tracks_label.config(text="Tracks: 0")

            if self.recognition_label and self.recognition_label.winfo_exists():
                self.recognition_label.config(text="Known: 0")

        except Exception as e:
            ui_logger.error(f"❌ Error clearing display: {e}")

    def update_controls_state(self, streaming: bool):
        """Update control buttons state"""
        try:
            if streaming:
                if self.start_btn and self.start_btn.winfo_exists():
                    self.start_btn.config(state='disabled')
                if self.stop_btn and self.stop_btn.winfo_exists():
                    self.stop_btn.config(state='normal')
            else:
                if self.start_btn and self.start_btn.winfo_exists():
                    self.start_btn.config(state='normal')
                if self.stop_btn and self.stop_btn.winfo_exists():
                    self.stop_btn.config(state='disabled')
        except Exception as e:
            ui_logger.error(f"❌ Error updating controls: {e}")

    def update_status(self, message: str):
        """Update status message"""
        try:
            if self.status_label and self.status_label.winfo_exists():
                self.status_label.config(text=f"Status: {message}")
            else:
                ui_logger.info(f"🧠 {self.camera.name}: {message}")
        except Exception as e:
            ui_logger.error(f"❌ Error updating status: {e}")

    def cleanup(self):
        """Cleanup resources"""
        if self.running:
            self.stop_stream()
        ui_logger.info(f"🧹 Cleaned up ADVANCED AI stream widget for camera: {self.camera.name}")


# Alias for backward compatibility
CameraStreamWidget = AdvancedCameraStreamWidget
