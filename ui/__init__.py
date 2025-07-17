"""
UI package for Camera RTSP Application
"""
from .main_window import MainWindow
from .camera_dialog import CameraSelectionDialog
from .stream_widget import CameraStreamWidget
from .face_manager import FaceManagerDialog, AddFaceDialog

__all__ = [
    'MainWindow',
    'CameraSelectionDialog',
    'CameraStreamWidget',
    'FaceManagerDialog',
    'AddFaceDialog'
]