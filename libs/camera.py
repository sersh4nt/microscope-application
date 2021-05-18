import platform
import subprocess

import cv2
import numpy as np
import qimage2ndarray
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class Camera(QObject):
    _DEFAULT_FPS = 60
    new_frame = pyqtSignal(np.ndarray)
    camera_err = pyqtSignal()

    def __init__(self, camera_id=0, mirrored=False, parent=None):
        super(Camera, self).__init__(parent)
        self.mirrored = mirrored

        if platform.system() == 'Linux':
            command = "v4l2-ctl -d 0 -c exposure_auto=3"
            subprocess.call(command, shell=True)
            self.cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        elif platform.system() == 'Windows':
            self.cap = cv2.VideoCapture(camera_id)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._query_frame)
        self.timer.setInterval(1000 // self.fps)
        self.paused = False

    @pyqtSlot()
    def _query_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.camera_err.emit()
        else:
            if self.mirrored:
                frame = cv2.flip(frame, 1)
            self.new_frame.emit(frame)

    @property
    def paused(self):
        return not self.timer.isActive()

    @paused.setter
    def paused(self, p):
        if p:
            self.timer.stop()
        else:
            self.timer.start()

    @property
    def frame_size(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def fps(self):
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = self._DEFAULT_FPS
        return fps

    def get_frame(self):
        ret, frame = self.cap.read()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else None


class CameraWidget(QLabel):
    new_frame = pyqtSignal(np.ndarray)

    def __init__(self, camera=None, parent=None):
        super(CameraWidget, self).__init__(parent)
        self.frame = None
        if camera:
            self.initialize(camera)

    def initialize(self, camera):
        self.camera = camera
        self.camera.new_frame.connect(self._on_new_frame)
        self.frame_size = self.camera.frame_size

    @pyqtSlot(np.ndarray)
    def _on_new_frame(self, frame):
        self.frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
        self.new_frame.emit(self.frame)
        self.update()

    def changeEvent(self, e):
        if e.type() == QEvent.EnabledChange:
            if self.isEnabled():
                self.camera.new_frame.connect(self._on_new_frame)
            else:
                self.camera.new_frame.disconnect(self._on_new_frame)

    def paintEvent(self, e):
        if self.frame is None:
            return
        w, h = self.width(), self.height()
        scale = max(h / self.frame_size[1], w / self.frame_size[0])
        frame = cv2.resize(self.frame, None, fx=scale, fy=scale)
        painter = QPainter(self)
        painter.drawImage(QPoint(0, 0), qimage2ndarray.array2qimage(frame))