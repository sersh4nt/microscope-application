from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from libs.user_interfaces import designer
from libs.camera import Camera

import sys
import qimage2ndarray
import numpy as np
import cv2


class ImageEditor(QMainWindow, designer.Ui_MainWindow):
    close_event = pyqtSignal()

    def __init__(self, camera=None):
        super(ImageEditor, self).__init__()
        self.setupUi(self)
        self.camera = camera
        self.camera.new_frame.connect(self._on_new_frame)
        self.stream_enabled = False
        self.frame = None

        self.connect()

    def connect(self):
        self.shotButton.clicked.connect(self.stop_video)
        pass

    def stop_video(self):
        if self.stream_enabled:
            self.stream_enabled = False
            self.shotButton.setText(QCoreApplication.translate("MainWindow", "Включить видео"))
            self.canvas.setEditing(False)
        else:
            self.stream_enabled = True
            self.shotButton.setText(QCoreApplication.translate("MainWindow", "Сделать снимок"))
            self.canvas.setEditing(True)

    @pyqtSlot(np.ndarray)
    def _on_new_frame(self, frame):
        if self.stream_enabled:
            self.frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
            self.canvas.loadPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(self.frame)))

    def closeEvent(self, e):
        self.close_event.emit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageEditor()
    window.show()
    app.exec_()
