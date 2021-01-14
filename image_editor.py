from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap
import designer_design
import qimage2ndarray
import cv2
import sys


class ImageEditor(QtWidgets.QMainWindow, designer_design.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.qpixmap = QPixmap()
        self.frame = None
        self.setupUi(self)

    def show_image(self, frame):
        self.frame = frame
        w, h = self.cameraView.size().width(), self.cameraView.size().height()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame[
                (frame.shape[0] - h) // 2: (frame.shape[0] + h) // 2 - 2,
                (frame.shape[1] - w) // 2: (frame.shape[1] + w) // 2 - 2
                ]
        image = qimage2ndarray.array2qimage(frame)
        self.cameraView.setPixmap(QPixmap.fromImage(image))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = ImageEditor()
    window.show()
    app.exec_()