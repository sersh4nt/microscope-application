from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap
import qimage2ndarray
import design
import cv2
import sys


class MainWindow(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.video_capture = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.setupUi(self)
        self.initiate_video_stream()

    def display_frame(self):
        scale_factor = QtWidgets.QApplication.desktop().width() / self.frameGeometry().width()
        w, h = self.microscopeView.size().width(), self.microscopeView.size().height()
        ret, frame = self.video_capture.read()
        frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame[
                (frame.shape[0] - h) // 2: (frame.shape[0] + h) // 2 - 2,
                (frame.shape[1] - w) // 2: (frame.shape[1] + w) // 2 - 2
        ]
        img = qimage2ndarray.array2qimage(frame)
        self.microscopeView.setPixmap(QPixmap.fromImage(img))

    def initiate_video_stream(self):
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.timer.timeout.connect(self.display_frame)
        self.timer.start(1000 // 60)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
