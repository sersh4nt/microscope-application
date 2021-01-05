import sys
from PyQt5 import QtWidgets
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PIL import Image
import dessing as design
import cv2


class Thread(QThread):
    changePixmap = pyqtSignal(QImage)

    def run(self):
        cap = cv2.VideoCapture(-1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        while True:
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                maxsize = (QImage.width(), QImage.height())
                frame.thumbnail(maxsize, Image.ANTIALIAS)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qt_format = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                p = qt_format.scaled(1920, 1080, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)


class MainWindow(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.initiate_video_stream()

    @pyqtSlot(QImage)
    def set_image(self, image):
        self.microscopeView.setPixmap(QPixmap.fromImage(image))

    def initiate_video_stream(self):
        self.microscopeView.scaleFactor = 1.0
        self.microscopeView.imageRelativeScale = self.frameGeometry().width() / QtWidgets.QApplication.desktop().width()
        print(self.microscopeView.imageRelativeScale)
        th = Thread(self)
        th.changePixmap.connect(self.set_image)
        th.start()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
