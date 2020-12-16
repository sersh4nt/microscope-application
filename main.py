from mainwindow import Ui_MainWindow
import sys
from PySide2.QtGui import QImage, QPixmap
from PySide2.QtWidgets import QApplication, QMainWindow
from PySide2.QtCore import QThread, Qt
from PyQt5.QtCore import pyqtSignal, pyqtSlot
import cv2

class Thread(QThread):
    changePixmap = pyqtSignal(QImage)

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret:
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                qtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = qtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.th = Thread(self)
        self.th.changePixmap.connect(self.setImage)
        self.th.start()
        self.show()

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.ui.microscopeView.setPixmap(QPixmap.fromImage(image))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
