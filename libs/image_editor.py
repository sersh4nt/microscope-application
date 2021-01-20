from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from libs.user_interfaces import designer_design
import sys
from libs.camera import Camera


class ImageEditor(QMainWindow, designer_design.Ui_MainWindow):
    close_event = pyqtSignal(int)

    def __init__(self, camera=None):
        super(ImageEditor, self).__init__()
        self.setupUi(self)
        # self.camera = camera
        # self.canvas.initialize(self.camera)

    def closeEvent(self, e):
        self.close_event.emit(1)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageEditor()
    window.show()
    app.exec_()
