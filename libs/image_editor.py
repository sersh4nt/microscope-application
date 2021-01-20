from PyQt5.QtWidgets import *
from libs.user_interfaces import designer_design
import sys
from libs.camera import Camera


class ImageEditor(QMainWindow, designer_design.Ui_MainWindow):
    def __init__(self, camera=None):
        super(ImageEditor, self).__init__()
        self.setupUi(self)
        self.camera = camera
        self.canvas.initialize(self.camera)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageEditor()
    window.show()
    app.exec_()