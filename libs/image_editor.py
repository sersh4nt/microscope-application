from PyQt5.QtWidgets import *
from libs.user_interfaces import designer_design
import sys

class ImageEditor(QMainWindow, designer_design.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageEditor()
    window.show()
    app.exec_()