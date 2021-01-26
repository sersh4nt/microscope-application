from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from libs.image_editor import ImageEditor
# from libs.user_editor import UserEditor
from libs.camera import Camera
from libs.user_interfaces import main

import cv2
import sys
import os
import qimage2ndarray

IMG_EXTENSIONS = ('.BMP', '.GIF', '.JPG', '.JPEG', '.PNG', '.PBM', '.PGM', '.PPM', '.TIFF', '.XBM')


class MainWindow(QMainWindow, main.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.show()

        # camera view
        self.camera = Camera(0)
        self.microscopeView.initialize(camera=self.camera)
        self.microscopeView.setEnabled(True)

        # database
        self.logs = []
        self.component_counter = 0
        self.item_dict = {}
        self.image_editor = ImageEditor(self.camera, os.path.join(os.getcwd(), 'data'))
        self.main_path = os.getcwd()
        # self.user_editor = UserEditor()
        self.load_database()

        self.connect()

    def connect(self):
        self.databaseEditButton.clicked.connect(self._show_database_editor)
        self.image_editor.close_event.connect(self._enable_videostream)
        self.listView.itemClicked.connect(self._clicked_on_item)
        # self.operatorDataEditButton.clicked.connect(self._show_user_editor)

    def _show_database_editor(self):
        self.microscopeView.setEnabled(False)
        self.image_editor.stream_enabled = True
        self.image_editor.show()

    @pyqtSlot()
    def _enable_videostream(self):
        self.image_editor.stream_enabled = False
        self.microscopeView.setEnabled(True)

    # def _show_user_editor(self):
    #    self.user_editor.show()

    def _clicked_on_item(self, item):
        self.component_counter = self.item_dict[item.text()]
        self.display_item()

    def display_item(self):
        path = self.logs[self.component_counter]['path']
        image = cv2.imread(path)
        scale = (self.databaseComponentView.size().width() - 2) / image.shape[1]
        image = cv2.resize(image, None, fx=scale, fy=scale)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = qimage2ndarray.array2qimage(image)
        self.databaseComponentView.setPixmap(QPixmap.fromImage(image))

    def load_database(self):
        directory = os.path.join(os.getcwd(), 'data')
        self.logs = get_images(directory)
        i = 0
        for log in self.logs:
            self.listView.addItem(QListWidgetItem(log['name']))
            self.item_dict[log['name']] = i
            i += 1
        self.display_item()


def get_images(directory):
    res = []
    for dir in os.listdir(directory):
        path = os.path.join(directory, dir)
        if os.path.isdir(path):
            for file in os.listdir(os.path.join(directory, dir)):
                if file.upper().endswith(IMG_EXTENSIONS):
                    img_obj = {'name': dir, 'path': os.path.join(path, file)}
                    res.append(img_obj)
    return res


def terminate_app():
    sys.exit()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
