from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from libs.database_editor import DatabaseEditor
from libs.network_handler import NetworkHandler
# from libs.user_editor import UserEditor
from libs.camera import Camera
from libs.user_interfaces import main

import cv2
import sys
import os
import qimage2ndarray
import torch
import threading


class MainWindow(QMainWindow, main.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # camera view
        self.camera = Camera(0)
        self.microscopeView.initialize(camera=self.camera)
        self.microscopeView.setEnabled(True)

        # database
        self.logs = []
        self.path = os.path.join(os.getcwd(), 'data')
        self.main_path = os.getcwd()
        self.component_counter = 0
        self.item_dict = {}
        self.database_editor = DatabaseEditor(self.camera, self.path)
        self.network_handler = NetworkHandler(self.main_path, '0' if torch.cuda.is_available() else 'cpu')
        # self.user_editor = UserEditor()
        self.load_database()

        self.connect()

    def connect(self):
        self.databaseEditButton.clicked.connect(self._show_database_editor)
        self.database_editor.close_event.connect(self._enable_videostream)
        self.listView.itemClicked.connect(self._clicked_on_item)
        # self.operatorDataEditButton.clicked.connect(self._show_user_editor)
        self.database_editor.database_handler.reload_database.connect(self._reload_database)
        self.startTrainingButton.clicked.connect(self._train_network)

    def _train_network(self):
        th = threading.Thread(target=self.network_handler.train_network)
        th.start()
        th.join()

    def _show_database_editor(self):
        self.microscopeView.setEnabled(False)
        self.database_editor.stream_enabled = True
        self.database_editor.show()

    @pyqtSlot()
    def _reload_database(self):
        self.listView.clear()
        self.load_database()

    @pyqtSlot()
    def _enable_videostream(self):
        self.database_editor.stream_enabled = False
        self.microscopeView.setEnabled(True)

    # def _show_user_editor(self):
    #    self.user_editor.show()

    def _clicked_on_item(self, item):
        self.component_counter = self.item_dict[item.text()]
        self.display_item()

    def display_item(self):
        if self.logs and len(self.logs):
            path = self.logs[self.component_counter]['path']
            image = cv2.imread(path)
            scale = (self.databaseComponentView.size().width() - 2) / image.shape[1]
            image = cv2.resize(image, None, fx=scale, fy=scale)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = qimage2ndarray.array2qimage(image)
            self.databaseComponentView.setPixmap(QPixmap.fromImage(image))

    def load_database(self):
        self.logs = get_images(self.path)
        i = 0
        for log in self.logs:
            self.listView.addItem(QListWidgetItem(log['name']))
            self.item_dict[log['name']] = i
            i += 1
        self.display_item()

    def resizeEvent(self, ev):
        self.display_item()
        super(MainWindow, self).resizeEvent(ev)

    def closeEvent(self, ev):
        terminate_app()
        super(MainWindow, self).closeEvent(ev)


def get_images(directory):
    res = []
    for dir in os.listdir(directory):
        path = os.path.join(directory, dir)
        if os.path.isdir(path):
            for file in os.listdir(os.path.join(directory, dir)):
                if file.lower().endswith('.jpeg'):
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
