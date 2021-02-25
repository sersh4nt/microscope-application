from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from libs.database_editor import DatabaseEditor
from libs.network_handler import NetworkHandler
from libs.camera import Camera
from libs.user_interfaces import main

import cv2
import sys
import os
import qimage2ndarray
from multiprocessing import Process


class MainWindow(QMainWindow, main.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.camera = Camera()
        self.microscopeView.initialize(camera=self.camera)
        self.microscopeView.setEnabled(True)

        self.main_path = os.getcwd()
        self.path = os.path.join(self.main_path, 'data')
        self.database_editor = DatabaseEditor(self.camera, self.path)
        self.network_handler = NetworkHandler(self.main_path)
        self.training_process = None

        self.connect()
        self.display_classes()

    def connect(self):
        self.databaseEditButton.clicked.connect(self.show_database_editor)
        self.database_editor.close_event.connect(self.enable_videostream)
        self.listView.itemClicked.connect(self.display_item)
        self.startTrainingButton.clicked.connect(self.train_network)
        self.camera.camera_err.connect(self.camera_error)
        self.database_editor.database_handler.update_classes.connect(self.display_classes)

    def camera_error(self):
        msg = 'Cannot get frames from camera!\nProceed to exit the application.'
        QMessageBox.critical(self, 'Error!', msg, QMessageBox.Close)
        self.close()

    def train_network(self):
        if self.training_process is None or not self.training_process.is_alive():
            self.training_process = Process(target=self.network_handler.train_network)
            self.training_process.start()
            message = 'Training process started.\n' \
                      'Please, do not close the application until training\'s done!\n' \
                      'You can check the progress down below at sidebar!'
            self.trainProgressBar.setVisible(True)
            self.trainProgressBar.setMaximum(self.network_handler.overall_progress)
            self.trainProgressBar.setValue(self.network_handler.current_progress)
            QMessageBox.information(self, 'Success!', message, QMessageBox.Ok)
        else:
            QMessageBox.warning(self, 'Attention', 'Training is still in progress!', QMessageBox.Ok)

    def stop_training_dialog(self):
        yes, cancel = QMessageBox.Yes, QMessageBox.Cancel
        message = 'Neural network training is still in progress.\nContinue exit?'
        return QMessageBox.warning(self, 'Warning!', message, yes | cancel)

    def show_database_editor(self):
        self.microscopeView.setEnabled(False)
        self.database_editor.stream_enabled = True
        self.database_editor.show()

    def enable_videostream(self):
        self.database_editor.stream_enabled = False
        self.microscopeView.setEnabled(True)

    def display_classes(self):
        self.listView.clear()
        self.listView.addItems(self.database_editor.database_handler.ideal_images.keys())

    def display_item(self, item=None):
        logs = self.database_editor.database_handler.ideal_images
        if len(logs):
            path = logs[item.text()] if item else list(logs.values())[0]
            image = cv2.imread(path)
            scale = (self.databaseComponentView.size().width() - 2) / image.shape[1]
            image = cv2.resize(image, None, fx=scale, fy=scale)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = qimage2ndarray.array2qimage(image)
            self.databaseComponentView.setPixmap(QPixmap.fromImage(image))

    def resizeEvent(self, ev):
        self.display_item()
        super(MainWindow, self).resizeEvent(ev)

    def closeEvent(self, ev):
        if self.training_process and self.training_process.is_alive():
            action = self.stop_training_dialog()
            if action == QMessageBox.Yes:
                self.training_process.terminate()
                self.camera.cap.release()
                cv2.destroyAllWindows()
                sys.exit()
            else:
                ev.ignore()
        else:
            sys.exit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()
