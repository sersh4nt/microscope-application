from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap
from image_editor import ImageEditor
# from user_editor import UserEditor
import qimage2ndarray
import main_window_design
import cv2
import sys

import os

IMG_EXTENSIONS = ('.BMP', '.GIF', '.JPG', '.JPEG', '.PNG', '.PBM', '.PGM', '.PPM', '.TIFF', '.XBM')


class MainWindow(QtWidgets.QMainWindow, main_window_design.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.video_capture = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.video_stream_qpixmap = QPixmap()
        self.logs = []
        self.component_counter = 0
        self.item_dict = {}
        self.image_editor = ImageEditor()
        #self.user_editor = UserEditor()

        self.initiate_video_stream()
        self.load_database()
        self.connect_buttons()

    def connect_buttons(self):
        self.databaseEditButton.clicked.connect(self._show_database_editor)
        # self.operatorDataEditBitton.clicked.connect(self._show_user_editor)

    def _show_database_editor(self):
        _, frame = self.video_capture.read()
        self.image_editor.show()
        self.image_editor.show_image(frame)


    # def _show_user_editor(self):
    #    self.user_editor.show()

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
        self.video_capture.set(cv2.CAP_PROP_FPS, 60)
        self.timer.timeout.connect(self.display_frame)
        self.timer.start(1)

    def _clicked_on_item(self, item):
        self.component_counter = self.item_dict[item.text()]
        self.display_item()

    def display_item(self):
        path = self.logs[self.component_counter]['path']
        image = cv2.imread(path)
        scale_factor = (self.databaseComponentView.width() - 2) / image.shape[1]
        image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = qimage2ndarray.array2qimage(image)
        self.databaseComponentView.setPixmap(QPixmap.fromImage(image))

    def load_database(self):
        directory = os.path.join(os.getcwd(), 'data')
        self.logs = get_images(directory)
        i = 0
        for log in self.logs:
            self.listView.addItem(QtWidgets.QListWidgetItem(log['name']))
            self.item_dict[log['name']] = i
            i += 1
        self.display_item()
        self.listView.itemClicked.connect(self._clicked_on_item)


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


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
