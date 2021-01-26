from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from libs.user_interfaces import designer
from libs.widgets import Camera, LabelDialog
from libs.database import DatabaseHandler

import sys
import qimage2ndarray
import numpy as np
import cv2
import hashlib
import os


class ImageEditor(QMainWindow, designer.Ui_MainWindow):
    close_event = pyqtSignal()

    def __init__(self, camera=None, path=None):
        super(ImageEditor, self).__init__()
        self.setupUi(self)
        self.camera = camera
        self.stream_enabled = False
        self.single_class = False
        self.frame = None
        self.items_to_shapes = {}
        self.shapes_to_items = {}
        self.label_list = []
        self.prev_label_text = ''
        self.last_label = None

        self.label_dialog = LabelDialog(parent=self, listItem=self.label_list)
        self.database_handler = DatabaseHandler(path)

        self.scrollBars = {
            Qt.Vertical: self.scrollArea.verticalScrollBar(),
            Qt.Horizontal: self.scrollArea.horizontalScrollBar()
        }
        self.canvas.newShape.connect(self.new_shape)

        self.modeEdit.setChecked(True)
        self.modeSelect.setChecked(False)

        self.labelCoordinates = QLabel('')
        self.statusBar().addPermanentWidget(self.labelCoordinates)

        self.connect()

    def new_shape(self):
        if len(self.label_list) > 0:
            self.label_dialog = LabelDialog(parent=self, listItem=self.label_list)

        if self.last_label and self.single_class:
            text = self.last_label
        else:
            text = self.label_dialog.popUp(text=self.prev_label_text)
            self.last_label = text

        if text is not None:
            self.prev_label_text = text
            color = generate_color_by_text(text)
            shape = self.canvas.setLastLabel(text, color, color)
            self.add_label(shape)
            if text not in self.label_list:
                self.label_list.append(text)
        else:
            self.canvas.resetAllLines()

    def add_label(self, shape):
        item = HashableQListWidgetItem(shape.label)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        item.setBackground(generate_color_by_text(shape.label))
        self.items_to_shapes[item] = shape
        self.shapes_to_items[shape] = item
        self.objectList.addItem(item)

    def edit_label(self):
        item = self.current_item()
        if not item:
            return
        text = self.labelDialog.popUp(item.text())
        if text is not None:
            item.setText(text)
            item.setBackground(generate_color_by_text(text))

    def current_item(self):
        items = self.objectList.selectedItems()
        if items:
            return items[0]
        return None

    def connect(self):
        self.camera.new_frame.connect(self._on_new_frame)
        self.shotButton.clicked.connect(self._stop_video)
        self.modeSelect.triggered.connect(self._mode_select)
        self.modeEdit.triggered.connect(self._mode_edit)
        self.objectList.itemDoubleClicked.connect(self.edit_label)

    def _mode_select(self):
        self.modeEdit.setChecked(False)
        self.modeSelect.setChecked(True)
        self.canvas.setEditing(False)

    def _mode_edit(self):
        self.modeEdit.setChecked(True)
        self.modeSelect.setChecked(False)
        self.canvas.setEditing(True)

    def _stop_video(self):
        if self.stream_enabled:
            self.stream_enabled = False
            self.shotButton.setText(QCoreApplication.translate("MainWindow", "Включить видео"))
        else:
            self.stream_enabled = True
            self.shotButton.setText(QCoreApplication.translate("MainWindow", "Сделать снимок"))

    @pyqtSlot(np.ndarray)
    def _on_new_frame(self, frame):
        if self.stream_enabled:
            self.frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
            self.canvas.loadPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(self.frame)))

    def closeEvent(self, e):
        self.close_event.emit()


class HashableQListWidgetItem(QListWidgetItem):
    def __init__(self, *args):
        super(HashableQListWidgetItem, self).__init__(*args)

    def __hash__(self):
        return hash(id(self))


def generate_color_by_text(text):
    hash = int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16)
    r = int((hash / 255) % 255)
    g = int((hash / 65025) % 255)
    b = int((hash / 16581375) % 255)
    return QColor(r, g, b, 100)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageEditor()
    window.show()
    app.exec_()
