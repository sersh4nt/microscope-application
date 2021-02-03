from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from libs.user_interfaces import designer
from libs.utils import *
from libs.label_dialog import LabelDialog
from libs.database import DatabaseHandler
from libs.shape import Shape

import sys
import qimage2ndarray
import numpy as np
import cv2
import os


class ImageEditor(QMainWindow, designer.Ui_MainWindow):
    close_event = pyqtSignal()

    def __init__(self, camera=None, path=None):
        super(ImageEditor, self).__init__()
        self.setupUi(self)
        self.camera = camera
        self.stream_enabled = False
        self.single_class = False
        self.dirty = False
        self.frame = None
        self.items_to_shapes = {}
        self.shapes_to_items = {}
        self.label_list = []
        self.prev_label_text = ''
        self.last_label = None
        self.shapes = []
        self.path = path
        self.rewrite = False
        self.current_component = ''
        self.current_filename = ''

        self.label_dialog = LabelDialog(parent=self, listItem=self.label_list)
        self.database_handler = DatabaseHandler(path)

        self.scrollBars = {
            Qt.Vertical: self.scrollArea.verticalScrollBar(),
            Qt.Horizontal: self.scrollArea.horizontalScrollBar()
        }
        self.canvas.newShape.connect(self.new_shape)

        # context menus
        delete_record_action = Action(self, 'Delete record', self.delete_record, enabled=True)
        add_record_action = Action(self, 'Add record', self.add_record, enabled=True)
        self.record_menu = QMenu()
        add_actions(self.record_menu, (delete_record_action, add_record_action))

        delete_component_action = Action(self, 'Delete component', self.delete_component, enabled=True)
        add_component_action = Action(self, 'Add component', self.add_component, enabled=True)
        add_component_image_action = Action(self, 'Add components image', self.add_component_image, enabled=True)
        self.component_menu = QMenu()
        add_actions(self.component_menu, (delete_component_action, add_component_action, add_component_image_action))

        delete_shape_action = Action(self, 'Delete rectangle', self.delete_shape, enabled=True)
        self.rectangle_menu = QMenu()
        add_actions(self.rectangle_menu, delete_shape_action)

        self.modeEdit.setChecked(True)
        self.modeSelect.setChecked(False)

        self.labelCoordinates = QLabel('')
        self.statusBar().addPermanentWidget(self.labelCoordinates)

        self.connect()
        self.display_classes()

    def delete_shape(self):
        self.remove_label(self.canvas.deleteSelected())
        self.rewrite = True
        self.save_labels()

    def remove_label(self, shape):
        if shape is None:
            return
        self.shapes.remove(shape)
        item = self.shapes_to_items[shape]
        self.rectangleList.takeItem(self.rectangleList.row(item))
        del self.shapes_to_items[shape]
        del self.items_to_shapes[item]

    def add_component_image(self):
        image = self.frame
        component = self.get_current_component().text()
        self.database_handler.add_ideal_image(image, component)

    def delete_component(self):
        component = self.get_current_component()
        if not component:
            return
        component = component.text()
        self.database_handler.delete_class(component)
        self.clear()
        self.display_classes()

    def add_component(self):
        self.clear()
        component = self.label_dialog.popUp()
        if component == '':
            return
        self.database_handler.add_class(component)
        self.display_classes()

    def new_shape(self):
        text = self.get_current_component().text()
        if text is not None:
            color = generate_color_by_text(text)
            shape = self.canvas.setLastLabel(text, color, color)
            self.add_label(shape)
            self.shapes.append(shape)
            if text not in self.label_list:
                self.label_list.append(text)
            self.set_dirty()
        else:
            self.canvas.resetAllLines()
        self._mode_select()

    def display_classes(self):
        self.componentList.clear()
        for key in self.database_handler.classes:
            self.componentList.addItem(key)

    def display_records(self):
        item = self.componentList.selectedItems()[0]
        if not item:
            return
        self.recordList.clear()
        if item.text() in self.database_handler.records.keys():
            for record in self.database_handler.records[item.text()]:
                self.recordList.addItem("{} №{}".format(record.date, record.number))

    def load_record(self):
        self.clear_labels()
        self.rewrite = True
        item = self.recordList.selectedItems()[0]
        if not item:
            return
        if self.stream_enabled:
            self._stop_video()

        component = self.componentList.selectedItems()[0].text()
        filename = item.text().replace('№', '')
        f = filename.split(' ')
        filename = '{}{:04d}'.format(f[0], int(f[1]))
        path = os.path.join(self.path, component, 'records', filename)

        self.current_component = component
        self.current_filename = filename

        img_path = path + '.jpg'
        txt_path = path + '.txt'
        self.frame = cv2.imread(img_path)
        if self.frame is None:
            return
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        self.canvas.loadPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(self.frame)))
        self.canvas.adjustSize()

        # loading shapes from .txt file
        with open(os.path.join(self.path, 'classes.txt')) as classes_file:
            classes = classes_file.read().strip('\n').split('\n')

        with open(txt_path, 'r') as bndboxes_file:
            for box in bndboxes_file:
                index, xcen, ycen, w, h = box.strip().split(' ')
                label = classes[int(index)]
                xmin, ymin, xmax, ymax = yolo2points(xcen, ycen, w, h, self.frame.shape[1], self.frame.shape[0])
                points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]

                shape = Shape(label=label)
                for x, y in points:
                    x, y, snapped = self.canvas.snapPointToCanvas(x, y)
                    if snapped:
                        self.set_dirty()
                    shape.addPoint(QPointF(x, y))
                shape.difficult = False
                shape.fill_color = generate_color_by_text(label)
                shape.line_color = generate_color_by_text(label)
                shape.close()
                self.shapes.append(shape)
                self.add_label(shape)
        self.canvas.loadShapes(self.shapes)

    def delete_record(self):
        record = self.get_current_record()
        component = self.get_current_component()
        if not record or not component:
            return
        component = component.text()
        filename = record.text().replace('№', '')
        f = filename.split(' ')
        filename = '{}{:04d}'.format(f[0], int(f[1]))
        path = os.path.join(self.path, component, 'records', filename)
        os.remove(path + '.jpg')
        os.remove(path + '.txt')
        for i, o in enumerate(self.database_handler.records[component]):
            if o.image == filename + '.txt':
                del self.database_handler.records[component][i]
                break
        self.clear_labels()
        self.display_records()

    def add_record(self):
        component = self.get_current_component()
        if not component:
            return
        component = component.text()
        if not self.stream_enabled:
            self.database_handler.add_record(component, self.frame, self.shapes)
            self.display_records()

    def clear_labels(self):
        self.label_list.clear()
        self.last_label = None
        self.items_to_shapes.clear()
        self.shapes_to_items.clear()
        self.prev_label_text = ''
        self.shapes.clear()
        if self.rectangleList.count():
            self.rectangleList.clear()

    def add_label(self, shape):
        item = HashableQListWidgetItem(shape.label)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        item.setBackground(generate_color_by_text(shape.label))
        self.items_to_shapes[item] = shape
        self.shapes_to_items[shape] = item
        self.rectangleList.addItem(item)

    def edit_label(self):
        item = self.get_current_rectangle()
        if not item:
            return
        text = self.label_dialog.popUp(item.text())
        if text is not None:
            item.setText(text)
            item.setBackground(generate_color_by_text(text))
            self.set_dirty()

    def save_labels(self):
        if self.rewrite:
            self.database_handler.edit_record(
                self.current_component,
                self.current_filename,
                self.frame,
                self.shapes
            )
        else:
            text = self.label_dialog.popUp(self.last_label)
            self.database_handler.add_record(text, self.frame, self.shapes)
            self.display_classes()
        self.rewrite = False

    def connect(self):
        self.camera.new_frame.connect(self._on_new_frame)

        self.canvas.shapeMoved.connect(self.set_dirty)

        self.shotButton.clicked.connect(self._stop_video)
        self.saveButton.clicked.connect(self.save_labels)

        self.modeSelect.triggered.connect(self._mode_select)
        self.modeEdit.triggered.connect(self._mode_edit)

        self.componentList.customContextMenuRequested.connect(self.class_menu_popup)
        self.componentList.itemClicked.connect(self.display_records)

        self.recordList.itemClicked.connect(self.load_record)
        self.recordList.customContextMenuRequested.connect(self.record_menu_popup)

        self.rectangleList.itemDoubleClicked.connect(self.edit_label)
        self.rectangleList.itemActivated.connect(self.select_shape)
        self.rectangleList.itemSelectionChanged.connect(self.select_shape)
        self.rectangleList.itemChanged.connect(self.change_shape)
        self.rectangleList.customContextMenuRequested.connect(self.rectangle_menu_popup)

    def change_shape(self, item):
        shape = self.items_to_shapes[item]
        label = item.text()
        if label != shape.label:
            shape.label = label
            shape.line_color = generate_color_by_text(label)
            self.set_dirty()
        else:
            self.canvas.setShapeVisible(shape, True)

    def select_shape(self):
        item = self.get_current_rectangle()
        if item:
            shape = self.items_to_shapes[item]
            self.canvas.selectShape(shape)

    # helpers
    def get_current_rectangle(self):
        items = self.rectangleList.selectedItems()
        if items:
            return items[0]
        return None

    def get_current_record(self):
        records = self.recordList.selectedItems()
        if records:
            return records[0]
        return None

    def get_current_component(self):
        component = self.componentList.selectedItems()
        if component:
            return component[0]
        return None

    def set_dirty(self):
        self.dirty = True

    def clear(self):
        self.rewrite = False
        if not self.stream_enabled:
            self._stop_video()
        self._mode_edit()
        self.shapes.clear()
        self.shapes_to_items.clear()
        self.items_to_shapes.clear()
        if self.recordList.count():
            self.recordList.clear()
        if self.rectangleList.count():
            self.rectangleList.clear()
        self.canvas.resetAllLines()
        self.canvas.adjustSize()

    # signal functions
    def record_menu_popup(self, point):
        self.record_menu.exec_(self.recordList.mapToGlobal(point))

    def class_menu_popup(self, point):
        self.component_menu.exec_(self.componentList.mapToGlobal(point))

    def rectangle_menu_popup(self, point):
        self.rectangle_menu.exec_(self.rectangleList.mapToGlobal(point))

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

    # event functions
    def closeEvent(self, e):
        self.clear()
        self.close_event.emit()

    def resizeEvent(self, ev):
        self.canvas.adjustSize()
        self.canvas.update()
        super(ImageEditor, self).resizeEvent(ev)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageEditor()
    window.show()
    app.exec_()
