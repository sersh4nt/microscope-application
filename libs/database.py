from datetime import datetime
from dataclasses import dataclass
from libs.utils import *
import os
import cv2
import shutil


class DatabaseHandler(QObject):
    update_classes = pyqtSignal()

    def __init__(self, path):
        super(DatabaseHandler, self).__init__()
        self.path = path
        self.records = dict()
        self.classes = []
        self.ideal_images = {}
        self.load()

    def delete_class(self, component):
        shutil.rmtree(os.path.join(self.path, component))
        self.classes.remove(component)
        with open(os.path.join(self.path, 'classes.txt'), 'w') as file:
            for component in self.classes:
                file.write(component + '\n')

    def add_class(self, component):
        component_dir = os.path.join(self.path, component)
        os.mkdir(component_dir)
        records_dir = os.path.join(component_dir, 'records')
        os.mkdir(records_dir)

        if component not in self.classes:
            self.classes.append(component)
            d = os.path.join(self.path, 'classes.txt')
            with open(d, 'r') as f:
                text = f.read()
            with open(d, 'a') as f:
                if not text.endswith('\n'):
                    f.write('\n')
                f.write(component)

    def index_ideal_images(self):
        self.ideal_images.clear()
        self.get_ideal_images()
        self.update_classes.emit()

    def add_ideal_image(self, image, component):
        if component not in self.classes:
            self.classes.append(component)
            d = os.path.join(self.path, 'classes.txt')
            with open(d, 'r') as f:
                text = f.read()
            with open(d, 'a') as f:
                if not text.endswith('\n'):
                    f.write('\n')
                f.write(component)
        os.chdir(os.path.join(self.path, component))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(component + '.jpeg', image)
        self.index_ideal_images()
        os.chdir(self.path)

    def get_ideal_images(self):
        for dir in os.listdir(self.path):
            path = os.path.join(self.path, dir)
            if os.path.isdir(path):
                for file in os.listdir(os.path.join(self.path, dir)):
                    if file.lower().endswith('.jpeg'):
                        self.ideal_images[dir] = os.path.join(path, file)

    def load(self):
        if not os.path.exists(self.path):
            os.mkdir('data')

        if not os.path.exists(os.path.join(self.path, 'classes.txt')):
            open(os.path.join(self.path, 'classes.txt'), 'w')

        with open(os.path.join(self.path, 'classes.txt'), 'r') as file:
            self.classes = file.read().strip('\n').strip('').split('\n')

        if self.classes == ['']:
            self.classes = []

        for component_name in os.listdir(self.path):
            component_folder = os.path.join(self.path, component_name)
            if os.path.isdir(component_folder):
                for folder in os.listdir(component_folder):
                    if folder == 'records':
                        records_folder = os.path.join(component_folder, folder)
                        for record in os.listdir(records_folder):
                            if record.lower().endswith('.txt'):
                                if component_name not in self.records:
                                    self.records[component_name] = list()
                                self.records[component_name].append(DataRecord().load(record, component_name))

        self.get_ideal_images()

    def add_record(self, component, img, shapes):
        dir = os.path.join(self.path, component)
        # check if component class folder exists
        if not os.path.exists(dir):
            os.mkdir(dir)
        dir = os.path.join(dir, 'records')
        # check if records folder exists
        if not os.path.exists(dir):
            os.mkdir(dir)
        os.chdir(dir)

        current_date = datetime.now().strftime("%d-%m-%Y")
        current_number = -1
        for file in os.listdir(dir):
            if file.lower().endswith('.txt'):
                date = file[:10]
                if date == current_date:
                    current_number = max(current_number, int(file[11:14]))
        current_number += 1
        filename = "{}{:04d}".format(current_date, current_number)

        image = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename + '.jpg', image)

        shapes = [shape2dict(shape) for shape in shapes]
        img_h, img_w = img.shape[0], img.shape[1]
        if component not in self.classes:
            self.classes.append(component)
            d = os.path.join(self.path, 'classes.txt')
            with open(d, 'r') as f:
                text = f.read()
            with open(d, 'a') as f:
                if not text.endswith('\n'):
                    f.write('\n')
                f.write(component)
        index = self.classes.index(component)

        with open(filename + '.txt', 'w') as file:
            for shape in shapes:
                xcen, ycen, w, h = points2yolo(shape['points'])
                xcen /= img_w
                ycen /= img_h
                w /= img_w
                h /= img_h
                file.write("%d %.6f %.6f %.6f %.6f\n" % (index, xcen, ycen, w, h))

        os.chdir(self.path)

        if component not in self.records:
            self.records[component] = list()
        self.records[component].append(DataRecord().load(filename + '.jpg', component))

    def edit_record(self, component, filename, frame, shapes):
        os.chdir(os.path.join(self.path, component, 'records'))

        image = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename + '.jpg', image)

        shapes = [shape2dict(shape) for shape in shapes]
        img_h, img_w = frame.shape[0], frame.shape[1]
        if component not in self.classes:
            self.classes.append(component)
            d = os.path.join(self.path, 'classes.txt')
            with open(d, 'r') as f:
                text = f.read()
            with open(d, 'a') as f:
                if not text.endswith('\n'):
                    f.write('\n')
                f.write(component)
        index = self.classes.index(component)

        with open(filename + '.txt', 'w') as file:
            for shape in shapes:
                xcen, ycen, w, h = points2yolo(shape['points'])
                xcen /= img_w
                ycen /= img_h
                w /= img_w
                h /= img_h
                file.write("%d %.6f %.6f %.6f %.6f\n" % (index, xcen, ycen, w, h))

        os.chdir(self.path)


@dataclass
class DataRecord:
    date: str = datetime.now().strftime("%d-%m-%Y")
    image: str = ''
    component: str = ''
    number: int = 0

    '''directory='data/<class_name>', example: directory='data/ad620b' '''
    def save(self, class_directory):
        if not os.path.exists(os.path.join(class_directory, 'records')):
            pass

    def load(self, record='', component=''):
        return DataRecord(record[:10], record, component, int(record[11:14]))

    def __str__(self):
        return '{} {} {} {:04d}'.format(self.date, self.image, self.component, self.number)

    def __repr__(self):
        return '{} {} {} {:04d}'.format(self.date, self.image, self.component, self.number)
