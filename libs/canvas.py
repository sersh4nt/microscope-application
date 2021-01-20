from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import numpy as np


class Canvas(QWidget):
    new_frame = pyqtSignal(np.ndarray)
    
    def __init__(self, parent=None):
        super(Canvas, self).__init__(parent)