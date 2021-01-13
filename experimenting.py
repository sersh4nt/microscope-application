from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer
import cv2 # OpenCV
import qimage2ndarray # for a memory leak,see gist
import sys # for exiting

# Minimal implementation...

def displayFrame():
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = qimage2ndarray.array2qimage(frame)
    label.setPixmap(QPixmap.fromImage(image))

app = QtWidgets.QApplication([])
window = QtWidgets.QWidget()

# OPENCV

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# timer for getting frames

timer = QTimer()
timer.timeout.connect(displayFrame)
timer.start(60)
label = QtWidgets.QLabel('No Camera Feed')
button = QtWidgets.QPushButton("Quiter")
button.clicked.connect(sys.exit) # quiter button
layout = QtWidgets.QVBoxLayout()
layout.addWidget(button)
layout.addWidget(label)
window.setLayout(layout)
window.show()
app.exec_()

# See also: https://gist.github.com/bsdnoobz/8464000

