from math import sqrt
import hashlib
import re
import sys

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *


def distance(p):
    return sqrt(p.x() * p.x() + p.y() * p.y())


def labelValidator():
    return QRegExpValidator(QRegExp(r'^[^ \t].+'), None)


def newIcon(icon):
    return QIcon(':/' + icon)


def points2yolo(points):
    xmin = float('inf')
    ymin = float('inf')
    xmax = float('-inf')
    ymax = float('-inf')
    for point in points:
        x = point[0]
        y = point[1]
        xmin = min(x, xmin)
        ymin = min(y, ymin)
        xmax = max(x, xmax)
        ymax = max(y, ymax)

    xcen = float(xmin + xmax) / 2
    ycen = float(ymin + ymax) / 2
    w = float(xmax - xmin)
    h = float(ymax - ymin)

    return xcen, ycen, w, h


def yolo2points(xcen, ycen, w, h, img_w, img_h):
    xmin = max(float(xcen) - float(w) / 2, 0)
    xmax = min(float(xcen) + float(w) / 2, 1)
    ymin = max(float(ycen) - float(h) / 2, 0)
    ymax = min(float(ycen) + float(h) / 2, 1)

    xmin = int(img_w * xmin)
    xmax = int(img_w * xmax)
    ymin = int(img_h * ymin)
    ymax = int(img_h * ymax)
    return xmin, ymin, xmax, ymax


def shape2dict(shape):
    return dict(
        label=shape.label,
        line_color=shape.line_color.getRgb(),
        fill_color=shape.fill_color.getRgb(),
        points=[(p.x(), p.y()) for p in shape.points],
        difficult=shape.difficult
    )
