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