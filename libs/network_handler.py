from ctypes import *
import cv2
import os
from PyQt5.QtCore import *


def c_array(ctype, values):
    return (ctype * len(values))(*values)


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


lib = CDLL('libdarknet.so', RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

make_boxes = lib.make_boxes
make_boxes.argtypes = [c_void_p]
make_boxes.restype = POINTER(BOX)

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

num_boxes = lib.num_boxes
num_boxes.argtypes = [c_void_p]
num_boxes.restype = c_int

make_probs = lib.make_probs
make_probs.argtypes = [c_void_p]
make_probs.restype = POINTER(POINTER(c_float))

detect = lib.network_predict
detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

ndarray_image = lib.ndarray_to_image
ndarray_image.argtypes = [POINTER(c_ubyte), POINTER(c_long), POINTER(c_long)]
ndarray_image.restype = IMAGE

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

network_detect = lib.network_detect
network_detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]


class NetworkHandler(QObject):

    def __init__(self, path):
        super(NetworkHandler, self).__init__()
        self.path = path
        self.network = None
        self.classes = []
        self.data = []
        self.meta = None

        self.load_network()

    def load_network(self):
        with open(os.path.join(self.path, 'classes.txt'), 'r') as file:
            self.classes = file.read().strip('\n').split('\n')

        self.network = load_net(b'.cfg', b'.weights', 0)
        self.meta = load_meta(b'.data')

    def detect_image(self, image, thresh=0.5, higher_thresh=0.5, nms=0.45):
        boxes = make_boxes(self.network)
        probs = make_probs(self.network)
        num = num_boxes(self.network)
        network_detect(self.network, image, thresh, higher_thresh, nms, boxes, probs)
        result = []
        for j in range(num):
            for i in range(self.meta.classes):
                if probs[j][i]:
                    result.append((self.meta.names[i], probs[j][i], (boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h)))
        result = sorted(result, key=lambda x: x[-1])
        free_image(image)
        free_ptrs(cast(probs, POINTER(c_void_p)), num)
        return result

    def index_images(self):
        pass

