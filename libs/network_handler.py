from PyQt5.QtCore import *

from libs.yolo.datasets import *
from models.experimental import *
from libs.yolo.general import *
from libs.yolo.plots import *
from libs.yolo.torch_utils import *


class NetworkHandler(QObject):

    def __init__(self, path, device):
        super(NetworkHandler, self).__init__()
        self.image_paths = []
        self.path = path
        self.network = None
        self.classes = []
        self.data = []
        self.meta = None
        self.device_name = device
        self.path = path
        self.device = None
        self.half = False
        self.model = None
        self.stride = 0
        self.image_size = 640

        self.load_network()
        self.load_image_paths()

    def load_image_paths(self):
        path = os.path.join(self.path, 'models', 'train')
        with open(os.path.join(path, 'train.txt'), 'r') as f:
            [self.image_paths.append(img) for img in f.read().strip('\n').split('\n')]
        with open(os.path.join(path, 'val.txt'), 'r') as f:
            [self.image_paths.append(img) for img in f.read().strip('\n').split('\n')]

    def load_network(self):
        # with open(os.path.join(self.path, 'classes.txt'), 'r') as file:
        #     self.classes = file.read().strip('\n').split('\n')
        path = os.path.join(self.path, 'models', 'train', 'best.pt')

        set_logging()
        self.device = select_device(self.device_name)
        self.half = self.device.type != 'cpu'
        self.model = attempt_load(path, map_location=self.device)
        self.stride = int(self.model.stride.max())
        if self.half:
            self.model.half()

    def train_network(self):
        # data = data.yaml file
        # cfg = model.yaml file
        # hyp = hyperparameters path

        self.update_indices()
        save_dir = os.path.join(self.path, 'models')
        weights_dir = os.path.join(save_dir, 'train')
        last = os.path.join(weights_dir, 'last.pt')
        best = os.path.join(weights_dir, 'best.pt')
        results_file = os.path.join(save_dir, 'results.txt')

    def update_indices(self):
        pass

    # this does work
    def detect(self, image):
        possible_result = []

        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        t0 = time.time()
        self.image_size = check_img_size(640, s=self.stride)
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.image_size, self.image_size)) \
                .to(self.device) \
                .type_as(next(self.model.parameters()))

        image = letterbox(image, self.image_size, stride=self.stride)[0]
        image = image[:, :, ::-1].transpose(2, 0, 1)
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).to(self.device)
        image = image.half() if self.half else image.float()
        image /= 255
        if image.ndimension() == 3:
            image = image.unsqueeze(0)

        t1 = time_synchronized()
        predictions = self.model(image)[0]
        predictions = non_max_suppression(predictions, 0.25, 0.45)
        t2 = time_synchronized()

        for detection in predictions:
            if len(detection):
                for *xyxy, conf, cls in reversed(detection):
                    possible_result.append(f'{names[int(cls)]} {conf:.2f}')
        print(predictions)

        return possible_result
