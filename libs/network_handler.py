from PyQt5.QtCore import *

from libs.yolo.datasets import *
from models.experimental import *
from models.yolo import *
from libs.yolo.general import *
from libs.yolo.plots import *
from libs.yolo.torch_utils import *
from libs.utils import *

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import torch

import yaml


class NetworkHandler(QObject):

    def __init__(self, path, device):
        super(NetworkHandler, self).__init__()
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

    def train_network(self, epochs=300, batch_size=16):
        rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
        # DDP parameter, do not modify
        local_rank = -1
        save_dir = os.path.join(self.path, 'models')
        weights_dir = os.path.join(save_dir, 'train')
        last = os.path.join(weights_dir, 'last.pt')
        best = os.path.join(weights_dir, 'best.pt')
        results_file = os.path.join(save_dir, 'results.txt')
        self.index_records(weights_dir)
        self.index_classes(self.path)

        data = check_file(os.path.join(weights_dir, 'data.yaml'))
        cfg = check_file(os.path.join(weights_dir, 'cfg.yaml'))
        hyp = check_file(os.path.join(weights_dir, 'hyp.yaml'))

        cuda = self.device.type != 'cpu'
        init_seeds(2 + rank)
        with open(data) as f:
            data_dict = yaml.load(f, Loader=yaml.SafeLoader)
        with open(hyp) as f:
            hyp_dict = yaml.load(f, Loader=yaml.SafeLoader)
        with torch_distributed_zero_first(rank):
            check_dataset(data_dict)
        train_path = data_dict['train']
        test_path = data_dict['val']
        nc = data_dict['nc']
        names = data_dict['names']
        assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, data)

        model = Model(cfg, ch=3, nc=nc).to(self.device)

        freeze = []
        for k, v in model.named_parameters():
            v.requires_grad = True
            if any(x in k for x in freeze):
                print('freezing %s' % k)
                v.requires_grad = False

        nbs = 64
        accumulate = max(round(nbs / 16), 1) # total batch size = 16 as default
        hyp_dict['weight_decay'] *= 16 * accumulate / nbs
        logger.info(f"Scaled weight_decay = {hyp_dict['weight_decay']}")

        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in model.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d):
                pg0.append(v.weight)  # no decay
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay

        # use adam optimizer, false as default
        adam = False
        if adam:
            optimizer = optim.Adam(pg0, lr=hyp_dict['lr0'], betas=(hyp_dict['momentum'], 0.999))  # adjust beta1 to momentum
        else:
            optimizer = optim.SGD(pg0, lr=hyp_dict['lr0'], momentum=hyp_dict['momentum'], nesterov=True)
        optimizer.add_param_group({'params': pg1, 'weight_decay': hyp_dict['weight_decay']})  # add pg1 with weight_decay
        optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
        del pg0, pg1, pg2

        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
        # false as default
        linear_lr = False
        if linear_lr:
            lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp_dict['lrf']) + hyp_dict['lrf']  # linear
        else:
            lf = one_cycle(1, hyp_dict['lrf'], epochs)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

        start_epoch, best_fitness = 0, 0.0

        # Image sizes
        gs = int(model.stride.max())  # grid size (max stride)
        nl = model.model[-1].nl  # number of detection layers (used for scaling hyp_dict['obj'])
        imgsz, imgsz_test = [check_img_size(x, gs) for x in [self.image_size, self.image_size]]  # verify imgsz are gs-multiples

        # DP mode
        if cuda and rank == -1 and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        # SyncBatchNorm, false as default
        sync_bn = False
        if sync_bn and cuda and rank != -1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(self.device)
            logger.info('Using SyncBatchNorm()')

        # EMA
        ema = ModelEMA(model) if rank in [-1, 0] else None

        # DDP mode
        if cuda and rank != -1:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        # Trainloader
        dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, False,
                                                hyp=hyp_dict, rank=rank,
                                                prefix=colorstr('train: '))
        mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
        nb = len(dataloader)  # number of batches
        assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, data, nc - 1)

    def index_classes(self, path):
        with open(os.path.join(path, 'data', 'classes.txt'), 'r') as f:
            self.classes = f.read().strip('\n').split('\n')

        with open(os.path.join(path, 'models', 'train', 'data.yaml'), 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        print(data)
        data['nc'] = len(self.classes)
        data['names'] = self.classes

        with open(os.path.join(path, 'models', 'train', 'data.yaml'), 'w') as f:
            data = yaml.dump(data, f)

        with open(os.path.join(path, 'models', 'train', 'cfg.yaml'), 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        data['nc'] = len(self.classes)

        with open(os.path.join(path, 'models', 'train', 'cfg.yaml'), 'w') as f:
            data = yaml.dump(data, f)

    def index_records(self, dir):
        # reindexing records
        paths = glob.glob(os.path.join(self.path, 'data', '**', '*.jpg'), recursive=True)
        combination = get_random_combination(len(paths), int(0.8 * len(paths)))
        train_paths = []
        val_paths = []
        for i, path in enumerate(paths):
            if i in combination:
                train_paths.append(path)
            else:
                val_paths.append(path)
        with open(os.path.join(dir, 'train.txt'), 'w') as f:
            f.write('\n'.join(train_paths))
        with open(os.path.join(dir, 'val.txt'), 'w') as f:
            f.write('\n'.join(val_paths))

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
