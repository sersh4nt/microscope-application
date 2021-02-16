from PyQt5.QtCore import *

from libs.yolo.datasets import *
from models.experimental import *
from models.yolo import *
from libs.yolo.general import *
from libs.yolo.plots import *
from libs.yolo.torch_utils import *
from libs.utils import *
from libs.yolo.autoanchor import *
from libs.yolo.loss import *
from libs.yolo import test

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.cuda import amp
import torch.distributed as dist
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
        save_dir = Path(os.path.join(self.path, 'models'))
        weights_dir = os.path.join(save_dir, 'train')
        last = os.path.join(weights_dir, 'last.pt')
        best = os.path.join(weights_dir, 'best.pt')
        results_file = os.path.join(save_dir, 'results.txt')
        self.index_records(weights_dir)
        self.index_classes(self.path)
        total_batch_size = 16
        # epochs = 5
        # batch_size = 1

        data = check_file(os.path.join(weights_dir, 'data.yaml'))
        cfg = check_file(os.path.join(weights_dir, 'cfg.yaml'))
        hyp = check_file(os.path.join(weights_dir, 'hyp.yaml'))

        cuda = self.device.type != 'cpu'
        init_seeds(2 + rank)
        with open(data) as f:
            data_dict = yaml.load(f, Loader=yaml.SafeLoader)
        with open(hyp) as f:
            hyp_dict = yaml.load(f, Loader=yaml.SafeLoader)
        logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp_dict.items()))
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
        accumulate = max(round(nbs / total_batch_size), 1)
        hyp_dict['weight_decay'] *= total_batch_size * accumulate / nbs
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

        tb_writer = None
        wandb = None
        if rank in [-1, 0] and wandb and wandb.run is None:
            opt.hyp = hyp  # add hyperparameters
            wandb_run = wandb.init(config=opt, resume="allow",
                                   project='YOLOv5' if opt.project == 'runs/train' else Path(opt.project).stem,
                                   name=save_dir.stem,
                                   id=ckpt.get('wandb_id') if 'ckpt' in locals() else None)
        loggers = {'wandb' : wandb}

        if rank in [-1, 0]:
            ema.updates = start_epoch * nb // accumulate  # set EMA updates
            testloader = create_dataloader(test_path, imgsz_test, batch_size * 2, gs, False,
                                           hyp=hyp_dict, rect=True, rank=-1,
                                           pad=0.5, prefix=colorstr('val: '))[0]

            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))

            plots = True # as default
            if plots:
                plot_labels(labels, save_dir, loggers)
                if tb_writer:
                    tb_writer.add_histogram('classes', c, 0)

                # Anchors
                check_anchors(dataset, model=model, thr=hyp_dict['anchor_t'], imgsz=imgsz)

        # Model parameters
        hyp_dict['box'] *= 3. / nl  # scale to layers
        hyp_dict['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
        hyp_dict['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
        model.nc = nc  # attach number of classes to model
        model.hyp = hyp_dict  # attach hyperparameters to model
        model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
        model.class_weights = labels_to_class_weights(dataset.labels, nc).to(self.device) * nc  # attach class weights
        model.names = names

        # Start training
        t0 = time.time()
        nw = max(round(hyp_dict['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
        # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
        maps = np.zeros(nc)  # mAP per class
        results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
        scheduler.last_epoch = start_epoch - 1  # do not move
        scaler = amp.GradScaler(enabled=cuda)
        compute_loss = ComputeLoss(model)  # init loss class
        logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                    f'Using {dataloader.num_workers} dataloader workers\n'
                    f'Logging results to {save_dir}\n'
                    f'Starting training for {epochs} epochs...')

        for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
            model.train()

            # Update image weights (optional)
            image_weights = False
            if image_weights:
                # Generate indices
                if rank in [-1, 0]:
                    cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                    iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                    dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
                # Broadcast if DDP
                if rank != -1:
                    indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                    dist.broadcast(indices, 0)
                    if rank != 0:
                        dataset.indices = indices.cpu().numpy()

            mloss = torch.zeros(4, device=self.device)  # mean losses
            if rank != -1:
                dataloader.sampler.set_epoch(epoch)
            pbar = enumerate(dataloader)
            logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'targets', 'img_size'))
            if rank in [-1, 0]:
                pbar = tqdm(pbar, total=nb)  # progress bar
            optimizer.zero_grad()
            for i, (
            imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
                ni = i + nb * epoch  # number integrated batches (since train start)
                imgs = imgs.to(self.device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

                # Warmup
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                    accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                    for j, x in enumerate(optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(ni, xi,
                                            [hyp_dict['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [hyp_dict['warmup_momentum'], hyp_dict['momentum']])

                # Multi-scale
                multi_scale = False
                if multi_scale:
                    sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                    sf = sz / max(imgs.shape[2:])  # scale factor
                    if sf != 1:
                        ns = [math.ceil(x * sf / gs) * gs for x in
                              imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                        imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

                # Forward
                with amp.autocast(enabled=cuda):
                    pred = model(imgs)  # forward
                    loss, loss_items = compute_loss(pred, targets.to(self.device))  # loss scaled by batch_size
                    if rank != -1:
                        loss *= int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1 # gradient averaged between devices in DDP mode

                # Backward
                scaler.scale(loss).backward()

                # Optimize
                if ni % accumulate == 0:
                    scaler.step(optimizer)  # optimizer.step
                    scaler.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)

                # Print
                if rank in [-1, 0]:
                    mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                    mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                    s = ('%10s' * 2 + '%10.4g' * 6) % (
                        '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                    pbar.set_description(s)

                    # Plot
                    if plots and ni < 3:
                        f = save_dir / f'train_batch{ni}.jpg'  # filename
                        Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()
                        # if tb_writer:
                        #     tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                        #     tb_writer.add_graph(model, imgs)  # add model to tensorboard
                    elif plots and ni == 10 and wandb:
                        wandb.log({"Mosaics": [wandb.Image(str(x), caption=x.name) for x in save_dir.glob('train*.jpg')
                                               if x.exists()]}, commit=False)

                # end batch ------------------------------------------------------------------------------------------------
            # end epoch ----------------------------------------------------------------------------------------------------

            # Scheduler
            lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
            scheduler.step()

            # DDP process 0 or single-GPU
            if rank in [-1, 0]:
                # mAP
                if ema:
                    ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
                final_epoch = epoch + 1 == epochs
                if final_epoch:  # Calculate mAP
                    results, maps, times = test.test(data,
                                                     batch_size=batch_size * 2,
                                                     imgsz=imgsz_test,
                                                     model=ema.ema,
                                                     single_cls=False,
                                                     dataloader=testloader,
                                                     save_dir=save_dir,
                                                     verbose=nc < 50 and final_epoch,
                                                     plots=plots and final_epoch,
                                                     log_imgs=0,
                                                     compute_loss=compute_loss)

                # Write
                with open(results_file, 'a') as f:
                    f.write(s + '%10.4g' * 7 % results + '\n')  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
                bucket = ''
                if bucket:
                    os.system('gsutil cp %s gs://results/results.txt' % results_file)

                # Log
                tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                        'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                        'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                        'x/lr0', 'x/lr1', 'x/lr2']  # params
                for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                    if tb_writer:
                        tb_writer.add_scalar(tag, x, epoch)  # tensorboard
                    if wandb:
                        wandb.log({tag: x}, step=epoch, commit=tag == tags[-1])  # W&B

                # Update best mAP
                fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
                if fi > best_fitness:
                    best_fitness = fi

                # Save model
                save = final_epoch
                if save:
                    with open(results_file, 'r') as f:  # create checkpoint
                        ckpt = {'epoch': epoch,
                                'best_fitness': best_fitness,
                                'training_results': f.read(),
                                'model': ema.ema,
                                'optimizer': None if final_epoch else optimizer.state_dict(),
                                'wandb_id': wandb_run.id if wandb else None}

                    # Save last, best and delete
                    torch.save(ckpt, last)
                    if best_fitness == fi:
                        torch.save(ckpt, best)
                    del ckpt
            # end epoch ----------------------------------------------------------------------------------------------------
        # end training

    def index_classes(self, path):
        with open(os.path.join(path, 'data', 'classes.txt'), 'r') as f:
            self.classes = f.read().strip('\n').split('\n')

        with open(os.path.join(path, 'models', 'train', 'data.yaml'), 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
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
