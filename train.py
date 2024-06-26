import datetime
import os
import argparse
from functools import partial

from generate_txt_in_imagesets import generate_txt_in_imagesets
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from module import Unet, Unet_Attn
from callbacks import EvalCallback, LossHistory
from datasets import UnetDataset, unet_dataset_collate
from utils import download_weights, seed_everything, show_config, worker_init_fn
from train_utils import fit_one_epoch
from losses_and_schedulers import get_lr_scheduler, set_optimizer_lr, weights_init

def parse_args():
    parser = argparse.ArgumentParser(description="Train a UNet model with CBSA module for semantic segmentation.")
    parser.add_argument('--seed', type=int, default=11, help='Random seed for reproducibility')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='Use sync batch normalization')
    parser.add_argument('--fp16', action='store_true', default=False, help='Use mixed precision training')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--backbone', type=str, default='resnet50', help='Backbone model: vgg or resnet50')
    parser.add_argument('--pretrained', action='store_true', default=False, help='Use pretrained weights')
    parser.add_argument('--model_path', type=str, default='', help='Path to pretrained model')
    parser.add_argument('--input_shape', type=int, nargs='+', default=[256, 256], help='Input shape')
    parser.add_argument('--init_epoch', type=int, default=0, help='Initial epoch')
    parser.add_argument('--freeze_epoch', type=int, default=50, help='Epoch to stop freezing backbone')
    parser.add_argument('--freeze_batch_size', type=int, default=4, help='Batch size during freezing')
    parser.add_argument('--unfreeze_epoch', type=int, default=300, help='Total number of epochs')
    parser.add_argument('--unfreeze_batch_size', type=int, default=4, help='Batch size during unfreezing')
    parser.add_argument('--freeze_train', action='store_true', default=False, help='Freeze training')
    parser.add_argument('--init_lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate')
    parser.add_argument('--optimizer_type', type=str, default='adam', help='Optimizer type')
    parser.add_argument('--momentum', type=float, default=0.85, help='Momentum')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--lr_decay_type', type=str, default='cos', help='Learning rate decay type')
    parser.add_argument('--save_period', type=int, default=300, help='Save period')
    parser.add_argument('--save_dir', type=str, default='logs/unet-res', help='Directory to save logs and models')
    parser.add_argument('--eval_flag', action='store_true', default=True, help='Evaluate during training')
    parser.add_argument('--eval_period', type=int, default=1, help='Evaluation period')
    parser.add_argument('--vocdevkit_path', type=str, default='VOCdevkit', help='Path to VOCdevkit dataset')
    parser.add_argument('--dice_loss', action='store_true', default=True, help='Use Dice loss')
    parser.add_argument('--focal_loss', action='store_true', default=False, help='Use Focal loss')
    parser.add_argument('--cuda', action='store_true', default=True, help='Use CUDA')
    parser.add_argument('--distributed', action='store_true', default=False, help='Use distributed training')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    cls_weights = np.ones([args.num_classes], np.float32)
    ngpus_per_node = torch.cuda.device_count()
    generate_txt_in_imagesets()
    if args.distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        rank = 0

    if args.pretrained:
        if args.distributed:
            if local_rank == 0:
                download_weights(args.backbone)
            dist.barrier()
        else:
            download_weights(args.backbone)

    # model = Unet_Attn(num_classes=args.num_classes, pretrained=args.pretrained, backbone=args.backbone).train()
    model = Unet(num_classes=args.num_classes, pretrained=args.pretrained, backbone=args.backbone).train()
    if not args.pretrained:
        weights_init(model)
    if args.model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(args.model_path))
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))

    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(args.save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=args.input_shape)
    else:
        loss_history = None

    if args.fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    if args.sync_bn and ngpus_per_node > 1 and args.distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif args.sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if args.cuda:
        if args.distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    with open(os.path.join(args.vocdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(args.vocdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        show_config(
            num_classes=args.num_classes, backbone=args.backbone, model_path=args.model_path, input_shape=args.input_shape,
            Init_Epoch=args.init_epoch, Freeze_Epoch=args.freeze_epoch, UnFreeze_Epoch=args.unfreeze_epoch, Freeze_batch_size=args.freeze_batch_size, Unfreeze_batch_size=args.unfreeze_batch_size, Freeze_Train=args.freeze_train,
            Init_lr=args.init_lr, Min_lr=args.min_lr, optimizer_type=args.optimizer_type, momentum=args.momentum, lr_decay_type=args.lr_decay_type,
            save_period=args.save_period, save_dir=args.save_dir, num_workers=args.num_workers, num_train=num_train, num_val=num_val
        )

    if True:
        UnFreeze_flag = False
        if args.freeze_train:
            model.freeze_backbone()

        batch_size = args.freeze_batch_size if args.freeze_train else args.unfreeze_batch_size

        nbs = 16
        lr_limit_max = 1e-4 if args.optimizer_type == 'adam' else 1e-1
        lr_limit_min = 1e-4 if args.optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * args.init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * args.min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(args.momentum, 0.999), weight_decay=args.weight_decay),
            'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)
        }[args.optimizer_type]

        lr_scheduler_func = get_lr_scheduler(args.lr_decay_type, Init_lr_fit, Min_lr_fit, args.unfreeze_epoch)

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("Can't get enough Dataset")

        train_dataset = UnetDataset(train_lines, args.input_shape, args.num_classes, True, args.vocdevkit_path)
        val_dataset = UnetDataset(val_lines, args.input_shape, args.num_classes, False, args.vocdevkit_path)

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size = batch_size // ngpus_per_node
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True

        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=args.num_workers, pin_memory=True,
                         drop_last=True, collate_fn=unet_dataset_collate, sampler=train_sampler,
                         worker_init_fn=partial(worker_init_fn, rank=rank, seed=args.seed))
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=args.num_workers, pin_memory=True,
                             drop_last=True, collate_fn=unet_dataset_collate, sampler=val_sampler,
                             worker_init_fn=partial(worker_init_fn, rank=rank, seed=args.seed))

        if local_rank == 0:
            eval_callback = EvalCallback(model, args.input_shape, args.num_classes, val_lines, args.vocdevkit_path, log_dir, args.cuda,
                                         eval_flag=args.eval_flag, period=args.eval_period)
        else:
            eval_callback = None

        for epoch in range(args.init_epoch, args.unfreeze_epoch):
            if epoch >= args.freeze_epoch and not UnFreeze_flag and args.freeze_train:
                batch_size = args.unfreeze_batch_size

                Init_lr_fit = min(max(batch_size / nbs * args.init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * args.min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                lr_scheduler_func = get_lr_scheduler(args.lr_decay_type, Init_lr_fit, Min_lr_fit, args.unfreeze_epoch)

                model.unfreeze_backbone()

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("Can't get enough Dataset.")

                if args.distributed:
                    batch_size = batch_size // ngpus_per_node

                gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=args.num_workers, pin_memory=True,
                                 drop_last=True, collate_fn=unet_dataset_collate, sampler=train_sampler,
                                 worker_init_fn=partial(worker_init_fn, rank=rank, seed=args.seed))
                gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=args.num_workers, pin_memory=True,
                                     drop_last=True, collate_fn=unet_dataset_collate, sampler=val_sampler,
                                     worker_init_fn=partial(worker_init_fn, rank=rank, seed=args.seed))

                UnFreeze_flag = True

            if args.distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, args.unfreeze_epoch, args.cuda, args.dice_loss, args.focal_loss, cls_weights, args.num_classes, args.fp16, scaler, args.save_period, args.save_dir, local_rank)

            if args.distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()
