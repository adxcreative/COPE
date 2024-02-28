import argparse
import os
import time
import math
import numpy as np
from collections import OrderedDict
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from tqdm import tqdm
import gc
import copy
import lmdb
import pickle
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.autograd as autograd
import torch.nn.functional as F
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

import models
from losses import *
from datasets import *
from utils.amp import MaxClipGradScaler
from transformers import BertTokenizer
import wandb

lr = 0
global_step = 0
wb = None

def get_args(description='Shark on Cross-domain Retrieval Task'):
    parser = argparse.ArgumentParser(description='Shark on Cross-domain Retrieval Task')
    parser.add_argument('--rank', type=int, default=0, help='rank')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--train_file', dest='train_file', type=str, metavar='PATH', help='train file')
    parser.add_argument('--output_dir', dest='output_dir', type=str, metavar='PATH', help='output dir')
    parser.add_argument('--spu2item_lmdb', dest='spu2item_lmdb', type=str, metavar='PATH', help='spu2item_lmdb')
    parser.add_argument('--spu2photo_lmdb', dest='spu2photo_lmdb', type=str, metavar='PATH', help='spu2photo_lmdb')
    parser.add_argument('--spu2live_lmdb', dest='spu2live_lmdb', type=str, metavar='PATH', help='spu2live_lmdb')
    parser.add_argument('--clip2live_lmdb', dest='clip2live_lmdb', type=str, metavar='PATH', help='clip2live_lmdb')
    parser.add_argument('--live2texts_lmdb', dest='live2texts_lmdb', type=str, metavar='PATH', help='live2texts_lmdb')
    parser.add_argument('--photo2frames_lmdb', dest='photo2frames_lmdb', type=str, metavar='PATH', help='photo2frames_lmdb')
    parser.add_argument('--photo2texts_lmdb', dest='photo2texts_lmdb', type=str, metavar='PATH', help='photo2texts_lmdb')
    parser.add_argument('--item_imgs_lmdb', dest='item_imgs_lmdb', type=str, metavar='PATH', help='item_imgs_lmdb')
    parser.add_argument('--item_titles_lmdb', dest='item_titles_lmdb', type=str, metavar='PATH', help='item_titles_lmdb')
    parser.add_argument('--photo2fullpath_lmdb', dest='photo2fullpath_lmdb', type=str, metavar='PATH', help='photo2fullpath_lmdb')
    parser.add_argument('--live2fullpath_lmdb', dest='live2fullpath_lmdb', type=str, metavar='PATH', help='live2fullpath_lmdb')

    parser.add_argument('--image_size', default=224, type=int, metavar='N', help='image_size')
    parser.add_argument('--embedding_size', default=128, type=int, metavar='N', help='embedding_size')
    parser.add_argument('--num_query_token', default=12, type=int, metavar='N', help='num_query_token in Q-Former')
    parser.add_argument('--qformer_cross_attention_freq', default=1, type=int, metavar='N', help='cross_attention_freq in Q-Former')
    parser.add_argument('--num_qformer_hidden_layer', default=4, type=int, metavar='N', help='num_qformer_hidden_layer in Bert QFormer layer')
    parser.add_argument('--num_fusion_hidden_layer', default=4, type=int, metavar='N', help='num_fusion_hidden_layer in Transformer fusion layer')

    parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--warm_up_iters', default=1000, type=int, metavar='N', help='number of warm up iters to run')
    parser.add_argument('--max_iters', default=10000, type=int, metavar='N', help='number of lr max iters to run')
    parser.add_argument('--lr_freq', default=100, type=int, metavar='N', help='lr change avg iters')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch_size', default=128, type=int,
                        metavar='N',
                        help='mini-batch size (default: 128), this is the '
                            'batch size of single GPU on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning_rate', default=0.1, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--text_lr_factor', default=0.01, type=float, metavar='text_lr_factor', help='text_lr_factor', dest='text_lr_factor')
    parser.add_argument('--visual_lr_factor', default=0.0001, type=float, metavar='visual_lr_factor', help='visual_lr_factor', dest='visual_lr_factor')
    parser.add_argument('--fusion_lr_factor', default=1.0, type=float, metavar='fusion_lr_factor', help='fusion_lr_factor', dest='fusion_lr_factor')
    parser.add_argument('--other_lr_factor', default=1.0, type=float, metavar='other_lr_factor', help='other_lr_factor', dest='other_lr_factor')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight_decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print_freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', default='', type=str, metavar='PATH', help='path to pre-trained model')
    parser.add_argument('--mixed_precision_training', action='store_true', help='mixed precision training')
    parser.add_argument('--finetune', action='store_true', help='finetune')
    parser.add_argument('--use_item_inner', default=0.0, type=float, help='use_item_inner')
    parser.add_argument('--use_photo_inner', default=0.0, type=float, help='use_photo_inner')
    parser.add_argument('--use_live_inner', default=0.0, type=float, help='use_live_inner')
    parser.add_argument('--use_item_photo_match', default=0.0, type=float, help='use_item_photo_match')
    parser.add_argument('--use_item_live_match', default=0.0, type=float, help='use_item_live_match')
    parser.add_argument('--use_item_text_item_visual_cross', default=0.0, type=float, help='use_item_text_item_visual_cross')
    parser.add_argument('--use_photo_text_photo_visual_cross', default=0.0, type=float, help='use_photo_text_photo_visual_cross')
    parser.add_argument('--use_live_text_live_visual_cross', default=0.0, type=float, help='use_live_text_live_visual_cross')
    parser.add_argument('--use_item_photo_cross', default=0.0, type=float, help='use_item_photo_cross')
    parser.add_argument('--use_item_text_photo_cross', default=0.0, type=float, help='use_item_text_photo_cross')
    parser.add_argument('--use_item_text_photo_text_cross', default=0.0, type=float, help='use_item_text_photo_text_cross')
    parser.add_argument('--use_item_visual_photo_cross', default=0.0, type=float, help='use_item_visual_photo_cross')
    parser.add_argument('--use_item_visual_photo_visual_cross', default=0.0, type=float, help='use_item_visual_photo_visual_cross')
    parser.add_argument('--use_item_text_live_cross', default=0.0, type=float, help='use_item_text_live_cross')
    parser.add_argument('--use_item_text_live_text_cross', default=0.0, type=float, help='use_item_text_live_text_cross')
    parser.add_argument('--use_item_visual_live_cross', default=0.0, type=float, help='use_item_visual_live_cross')
    parser.add_argument('--use_item_visual_live_visual_cross', default=0.0, type=float, help='use_item_visual_live_visual_cross')
    parser.add_argument('--use_item_live_cross', default=0.0, type=float, help='use_item_live_cross')
    parser.add_argument('--use_item_photo_text_cross', default=0.0, type=float, help='use_item_photo_text_cross')
    parser.add_argument('--use_item_live_text_cross', default=0.0, type=float, help='use_item_live_text_cross')
    parser.add_argument('--use_item_photo_visual_cross', default=0.0, type=float, help='use_item_photo_visual_cross')
    parser.add_argument('--use_item_live_visual_cross', default=0.0, type=float, help='use_item_live_visual_cross')
    parser.add_argument('--use_photo_live_cross', default=0.0, type=float, help='use_photo_live_cross')
    parser.add_argument('--use_photo_text_live_text_cross', default=0.0, type=float, help='use_photo_text_live_text_cross')
    parser.add_argument('--use_photo_text_live_cross', default=0.0, type=float, help='use_photo_text_live_cross')
    parser.add_argument('--use_photo_visual_live_cross', default=0.0, type=float, help='use_photo_visual_live_cross')
    parser.add_argument('--use_photo_live_visual_cross', default=0.0, type=float, help='use_photo_live_visual_cross')
    parser.add_argument('--use_photo_live_text_cross', default=0.0, type=float, help='use_photo_live_text_cross')
    parser.add_argument('--use_photo_visual_live_visual_cross', default=0.0, type=float, help='use_photo_visual_live_visual_cross')
    parser.add_argument('--use_spu_classify', default=0.0, type=float, help='use_spu_classify')
    parser.add_argument('--is_xbm', action='store_true', help='is_xbm')
    parser.add_argument('--clip_length', default=0, type=int, metavar='N', help='clip length')
    
    parser.add_argument('--spu_num_classes', default=-1, type=int, help='spu_num_classes')
    parser.add_argument('--use_wandb_recorder', action='store_true', help='use wandb or not')
    parser.add_argument('--use_ogm_ge_gradient_modualtion', action='store_true', help='use gradient modualtion or not')
    args = parser.parse_args()
    return args

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).cpu().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_checkpoint(state, checkpoints_path):
    filename = os.path.join(checkpoints_path, 'checkpoint_{}.pth.tar'.format(state['epoch']))
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def build_model(args):
    shark_model = models.__dict__['shark'](
        emb_dim = args.embedding_size,
        qformer_cross_attention_freq = args.qformer_cross_attention_freq,
        num_qformer_hidden_layer = args.num_qformer_hidden_layer,
        num_fusion_hidden_layer = args.num_fusion_hidden_layer, 
        mixed_precision_training = args.mixed_precision_training,
    )

    margin_loss = CombinedMarginLoss(64, 1.0, 0.0, 0.4, 0)
    spu_cls_model = PartialFC_V2(margin_loss=margin_loss, embedding_size=args.embedding_size, num_classes=args.spu_num_classes, sample_rate=1.0, fp16=args.mixed_precision_training)
    
    # load pretrained model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.pretrained))
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(args.local_rank)
        checkpoint = torch.load(args.pretrained, map_location=loc)

        model_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            if k.startswith('module.'):
                k = k[7:]
            model_state_dict[k] = v
        shark_model.load_state_dict(model_state_dict)
    else:
        print("=> creating model from scrach ...")

    shark_model.cuda(args.local_rank)
    shark_model = torch.nn.parallel.DistributedDataParallel(module=shark_model, device_ids=[args.local_rank], broadcast_buffers=False, find_unused_parameters=True)
    shark_model.train()

    spu_cls_model.cuda(args.local_rank)
    spu_cls_model.train()

    if not args.finetune:
        if hasattr(shark_model, 'module'):
            for name, param in shark_model.module.text_encoder.named_parameters():
                param.requires_grad = False
            shark_model.module.text_encoder.eval() # freeze text encoder
            for name, param in shark_model.module.visual_encoder.named_parameters():
                param.requires_grad = False
            shark_model.module.visual_encoder.eval() # freeze visual encoder
        else:
            for name, param in shark_model.text_encoder.named_parameters():
                param.requires_grad = False
            shark_model.text_encoder.eval() # freeze text encoder
            for name, param in shark_model.visual_encoder.named_parameters():
                param.requires_grad = False
            shark_model.visual_encoder.eval() # freeze visual encoder

    return shark_model, spu_cls_model

def build_optimizer(args, shark_model, spu_cls_model):
    text_lr_factor = args.text_lr_factor
    visual_lr_factor = args.visual_lr_factor
    fusion_lr_factor = args.fusion_lr_factor
    other_lr_factor = args.other_lr_factor
    print("=> text/visual/cls lr factor: {}/{}".format(text_lr_factor, visual_lr_factor))
    text_encoder_param_list = []
    visual_encoder_param_list = []
    fusion_module_param_list = []
    other_module_param_list = []
    model_params = list(shark_model.named_parameters())
    for n, p in model_params:
        if 'text_encoder' in n:
            text_encoder_param_list.append(p)
        elif 'visual_encoder' in n:
            visual_encoder_param_list.append(p)
        elif 'fusion_model' in n or 'visual_proj_layer' in n or 'fusion_proj_layer' or 'text_proj_layer'  in n or 'query_tokens' in n:
            fusion_module_param_list.append(p)
        else:
            other_module_param_list.append(p)

    optimizer_grouped_parameters = [
        {'params': text_encoder_param_list, 'lr': args.lr * text_lr_factor},
        {'params': visual_encoder_param_list, 'lr': args.lr * visual_lr_factor},
        {'params': fusion_module_param_list, 'lr': args.lr * fusion_lr_factor},
        {'params': other_module_param_list, 'lr': args.lr * other_lr_factor},
        {'params': spu_cls_model.parameters(), 'lr': args.lr * other_lr_factor},
    ]

    optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters, lr=args.lr)

    if args.warm_up_iters != 0:
        warm_up_with_cosine_lr = lambda epoch: epoch / args.warm_up_iters if epoch <= args.warm_up_iters else 0.5 * ( math.cos((epoch - args.warm_up_iters) /(args.max_iters - args.warm_up_iters) * math.pi) + 1)
    else:
        warm_up_with_cosine_lr = lambda epoch: 0.5 * ( math.cos((epoch - args.warm_up_iters) /(args.max_iters - args.warm_up_iters) * math.pi) + 1)        

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)

    return optimizer, scheduler

def get_loaders(args, epoch=0):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = video_transforms.Compose([
                        video_transforms.RandomResizedCrop(args.image_size, scale=(0.64, 1.0)),
                        video_transforms.RandomHorizontalFlip(),
                        volume_transforms.ClipToTensor2(),
                        normalize,
    ])

    train_file = args.train_file
    
    if args.use_item_inner or args.use_photo_inner or args.use_live_inner:
        train_dataset = SharkEmbSpuNewLmdbDataset(train_file, args.clip_length,
                                                args.spu2item_lmdb, args.spu2photo_lmdb, args.spu2live_lmdb,
                                                args.clip2live_lmdb, args.live2texts_lmdb, args.photo2frames_lmdb, args.photo2texts_lmdb,
                                                args.item_imgs_lmdb, args.item_titles_lmdb, args.photo2fullpath_lmdb, args.live2fullpath_lmdb, 
                                                    shuffle=False, transform=train_transform, use_positive_sample=True)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True, collate_fn=ThreePartPositiveSampleCollateFn())   
    else:
        train_dataset = SharkEmbSpuNewLmdbDataset(train_file, args.clip_length,
                                            args.spu2item_lmdb, args.spu2photo_lmdb, args.spu2live_lmdb,
                                            args.clip2live_lmdb, args.live2texts_lmdb, args.photo2frames_lmdb, args.photo2texts_lmdb,
                                            args.item_imgs_lmdb, args.item_titles_lmdb, args.photo2fullpath_lmdb, args.live2fullpath_lmdb, 
                                                shuffle=False, transform=train_transform)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True, collate_fn=ThreePartCollateFn())
    return train_loader, train_sampler

def main():
    global global_step, wb
    args = get_args()
    torch.backends.cudnn.benchmark = True

    try:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        dist_url = "tcp://{}:{}".format(os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"])
    except KeyError:
        world_size = 1
        rank = 0
        dist_url = "tcp://127.0.0.1:12584"
    
    checkpoints_path = os.path.join(args.output_dir, 'checkpoints')
    logs_path = os.path.join(args.output_dir, 'logs', time.strftime('%Y-%m-%d-%H-%M-%S'))
    if not os.path.exists(checkpoints_path) and rank == 0:
        os.makedirs(checkpoints_path)
    if not os.path.exists(logs_path) and rank == 0:
        os.makedirs(logs_path)

    args.world_size = world_size
    args.rank = rank
    args.dist_url = dist_url
    print("=> world size:", world_size)
    print("=> rank:", rank)
    print("=> dist_url:", dist_url)

    dist.init_process_group(backend='nccl', init_method=dist_url, rank=rank, world_size=world_size)
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)

    if local_rank == 0:
        print("=> args:", args)

    if args.use_wandb_recorder and rank == 0:
        wb = wandb.init(project="shark", name='test', config=args)
        # wb = None
    else:
        wb = None
        
    if not (args.use_item_text_item_visual_cross or args.use_item_inner or args.use_photo_inner or args.use_live_inner or args.use_item_photo_cross or args.use_item_live_cross or args.use_photo_live_cross or args.use_spu_classify or args.use_item_photo_match or args.use_item_live_match):
        print("at least point one loss function")
        exit(0)

    # create model
    spus = list()
    with open(args.train_file, "r") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            spu = line.strip()
            spus.append(spu)
    spu2label = dict()
    for idx, spu in enumerate(spus):
        spu2label[spu] = idx
    args.spu_num_classes = len(spu2label)

    shark_model, spu_cls_model = build_model(args)
    print("=> backbone output size: text encoder: {} visual encoder: {}".format(shark_model.module.text_encoder.output_size, shark_model.module.visual_encoder.output_size))

    gc.collect()

    # create optimizer and scheduler
    optimizer, scheduler = build_optimizer(args, shark_model, spu_cls_model)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(local_rank)
            checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = int(str(checkpoint['epoch']).split('_')[0])
            global_step = checkpoint['global_step']
            shark_model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    #这里gc是为了清空缓存在cpu里的模型
    gc.collect()

    # Data loading
    train_loader, train_sampler = get_loaders(args)

    grad_amp = MaxClipGradScaler(args.batch_size, 128*args.batch_size, growth_interval=100) if args.mixed_precision_training else None

    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)

        print("is_xbm: ", args.is_xbm)
        # train for one epoch
        train_one_epoch(args, epoch, train_loader, spu2label, shark_model, spu_cls_model, optimizer, scheduler, grad_amp)
        
        if rank == 0 and epoch % 5 == 0:
            save_checkpoint({
                'epoch': epoch,
                'global_step': global_step,
                'state_dict': shark_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, checkpoints_path)

    dist.destroy_process_group()

def train_one_epoch(args, epoch, train_loader, spu2label, shark_model, spu_cls_model, optimizer, scheduler, grad_amp):
    global lr, global_step
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    avg_loss = AverageMeter('Loss', ':.4e')
    
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, avg_loss], prefix="Epoch: [{}]".format(epoch))

    end = time.time()
    
    n = len(train_loader)
    loss_total = 0
    
    for i, content in enumerate(train_loader):
        if args.use_item_inner or args.use_photo_inner or args.use_live_inner:
            pass
        else:
            item_titles, item_titles_mask, item_imgs, item_ids, live_texts, live_texts_mask, live_frames, lives, photo_texts, photo_texts_mask, photo_imgs, photo_ids, spu_ids = content

        for k in item_titles:
            item_titles[k] = item_titles[k].cuda(args.local_rank)
        item_titles_mask = item_titles_mask.cuda(args.local_rank)
        item_imgs = item_imgs.cuda(args.local_rank)
        for k in photo_texts:
            photo_texts[k] = photo_texts[k].cuda(args.local_rank)
        photo_texts_mask = photo_texts_mask.cuda(args.local_rank)
        photo_imgs = photo_imgs.cuda(args.local_rank)
        for k in live_texts:
            live_texts[k] = live_texts[k].cuda(args.local_rank)
        live_texts_mask = live_texts_mask.cuda(args.local_rank)
        live_frames = live_frames.cuda(args.local_rank)

        b = item_imgs.shape[0]
        bt, c, h, w = photo_imgs.shape
        bt_2, c_2, h_2, w_2 = live_frames.shape
        item_imgs = item_imgs.view(b, -1, c, h, w)
        photo_imgs = photo_imgs.view(b, -1, c, h, w)
        live_frames = live_frames.view(b, -1, c_2, h_2, w_2)
        
        # compute output
        text_data = {'item': item_titles, 'photo': photo_texts, 'live': live_texts}
        visual_data = {'item': item_imgs, 'photo': photo_imgs, 'live': live_frames}
        loss_dict, emb_dict = shark_model(text_data, visual_data)

        if args.use_spu_classify:
            spu_labels = [spu2label[spu] for spu in spu_ids]
            classfy_labels = torch.from_numpy(np.array(spu_labels)).cuda(args.local_rank)
            loss_dict['loss_classfy_item'], acc_dict['spu_cls_item'] = spu_cls_model(emb_dict['item_emb_logit'], classfy_labels)
            loss_dict['loss_classfy_photo'], acc_dict['spu_cls_photo'] = spu_cls_model(emb_dict['photo_emb_logit'], classfy_labels)
            loss_dict['loss_classfy_live'], acc_dict['spu_cls_live'] = spu_cls_model(emb_dict['live_emb_logit'], classfy_labels)
            loss_dict['loss_classfy'] = loss_dict['loss_classfy_item'] + loss_dict['loss_classfy_photo'] + loss_dict['loss_classfy_live']
            loss_total += args.use_spu_classify * loss_dict['loss_classfy']
        # measure data loading time
        data_time.update(time.time() - end)
        global_step += 1

        if args.use_item_photo_match:
            loss_total += args.use_item_photo_match * loss_dict['loss_ipm']
        if args.use_item_live_match:
            loss_total += args.use_item_live_match * loss_dict['loss_ilm']
        if args.use_item_text_item_visual_cross:
            loss_total += args.use_item_text_item_visual_cross * loss_dict['loss_itivc']
        if args.use_photo_text_photo_visual_cross:
            loss_total += args.use_photo_text_photo_visual_cross * loss_dict['loss_ptpvc']
        if args.use_live_text_live_visual_cross:
            loss_total += args.use_live_text_live_visual_cross * loss_dict['loss_ltlvc']
        if args.use_item_photo_cross:
            loss_total += args.use_item_photo_cross * loss_dict['loss_ipc']
        if args.use_item_live_cross:
            loss_total += args.use_item_live_cross * loss_dict['loss_ilc']
        if args.use_photo_live_cross:
            loss_total += args.use_photo_live_cross * loss_dict['loss_plc']
        if args.use_item_text_photo_cross:
            loss_total += args.use_item_text_photo_cross * loss_dict['loss_itpc']
        if args.use_item_text_photo_text_cross:
            loss_total += args.use_item_text_photo_text_cross * loss_dict['loss_itptc']
        if args.use_item_visual_photo_cross:
            loss_total += args.use_item_visual_photo_cross * loss_dict['loss_ivpc']
        if args.use_item_visual_photo_visual_cross:
            loss_total += args.use_item_visual_photo_visual_cross * loss_dict['loss_ivpvc']
        if args.use_item_text_live_cross:
            loss_total += args.use_item_text_live_cross * loss_dict['loss_itlc']
        if args.use_item_text_live_text_cross:
            loss_total += args.use_item_text_live_text_cross * loss_dict['loss_itltc']
        if args.use_item_visual_live_cross:
            loss_total += args.use_item_visual_live_cross * loss_dict['loss_ivlc']
        if args.use_item_visual_live_visual_cross:
            loss_total += args.use_item_visual_live_visual_cross * loss_dict['loss_ivlvc']
        if args.use_item_live_visual_cross:
            loss_total += args.use_item_live_visual_cross * loss_dict['loss_ilvc']
        if args.use_item_live_text_cross:
            loss_total += args.use_item_live_text_cross * loss_dict['loss_iltc']
        if args.use_item_photo_visual_cross:
            loss_total += args.use_item_photo_visual_cross * loss_dict['loss_ipvc']
        if args.use_item_photo_text_cross:
            loss_total += args.use_item_photo_text_cross * loss_dict['loss_iptc']
        if args.use_photo_text_live_text_cross:
            loss_total += args.use_photo_text_live_text_cross * loss_dict['loss_ptltc']
        if args.use_photo_visual_live_visual_cross:
            loss_total += args.use_photo_visual_live_visual_cross * loss_dict['loss_pvlvc']
        if args.use_photo_live_text_cross:
            loss_total += args.use_photo_live_text_cross * loss_dict['loss_pltc']
        if args.use_photo_live_visual_cross:
            loss_total += args.use_photo_live_visual_cross * loss_dict['loss_plvc']
        if args.use_photo_text_live_cross:
            loss_total += args.use_photo_text_live_cross * loss_dict['loss_ptlc']
        if args.use_photo_visual_live_cross:
            loss_total += args.use_photo_visual_live_cross * loss_dict['loss_pvlc']

        # compute gradient and do SGD step
        if args.mixed_precision_training:
            grad_amp.scale(loss_total).backward()
            grad_amp.unscale_(optimizer)
            clip_grad_norm_(shark_model.parameters(), max_norm=5, norm_type=2)
            grad_amp.step(optimizer)
            grad_amp.update()
        else:
            loss_total.backward()
            clip_grad_norm_(shark_model.parameters(), max_norm=5, norm_type=2)
            optimizer.step()
            
        optimizer.zero_grad()

        if global_step % args.lr_freq == 0:
            scheduler.step()
    
        # measure accuracy and record loss
        avg_loss.update(loss_total.item(), 1)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.local_rank == 0:
            cur_time = time.strftime('%Y-%m-%d %H:%M:%S')
            print('%s\tOtherLR: %.6f\tGlobalStep: %8d' % (cur_time, optimizer.state_dict()['param_groups'][2]['lr'], global_step), flush=True)
            progress.display(i)

        # Record logs in wandb
        if wb is not None:
            wb.log({'epoch': epoch, 'global_step': global_step, 'lr/text_lr': optimizer.state_dict()['param_groups'][0]['lr'],
                    'lr/visual_lr': optimizer.state_dict()['param_groups'][1]['lr'], 'lr/other_lr': optimizer.state_dict()['param_groups'][2]['lr'],
                    'losses/loss': loss_total.item()})
            if args.use_item_photo_match:
                wb.log({'losses/item_photo_matching_loss': loss_dict['loss_ipm'].item()})
                wb.log({'matching_acc/item_photo_matching_acc': acc_dict['acc_ipm'].item()})
            if args.use_item_live_match:
                wb.log({'losses/item_live_matching_loss': loss_dict['loss_ilm'].item()})
                wb.log({'matching_acc/item_live_matching_acc': acc_dict['acc_ilm'].item()})
            if args.use_item_text_item_visual_cross:
                wb.log({'losses/item_text_item_visual_cross': loss_dict['loss_itivc'].item()})
            if args.use_photo_text_photo_visual_cross:
                wb.log({'losses/photo_text_photo_visual_cross': loss_dict['loss_ptpvc'].item()})
            if args.use_live_text_live_visual_cross:
                wb.log({'losses/live_text_live_visual_cross': loss_dict['loss_ltlvc'].item()})
            if args.use_item_photo_cross:
                wb.log({'losses/item_photo_cross_loss': loss_dict['loss_ipc'].item()})
            if args.use_item_live_cross:
                wb.log({'losses/item_live_cross_loss': loss_dict['loss_ilc'].item()})
            if args.use_photo_live_cross:
                wb.log({'losses/photo_live_cross_loss': loss_dict['loss_plc'].item()})
            if args.use_item_text_photo_cross:
                wb.log({'losses/item_text_photo_cross_loss': loss_dict['loss_itpc'].item()})
            if args.use_item_text_photo_text_cross:
                wb.log({'losses/item_text_photo_text_cross_loss': loss_dict['loss_itptc'].item()})
            if args.use_item_visual_photo_cross:
                wb.log({'losses/item_visual_photo_cross_loss': loss_dict['loss_ivpc'].item()})
            if args.use_item_visual_photo_visual_cross:
                wb.log({'losses/item_visual_photo_visual_cross_loss': loss_dict['loss_ivpvc'].item()})
            if args.use_item_text_live_cross:
                wb.log({'losses/item_text_live_cross_loss': loss_dict['loss_itlc'].item()})
            if args.use_item_text_live_text_cross:
                wb.log({'losses/item_text_live_text_cross_loss': loss_dict['loss_itltc'].item()})
            if args.use_item_live_text_cross:
                wb.log({'losses/item_live_text_cross_loss': loss_dict['loss_iltc'].item()})
            if args.use_item_live_visual_cross:
                wb.log({'losses/item_live_visual_cross_loss': loss_dict['loss_ilvc'].item()})
            if args.use_item_photo_text_cross:
                wb.log({'losses/item_photo_text_cross_loss': loss_dict['loss_iptc'].item()})
            if args.use_item_photo_visual_cross:
                wb.log({'losses/item_photo_visual_cross_loss': loss_dict['loss_ipvc'].item()})
            if args.use_item_visual_live_cross:
                wb.log({'losses/item_visual_live_cross_loss': loss_dict['loss_ivlc'].item()})
            if args.use_item_visual_live_visual_cross:
                wb.log({'losses/item_visual_live_visual_cross_loss': loss_dict['loss_ivlvc'].item()})
            if args.use_photo_text_live_text_cross:
                wb.log({'losses/photo_text_live_text_cross_loss': loss_dict['loss_ptltc'].item()})
            if args.use_photo_visual_live_visual_cross:
                wb.log({'losses/photo_visual_live_visual_cross_loss': loss_dict['loss_pvlvc'].item()})
            if args.use_photo_visual_live_cross:
                wb.log({'losses/photo_visual_live_cross_loss': loss_dict['loss_pvlc'].item()})
            if args.use_photo_text_live_cross:
                wb.log({'losses/photo_text_live_cross_loss': loss_dict['loss_ptlc'].item()})
            if args.use_photo_live_text_cross:
                wb.log({'losses/photo_live_text_cross_loss': loss_dict['loss_pltc'].item()})
            if args.use_photo_live_visual_cross:
                wb.log({'losses/photo_live_visual_cross_loss': loss_dict['loss_plvc'].item()})
            if args.use_spu_classify:
                wb.log({'losses/loss_classfy': loss_dict['loss_classfy'].item()})                    
                wb.log({'losses/loss_classfy_goods': loss_dict['loss_classfy_item'].item()})                    
                wb.log({'losses/loss_classfy_photo': loss_dict['loss_classfy_photo'].item()})
                wb.log({'losses/loss_classfy_live': loss_dict['loss_classfy_live'].item()})
                wb.log({'spu_classfy_acc/spu_cls_item': acc_dict['spu_cls_item'].item()})
                wb.log({'spu_classfy_acc/spu_cls_photo': acc_dict['spu_cls_photo'].item()})
                wb.log({'spu_classfy_acc/spu_cls_live': acc_dict['spu_cls_live'].item()})
            if args.use_ogm_ge_gradient_modualtion:
                wb.log({'gradient_modualtion_coeff/coeff_t': coeff_dict['coeff_t']})
                wb.log({'gradient_modualtion_coeff/coeff_v': coeff_dict['coeff_v']})
                wb.log({'gradient_modualtion_coeff/loss_t': coeff_dict['loss_t']})
                wb.log({'gradient_modualtion_coeff/loss_v': coeff_dict['loss_v']})

        loss_total = 0


if __name__ == "__main__":
    main()
