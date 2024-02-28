import argparse
import os
# Last Change:  2024-02-28 15:46:13
import time
import math
import numpy as np
from collections import OrderedDict
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from tqdm import tqdm
import gc
import copy
import lmdb

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.autograd as autograd
import torch.nn.functional as F
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from transformers import BertTokenizer

import models
from datasets import *
from utils.amp import MaxClipGradScaler


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Search Recall Training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--output_dir', dest='output_dir', type=str, metavar='PATH', help='output dir')
    parser.add_argument('--item2fullpath', dest='item2fullpath', type=str, metavar='PATH', help='item2fullpath')
    parser.add_argument('--item2title', dest='item2title', type=str, metavar='PATH', help='item2title')
    parser.add_argument('--photo2fullpath', dest='photo2fullpath', type=str, metavar='PATH', help='photo2fullpath')
    parser.add_argument('--photo2text', dest='photo2text', type=str, metavar='PATH', help='photo2text')
    parser.add_argument('--live2fullpath', dest='live2fullpath', type=str, metavar='PATH', help='live2fullpath')
    parser.add_argument('--live2text', dest='live2text', type=str, metavar='PATH', help='live2text')
    parser.add_argument('--image_size', default=224, type=int, metavar='N', help='image_size')
    parser.add_argument('--embedding_size', default=128, type=int, metavar='N', help='embedding_size')
    parser.add_argument('--num_query_token', default=12, type=int, metavar='N', help='num_query_token in Q-Former')
    parser.add_argument('--qformer_cross_attention_freq', default=2, type=int, metavar='N', help='cross_attention_freq in Q-Former')
    parser.add_argument('--num_qformer_hidden_layer', default=4, type=int, metavar='N', help='num_qformer_hidden_layer in Bert QFormer layer')
    parser.add_argument('--num_fusion_hidden_layer', default=4, type=int, metavar='N', help='num_fusion_hidden_layer in Transformer fusion layer')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch_size', default=128, type=int,
                        metavar='N',
                        help='mini-batch size (default: 128), this is the '
                            'batch size of single GPU on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--checkpoints', default='./outputs/checkpoints/checkpoints', type=str, metavar='PATH', help='path to checkpoints (default: none)')
    parser.add_argument('--clip_length', default=8, type=int, metavar='N', help='clip length')
    parser.add_argument('--mixed_precision_training', action='store_true', help='mixed precision training')
    
    args = parser.parse_args()
    return args

def build_model(args):
    shark_model = models.__dict__['shark'](
        emb_dim = args.embedding_size,
        qformer_cross_attention_freq = args.qformer_cross_attention_freq,
        num_qformer_hidden_layer = args.num_qformer_hidden_layer,
        num_fusion_hidden_layer = args.num_fusion_hidden_layer, 
        mixed_precision_training = args.mixed_precision_training,
    )

    shark_model.cuda(args.local_rank)
    shark_model = torch.nn.parallel.DistributedDataParallel(module=shark_model, device_ids=[args.local_rank], broadcast_buffers=False, find_unused_parameters=True)
    
    if os.path.isfile(args.checkpoints):
        print("=> loading checkpoint '{}'".format(args.checkpoints))
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(args.local_rank)
        checkpoint = torch.load(args.checkpoints, map_location=loc)
        shark_model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.checkpoints, checkpoint['epoch']))
    else:
        print("=> checkpoint '{}' not exists...)".format(args.checkpoints))
        exit()
    
    return shark_model

def main():
    global global_step
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

    if not os.path.exists(args.output_dir) and rank == 0:
        os.makedirs(args.output_dir)

    # create model
    shark_model = build_model(args)
    print("=> backbone output size: text encoder: {} visual encoder: {}".format(shark_model.module.text_encoder.output_size, shark_model.module.visual_encoder.output_size))

    gc.collect()

    item_test_loader, item_test_sampler = get_loaders(args, val_type="items")
    photo_test_loader, photo_test_sampler = get_loaders(args, val_type="photos")
    live_test_loader, live_test_sampler = get_loaders(args, val_type="lives")
    evaluate([item_test_loader, photo_test_loader, live_test_loader], shark_model, args)

    dist.destroy_process_group()


def get_loaders(args, epoch=0, val_type=None):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = video_transforms.Compose([
                    video_transforms.Resize((args.image_size, args.image_size)),
                    video_transforms.CenterCrop(args.image_size),
                    volume_transforms.ClipToTensor2(),
                    normalize,
    ])


    if val_type == "items":
        test_dataset = SharkEmbItemInferDataset(args.item2fullpath, args.item2title,
                                    shuffle=False, transform=train_transform, return_img_id=True)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=(test_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=test_sampler, drop_last=False, collate_fn=CollateFnItem())
    if val_type == "photos":
        test_dataset = SharkEmbPhotoInferDataset(args.photo2fullpath, args.photo2text, args.clip_length,
                                    shuffle=False, transform=train_transform, return_img_id=True)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=(test_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=test_sampler, drop_last=False, collate_fn=CollateFnPhoto())
    if val_type == "lives":
        test_dataset = SharkEmbLiveInferDataset(args.live2fullpath, args.live2text, args.clip_length,
                                    shuffle=False, transform=train_transform, return_img_id=True)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=(test_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=test_sampler, drop_last=False, collate_fn=CollateFnLive())
    return test_loader, test_sampler

def evaluate(test_loader_list, shark_model, args):
    item_test_loader, photo_test_loader, live_test_loader = test_loader_list
    if hasattr(shark_model, 'module'):
        shark_model = shark_model.module.eval()
    else:
        shark_model = shark_model.eval()
    
    fw1 = open(os.path.join(args.output_dir, str(args.local_rank) + "_item"), 'w')
    for i, (item_titles, item_title_masks, item_imgs, itemids) in enumerate(tqdm(item_test_loader)):
        for k in item_titles:
            item_titles[k] = item_titles[k].cuda(args.local_rank)
        item_imgs = item_imgs.cuda(args.local_rank)
        
        b = item_imgs.shape[0]
        bt, c, h, w = item_imgs.shape
        item_imgs = item_imgs.view(b, -1, c, h, w)
        text_data = {'item': item_titles, 'photo': None, 'live': None}
        visual_data = {'item': item_imgs, 'photo': None, 'live': None}

        goods_emb, _, _ = shark_model.extract_features(text_data, visual_data)
        goods_emb = goods_emb.detach().cpu().numpy().astype('float32').tolist()

        for item, feat in zip(itemids, goods_emb):
            fw1.write("\t".join([item, ",".join([str(x) for x in feat])]) + "\n")
    fw1.flush()
    fw1.close()

    fw2 = open(os.path.join(args.output_dir, str(args.local_rank) + "_photo"), 'w')
    for i, (photo_texts, photo_text_masks, photo_imgs, photo_ids) in enumerate(tqdm(photo_test_loader)):
        for k in photo_texts:
            photo_texts[k] = photo_texts[k].cuda(args.local_rank)
        photo_imgs = photo_imgs.cuda(args.local_rank)
        
        b = len(photo_ids)
        bt, c, h, w = photo_imgs.shape
        photo_imgs = photo_imgs.view(b, -1, c, h, w)
        text_data = {'item': None, 'photo': photo_texts, 'live': None}
        visual_data = {'item': None, 'photo': photo_imgs, 'live': None}

        _, photo_emb, _ = shark_model.extract_features(text_data, visual_data)
        photo_emb = photo_emb.detach().cpu().numpy().astype('float32').tolist()

        for photo, feat in zip(photo_ids, photo_emb):
            fw2.write("\t".join([photo, ",".join([str(x) for x in feat])]) + "\n")
    fw2.flush()
    fw2.close()
    
    fw3 = open(os.path.join(args.output_dir, str(args.local_rank) + "_live"), 'w')
    for i, (live_texts, live_text_masks, live_imgs, live_ids, starttimes, endtimes) in enumerate(tqdm(live_test_loader)):
        for k in live_texts:
            live_texts[k] = live_texts[k].cuda(args.local_rank)
        live_imgs = live_imgs.cuda(args.local_rank)
        
        b = len(live_ids)
        bt, c, h, w = live_imgs.shape
        live_imgs = live_imgs.view(b, -1, c, h, w)
        text_data = {'item': None, 'photo': None, 'live': live_texts}
        visual_data = {'item': None, 'photo': None, 'live': live_imgs}

        _, _, live_emb = shark_model.extract_features(text_data, visual_data)
        live_emb = live_emb.detach().cpu().numpy().astype('float32').tolist()

        for live_id, starttime, endtime, feat in zip(live_ids, starttimes, endtimes, live_emb):
            fw3.write("\t".join([live_id, starttime, endtime, ",".join([str(x) for x in feat])]) + "\n")
    fw3.flush()
    fw3.close()

if __name__ == "__main__":
    main()
