import copy
import os
import sys
# Last Change:  2024-02-28 16:46:14
import random
from PIL import Image
from multiprocessing import Manager
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import math
from tqdm import tqdm
import cv2
import pickle
from transformers import BertTokenizer
import cn_clip.clip as clip

def image_loader(filename):
    try:
        img = Image.open(filename).convert('RGB')
        return img
    except Exception as e:
        img = cv2.imread(filename)
        return img

def title_augmentation(title):
    if (len(title) >= 10) and (random.uniform(0, 1) > 0.4):
        rnd_index = list(range(len(title)))
        random.shuffle(rnd_index)
        rnd_index = rnd_index[:int(0.2*len(title))]
        title_aug = []
        for i, elem in enumerate(title):
            if i in rnd_index:
                rnd = random.uniform(0, 1)
                if rnd < 0.333:
                    title_aug.append(elem * 2)
                elif rnd < 0.666:
                    title_aug.append('')
                else:
                    title_aug.append(elem + ' ')
            else:
                title_aug.append(elem)
        title = ''.join(title_aug)
    return title

def get_indices(cate_group, n):
    indices_list = []
    for cate_list in cate_group:
        random.shuffle(cate_list)
        for i in range(0, len(cate_list), n):
            indices_list.append(cate_list[i: i+n])

    random.shuffle(indices_list)
    indices = []
    for tmp in indices_list:
        indices.extend(tmp)
    return indices

def get_indices_for_hard_sample(type2datalist, level, n):
    indices_list = []
    for key in type2datalist: # type->level->cluster
        cluster_dict = type2datalist[key][level]
        cluster_keys = list(cluster_dict.keys())
        random.shuffle(cluster_keys)
        for cluster in cluster_keys:
            index_list = type2datalist[key][level][cluster]
            batch_num = int(len(index_list) / n)
            for i in range(batch_num):
                indices_list.append(index_list[i * n : (i + 1) * n])

    random.shuffle(indices_list)
    indices = []
    for tmp in indices_list:
        indices.extend(tmp)
    return indices

class SharkEmbItemInferDataset(data.Dataset):
    def __init__(self, item2fullpath, item2title, 
                shuffle=False, transform=None, return_img_id=False):
        with open(item2fullpath, "rb") as f:
            self.item2fullpath = pickle.load(f)
        with open(item2title, "rb") as f:
            self.item2title = pickle.load(f)
        self.shuffle = shuffle
        self.return_img_id = return_img_id
        self.transform = transform

        self.items = list(self.item2fullpath.keys()&self.item2title.keys())
        self.train_num = len(self.items)
        #d_a = dict(self.train_list)
        #self.train_list = pd.Series(d_a, index=d_a.keys())
        print("total item num: {}".format(self.train_num))

    def __getitem__(self, index):
        item_id = self.items[index]
        itme_img_path = self.item2fullpath[item_id]
        item_title = self.item2title[item_id]
        item_title_mask = torch.tensor([1])
        if item_title == '空':
            item_title_mask = torch.tensor([0])
        item_img = image_loader(itme_img_path)
        item_img = [item_img]
        
        if self.transform is not None:
            item_img = self.transform(item_img)

        return item_title, item_title_mask, item_img, item_id

    def __len__(self):
        return self.train_num

class SharkEmbPhotoInferDataset(data.Dataset):
    def __init__(self, photo2fullpath, photo2text, len_clip, 
                shuffle=False, transform=None, return_img_id=False):
        with open(photo2fullpath, "rb") as f:
            self.photo2fullpath = pickle.load(f)
        with open(photo2text, 'rb') as f:
            self.photo2text = pickle.load(f)
        self.len_clip = len_clip
        self.shuffle = shuffle
        self.return_img_id = return_img_id
        self.transform = transform

        self.photos = list(self.photo2fullpath.keys() & self.photo2text.keys())
        self.train_num = len(self.photos)
        #d_a = dict(self.train_list)
        #self.train_list = pd.Series(d_a, index=d_a.keys())
        print("total photo num: {}".format(self.train_num))

    def __getitem__(self, index):
        photo_id = self.photos[index]
        photo_path_list = self.photo2fullpath[photo_id]
        photo_text = self.photo2text[photo_id]
        photo_text_mask = torch.tensor([1])
        if photo_text == '空':
            photo_texts_mask = torch.tensor([0])
        
        clip_length = self.len_clip
        L = len(photo_path_list) / clip_length
        seed = [int((2*i+1)*L/2) for i in range(clip_length)]
        imgs = []
        for idx in seed:
            img_full_path = photo_path_list[idx]
            img = image_loader(img_full_path)
            if img is None:
                continue
            imgs.append(img)

        if self.transform is not None:
            imgs = self.transform(imgs)

        return photo_text, photo_text_mask, imgs, photo_id

    def __len__(self):
        return self.train_num

class SharkEmbLiveInferDataset(data.Dataset):
    def __init__(self, live2fullpath, live2text, len_clip, 
                shuffle=False, transform=None, return_img_id=False):
        with open(live2fullpath, "rb") as f:
            self.live2fullpath = pickle.load(f)
        with open(live2text, 'rb') as f:
            self.live2text = pickle.load(f)
        self.len_clip = len_clip
        self.shuffle = shuffle
        self.return_img_id = return_img_id
        self.transform = transform

        self.lives = list(self.live2fullpath.keys() & self.live2text.keys())
        self.train_num = len(self.lives)
        #d_a = dict(self.train_list)
        #self.train_list = pd.Series(d_a, index=d_a.keys())
        print("total live num: {}".format(self.train_num))

    def __getitem__(self, index):
        live = self.lives[index]
        live_id, starttime, endtime = live[0], live[1], live[2]
        live_path_list = self.live2fullpath[live]
        live_text = self.live2text[live]
        live_text_mask = torch.tensor([1])
        if live_text == '空':
            live_text_mask = torch.tensor([0])
        clip_length = self.len_clip
        L = len(live_path_list) / clip_length
        seed = [int((2*i+1)*L/2) for i in range(clip_length)]
        imgs = []
        for idx in seed:
            img_full_path = live_path_list[idx]
            img = image_loader(img_full_path)
            if img is None:
                continue
            imgs.append(img)
        while len(imgs) < self.len_clip:
            imgs.append(imgs[-1])

        if self.transform is not None:
            imgs = self.transform(imgs)


        return live_text, live_text_mask, imgs, live_id, starttime, endtime

    def __len__(self):
        return self.train_num

class CollateFnItem(object):
    def __init__(self):
        self.tokenizer1 = BertTokenizer.from_pretrained('/share/ad/baixuehan03/pretrained/rbt3', local_files_only=True)
    def __call__(self, data):
        batch_data = list(zip(*data))
        batch_data[0] = self.tokenizer1(list(batch_data[0]), return_tensors='pt', padding=True, truncation=True, max_length=32)
        batch_data[1] = torch.cat(batch_data[1], 0)
        batch_data[2] = torch.cat(batch_data[2], 0)
        return batch_data

class CollateFnPhoto(object):
    def __init__(self):
        self.tokenizer1 = BertTokenizer.from_pretrained('/share/ad/baixuehan03/pretrained/rbt3', local_files_only=True)
    def __call__(self, data):
        batch_data = list(zip(*data))
        batch_data[0] = self.tokenizer1(list(batch_data[0]), return_tensors='pt', padding=True, truncation=True, max_length=32)
        batch_data[1] = torch.cat(batch_data[1], 0)
        batch_data[2] = torch.cat(batch_data[2], 0)
        return batch_data

class CollateFnLive(object):
    def __init__(self):
        self.tokenizer1 = BertTokenizer.from_pretrained('/share/ad/baixuehan03/pretrained/rbt3', local_files_only=True)
    def __call__(self, data):
        batch_data = list(zip(*data))
        batch_data[0] = self.tokenizer1(list(batch_data[0]), return_tensors='pt', padding=True, truncation=True, max_length=32)
        batch_data[1] = torch.cat(batch_data[1], 0)
        batch_data[2] = torch.cat(batch_data[2], 0)
        return batch_data

class ChineseCLIPCollateFnTest(object):
    def __call__(self, data):
        batch_data = list(zip(*data))
        batch_data[0] = clip.tokenize(list(batch_data[0]))
        batch_data[1] = torch.cat(batch_data[1], 0)
        return batch_data

