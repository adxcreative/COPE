import base64
import copy
import io
import os
import sys
# Last Change:  2024-02-28 16:47:07
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
import lmdb
from random import choice, random, uniform, shuffle
from transformers import BertTokenizer
import cn_clip.clip as clip

def image_loader(filename):
    try:
        img = Image.open(filename).convert('RGB')
        width, height = img.size
        if width <= 0 or height <= 0:
            raise Exception("width <= 0 or height <= 0")
        return img
    except Exception as e:
        print(e)
        return None

# 文本处理复用之前的逻辑
def title_augmentation(title):
    if (len(title) >= 10) and (uniform(0, 1) > 0.4):
        rnd_index = list(range(len(title)))
        shuffle(rnd_index)
        rnd_index = rnd_index[:int(0.2*len(title))]
        title_aug = []
        for i, elem in enumerate(title):
            if i in rnd_index:
                rnd = uniform(0, 1)
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

def frames_sample_seed(frames_num, clip_length):
    L = frames_num / clip_length
    seed = [int((2*i+1)*L/2) for i in range(clip_length)]
    return seed

def frames_sample(frames, clip_length):
    frames_num = len(frames)
    seed = frames_sample_seed(frames_num, clip_length)
    return [frames[idx] for idx in seed]

class SharkEmbSpuNewLmdbDataset(data.Dataset):

    def __init__(self, spu_list_path, len_clip,
                 spu2item_lmdb, spu2photo_lmdb, spu2live_lmdb, 
                 clip2live_lmdb, live2texts_lmdb, photo2frames_lmdb, photo2texts_lmdb,
                 item_imgs_lmdb, item_titles_lmdb, photo2fullpath_lmdb, live2fullpath_lmdb,
                 shuffle=False, transform=None, use_two_part=False, use_positive_sample=False):
        self.spu_list_path = spu_list_path
        self.spu2item_lmdb = spu2item_lmdb
        self.spu2photo_lmdb = spu2photo_lmdb
        self.spu2live_lmdb = spu2live_lmdb
        self.clip2live_lmdb = clip2live_lmdb
        self.live2texts_lmdb = live2texts_lmdb
        self.photo2frames_lmdb = photo2frames_lmdb
        self.photo2texts_lmdb = photo2texts_lmdb
        self.item_imgs_lmdb = item_imgs_lmdb
        self.item_titles_lmdb = item_titles_lmdb
        self.photo2fullpath_lmdb = photo2fullpath_lmdb
        self.live2fullpath_lmdb = live2fullpath_lmdb
        self.transform = transform
        self.len_clip = len_clip
        self.use_two_part = use_two_part
        self.use_positive_sample = use_positive_sample
        
        with open(self.spu_list_path, "r") as f:
            self.spu_list = [line.strip() for line in f.readlines()]

        self.spu2item_env = lmdb.open(self.spu2item_lmdb, map_size=1099511627776, lock=False)
        self.spu2photo_env = lmdb.open(self.spu2photo_lmdb, map_size=1099511627776, lock=False)
        self.spu2live_env = lmdb.open(self.spu2live_lmdb, map_size=1099511627776, lock=False)
        self.clip2live_env = lmdb.open(self.clip2live_lmdb, map_size=1099511627776, lock=False)
        self.live2texts_env = lmdb.open(self.live2texts_lmdb, map_size=1099511627776, lock=False)
        self.photo2frames_env = lmdb.open(self.photo2frames_lmdb, map_size=1099511627776, lock=False)
        self.photo2texts_env = lmdb.open(self.photo2texts_lmdb, map_size=1099511627776, lock=False)
        self.item_imgs_env = lmdb.open(self.item_imgs_lmdb, map_size=1099511627776, lock=False)
        self.item_titles_env = lmdb.open(self.item_titles_lmdb, map_size=1099511627776, lock=False)
        self.photo2fullpath_env = lmdb.open(self.photo2fullpath_lmdb, map_size=1099511627776, lock=False)
        self.live2fullpath_env = lmdb.open(self.live2fullpath_lmdb, map_size=1099511627776, lock=False)
        
        self.spu2item_txn = self.spu2item_env.begin()
        self.spu2photo_txn = self.spu2photo_env.begin()
        self.spu2live_txn = self.spu2live_env.begin()
        self.clip2live_txn = self.clip2live_env.begin()
        self.live2texts_txn = self.live2texts_env.begin()
        self.photo2frames_txn = self.photo2frames_env.begin()
        self.photo2texts_txn = self.photo2texts_env.begin()
        self.item_imgs_txn = self.item_imgs_env.begin()
        self.item_titles_txn = self.item_titles_env.begin()
        self.photo2fullpath_txn = self.photo2fullpath_env.begin()
        self.live2fullpath_txn = self.live2fullpath_env.begin()
        
        self.train_num = len(self.spu_list)
        print("train file: {}".format(self.spu_list_path))
        print("total spu nums: {}".format(self.train_num))

    def get_items_by_spu(self, spu_id):
        items = self.spu2item_txn.get(spu_id.encode("utf-8")).decode("utf-8").split(",")
        return items
    
    def get_sample_item_by_spu(self, spu_id):
        items = self.get_items_by_spu(spu_id)
        item_id = choice(items)
        # item_id = items[0]
        return item_id
    
    def get_item_img_by_item(self, item_id):
        value = self.item_imgs_txn.get(item_id.encode("utf-8"))
        img_bytes = base64.b64decode(value)
        img = Image.open(io.BytesIO(img_bytes))
        img = img.convert('RGB')
        return img
    
    def get_item_title_by_item(self, item_id):
        return self.item_titles_txn.get(item_id.encode("utf-8")).decode("utf-8")
        
    def get_photos_by_spu(self, spu_id):
        photos = self.spu2photo_txn.get(spu_id.encode("utf-8")).decode("utf-8").split(",")
        return photos
    
    def get_sample_photo_by_spu(self, spu_id):
        photos = self.get_photos_by_spu(spu_id)
        photo_id = choice(photos)
        # photo_id = photos[0]
        return photo_id
    
    def get_imgs_by_photo(self, photo_id):
        frame_ids = self.photo2frames_txn.get(photo_id.encode("utf-8")).decode("utf-8").split(",")
        sample_frame_ids = frames_sample(frame_ids, self.len_clip)
        frame_imgs = list()
        for frame_id in sample_frame_ids:
            frame_path = self.photo2fullpath_txn.get(frame_id.encode("utf-8")).decode("utf-8")
            img = Image.open(frame_path)
            img = img.convert('RGB')
            frame_imgs.append(img)
        return frame_imgs

    def get_txts_by_photo(self, photo_id):
        return self.photo2texts_txn.get(photo_id.encode("utf-8")).decode("utf-8").strip().replace('\n', ' ')
    
    def get_lives_by_spu(self, spu_id):
        lives = self.spu2live_txn.get(spu_id.encode("utf-8")).decode("utf-8").split(",")
        return lives
    
    def get_sample_live_by_spu(self, spu_id):
        lives = self.get_lives_by_spu(spu_id)
        live = choice(lives)
        # live = lives[0]
        return live

    def get_imgs_by_live(self, live):
        frame_ids = self.clip2live_txn.get(live.encode("utf-8")).decode("utf-8").split(",")
        sample_frame_ids = frames_sample(frame_ids, self.len_clip)
        frame_imgs = list()
        for frame_id in sample_frame_ids:
            frame_path = self.live2fullpath_txn.get(frame_id.encode("utf-8")).decode("utf-8")
            img = Image.open(frame_path)
            img = img.convert('RGB')
            frame_imgs.append(img)
        return frame_imgs

    def get_txts_by_live(self, live):
        return self.live2texts_txn.get(live.encode("utf-8")).decode("utf-8").strip().replace('\n', ' ')

    def __getitem__(self, index):
        spu_id = self.spu_list[index]
        item_id = self.get_sample_item_by_spu(spu_id)
        item_img = self.get_item_img_by_item(item_id)
        item_title = self.get_item_title_by_item(item_id)
        item_title_mask = torch.tensor([1])
        if item_title == '空':
            item_title_mask = torch.tensor([0])

        if self.use_two_part:
            while True:
                try:
                    if random() > 0.5:
                        photo_id = self.get_sample_photo_by_spu(spu_id)
                        pid = photo_id
                        pid_source = "photo"
                        frames = self.get_imgs_by_photo(photo_id)
                    else:
                        live = self.get_sample_live_by_spu(spu_id)
                        pid = live
                        pid_source = "live"
                        frames = self.get_imgs_by_live(live)

                    item_img = [item_img]

                    if self.transform is not None:
                        item_img = self.transform(item_img)
                        frames = self.transform(frames)
                 
                    break
                    
                except Exception as e:
                    print("fail get data {}".format(e))
                

            return item_title, item_img, frames, pid, item_id, pid_source, spu_id
        else:
            while True:
                try:
                    photo_id = self.get_sample_photo_by_spu(spu_id)
                    photo_frames = self.get_imgs_by_photo(photo_id)
                    photo_texts = self.get_txts_by_photo(photo_id)
                    photo_texts_mask = torch.tensor([1])
                    if photo_texts == '空':
                        photo_texts_mask = torch.tensor([0])
                    live = self.get_sample_live_by_spu(spu_id)
                    live_frames = self.get_imgs_by_live(live)
                    live_texts = self.get_txts_by_live(live)
                    live_texts_mask = torch.tensor([1])
                    if live_texts == '空':
                        live_texts_mask = torch.tensor([0])

                    item_img = [item_img]

                    if self.transform is not None:
                        item_img = self.transform(item_img)
                        photo_frames = self.transform(photo_frames)
                        live_frames = self.transform(live_frames)
                        
                    break

                except Exception as e:
                    print("fail get data {}".format(e))
                    
            if self.use_positive_sample:
                while True:
                    try:
                        positive_item_id = self.get_sample_item_by_spu(spu_id)
                        positive_item_img = self.get_item_img_by_item(positive_item_id)
                        positive_item_title = self.get_item_title_by_item(positive_item_id)
                        positive_item_title = title_augmentation(positive_item_title)
                        break
                    except Exception as e:
                        print("fail get data {}".format(e))

                while True:
                    try:
                        positive_photo_id = self.get_sample_photo_by_spu(spu_id)
                        positive_photo_frames = self.get_imgs_by_photo(positive_photo_id)
                        positive_photo_texts = self.get_txts_by_photo(positive_photo_id)
                        break
                    except Exception as e:
                        print("fail get data {}".format(e))
                
                while True:
                    try:                
                        positive_live = self.get_sample_live_by_spu(spu_id)
                        positive_live_frames = self.get_imgs_by_live(positive_live)
                        positive_live_texts = self.get_txts_by_live(positive_live)                    
                        break
                    except Exception as e:
                        print("fail get data {}".format(e))                    
                
                positive_item_img = [positive_item_img]

                if self.transform is not None:
                    positive_item_img = self.transform(positive_item_img)
                    positive_photo_frames = self.transform(positive_photo_frames)
                    positive_live_frames = self.transform(positive_live_frames) 
                
                return item_title, item_img, item_id, live_texts, live_frames, live, photo_texts, photo_frames, photo_id, spu_id, positive_item_title, positive_item_img, positive_item_id, positive_live_texts, positive_live_frames, positive_live, positive_photo_texts, positive_photo_frames, positive_photo_id
                
            return item_title, item_title_mask, item_img, item_id, live_texts, live_texts_mask, live_frames, live, photo_texts, photo_texts_mask, photo_frames, photo_id, spu_id


    def __len__(self):
        return self.train_num
    

class TwoPartCollateFn(object):
    def __init__(self):
        self.tokenizer1 = BertTokenizer.from_pretrained('/share/ad/baixuehan03/pretrained/rbt3', local_files_only=True)
#         self.tokenizer2 = BertTokenizer.from_pretrained('hfl/rbt6', cache_dir='/data/baixuehan03/cross_domain_emb-master/pretrained/hfl-rbt6', local_files_only=True)
    def __call__(self, data):
        batch_data = list(zip(*data))
        # goods_title_text, goods_img,  imgs
        batch_data[0] = self.tokenizer1(list(batch_data[0]), return_tensors='pt', padding=True, truncation=True, max_length=15)
        # batch_data[1] = self.tokenizer2(list(batch_data[1]), return_tensors='pt', padding=True, truncation=True, max_length=60)
        #batch_data[1] = torch.stack(batch_data[1])
        #batch_data[2] = torch.stack(batch_data[2])
        batch_data[1] = torch.cat(batch_data[1], 0)
        batch_data[2] = torch.cat(batch_data[2], 0)
        return batch_data
    
class ThreePartCollateFn(object):
    def __init__(self):
        self.tokenizer1 = BertTokenizer.from_pretrained('/share/ad/baixuehan03/pretrained/rbt3', local_files_only=True)
#         self.tokenizer2 = BertTokenizer.from_pretrained('hfl/rbt6', cache_dir='/data/baixuehan03/cross_domain_emb-master/pretrained/hfl-rbt6', local_files_only=True)
    def __call__(self, data):
        batch_data = list(zip(*data))
        # print(batch_data[0])
        # goods_title_text, goods_img,  imgs
        batch_data[0] = self.tokenizer1(list(batch_data[0]), return_tensors='pt', padding=True, truncation=True, max_length=32)
        # batch_data[1] = self.tokenizer2(list(batch_data[1]), return_tensors='pt', padding=True, truncation=True, max_length=60)
        #batch_data[1] = torch.stack(batch_data[1])
        #batch_data[2] = torch.stack(batch_data[2])
        # print(batch_data[1])
        # print(batch_data[2])
        batch_data[1] = torch.cat(batch_data[1], 0)
        batch_data[2] = torch.cat(batch_data[2], 0)
        batch_data[4] = self.tokenizer1(list(batch_data[4]), return_tensors='pt', padding=True, truncation=True, max_length=32)
        batch_data[5] = torch.cat(batch_data[5], 0)
        batch_data[6] = torch.cat(batch_data[6], 0)
        batch_data[8] = self.tokenizer1(list(batch_data[8]), return_tensors='pt', padding=True, truncation=True, max_length=32)
        batch_data[9] = torch.cat(batch_data[9], 0)
        batch_data[10] = torch.cat(batch_data[10], 0)
        return batch_data

class ChineseCLIPCollateFn(object):
    def __call__(self, data):
        batch_data = list(zip(*data))
        batch_data[0] = clip.tokenize(list(batch_data[0]))
        batch_data[1] = torch.cat(batch_data[1], 0)
        batch_data[2] = torch.cat(batch_data[2], 0)
        batch_data[4] = clip.tokenize(list(batch_data[4]))
        batch_data[5] = torch.cat(batch_data[5], 0)
        batch_data[6] = torch.cat(batch_data[6], 0)
        batch_data[8] = clip.tokenize(list(batch_data[8]))
        batch_data[9] = torch.cat(batch_data[9], 0)
        batch_data[10] = torch.cat(batch_data[10], 0)
        return batch_data

class ThreePartPositiveSampleCollateFn(object):
    def __init__(self):
        self.tokenizer1 = BertTokenizer.from_pretrained('/share/ad/baixuehan03/pretrained/rbt3', local_files_only=True)
#         self.tokenizer2 = BertTokenizer.from_pretrained('hfl/rbt6', cache_dir='/data/baixuehan03/cross_domain_emb-master/pretrained/hfl-rbt6', local_files_only=True)
    def __call__(self, data):
        batch_data = list(zip(*data))
        # goods_title_text, goods_img,  imgs
        batch_data[0] = self.tokenizer1(list(batch_data[0]), return_tensors='pt', padding=True, truncation=True, max_length=32)
        batch_data[1] = torch.cat(batch_data[1], 0)
        batch_data[3] = self.tokenizer1(list(batch_data[3]), return_tensors='pt', padding=True, truncation=True, max_length=120)
        batch_data[4] = torch.cat(batch_data[4], 0)
        batch_data[6] = self.tokenizer1(list(batch_data[6]), return_tensors='pt', padding=True, truncation=True, max_length=120)
        batch_data[7] = torch.cat(batch_data[7], 0)
        
        batch_data[10] = self.tokenizer1(list(batch_data[10]), return_tensors='pt', padding=True, truncation=True, max_length=32)
        batch_data[11] = torch.cat(batch_data[11], 0)
        batch_data[13] = self.tokenizer1(list(batch_data[13]), return_tensors='pt', padding=True, truncation=True, max_length=120)
        batch_data[14] = torch.cat(batch_data[14], 0)
        batch_data[16] = self.tokenizer1(list(batch_data[16]), return_tensors='pt', padding=True, truncation=True, max_length=120)
        batch_data[17] = torch.cat(batch_data[17], 0)
        return batch_data
    

