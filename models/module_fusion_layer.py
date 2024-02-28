import torch
import torch.nn as nn
# Last Change:  2022-09-28 21:05:19
import torch.nn.functional as F
import numpy as np
import random
# from lavis.models.clip_models.model import Transformer
from models.module_transformer import Transformer, LayerNorm
from cn_clip.clip.modeling_bert import BertLayer, BertEncoder
from cn_clip.clip.configuration_bert import BertConfig

__all__ = [
    'TextVideoAttentionFusionModel',
    'TextVideoTransformerFusionModel',
]

class AttentionPooling(nn.Module):

    def __init__(self, emb_dim, emb_num):
        super(AttentionPooling, self).__init__()
        self.emb_dim = emb_dim
        self.emb_num = emb_num
        self.projection = nn.Linear(emb_dim * emb_num, emb_num)

    def forward(self, inputs, tb_tools, is_train=True):
        # (B, T, H) -> (B, T)
        energy = self.projection(inputs.view(inputs.shape[0], -1))
        weights = F.softmax(energy, dim=1)
        # (B, T, H) * (B, T, 1) -> (B, H)
        outputs = (inputs * weights.unsqueeze(-1)).sum(dim=1)

        if tb_tools is not None and tb_tools['local_rank'] == 0:
            with torch.no_grad():
                mean_weights = torch.mean(weights, dim=0, keepdim=False)
                tb_tools['tb_writer'].add_scalar('attention_weights/title', mean_weights[0], global_step=tb_tools['global_step'])
                tb_tools['tb_writer'].add_scalar('attention_weights/image', mean_weights[1], global_step=tb_tools['global_step'])

        return outputs

class AttentionPooling2(nn.Module):

    def __init__(self, emb_dim, emb_num, emb_types):
        super(AttentionPooling2, self).__init__()
        self.emb_dim = emb_dim
        self.emb_num = emb_num
        self.emb_types = emb_types
        self.projection = nn.Linear(emb_dim*emb_num, emb_num)

    def forward(self, inputs, tb_tools):
        # (B, T, H) -> (B, T)
        energy = self.projection(inputs.view(inputs.shape[0], -1))
        weights = F.softmax(energy, dim=1)
        if tb_tools is not None and tb_tools['local_rank'] == 0:
            with torch.no_grad():
                mean_weights = torch.mean(weights, dim=0, keepdim=False)
                for i in range(self.emb_num):
                    tb_tools['tb_writer'].add_scalar(
                        'attention_weights/{}'.format(self.emb_types[i]), mean_weights[i], global_step=tb_tools['global_step'])
        # (B, T, H) * (B, T, 1) -> (B, H)
        outputs = (inputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs

class AttentionPooling3(nn.Module):
    def __init__(self, emb_dim, emb_num):
        super(AttentionPooling3, self).__init__()
        self.emb_dim = emb_dim
        self.emb_num = emb_num
        self.projection = nn.Linear(emb_dim*emb_num, emb_dim)

    def forward(self, inputs, tb_tools, wb, is_train=True):
        # (B, T, H) -> (B, H)
        outputs = self.projection(inputs.view(inputs.shape[0], -1))
        return outputs

class TextVideoAttentionFusionModel(nn.Module):
    def __init__(self, input_dims, emb_dim, fp16=True):
        super(TextVideoAttentionFusionModel, self).__init__()
        # doc_text_dim, query_text_dim, doc_image_dim
        if input_dims[0] is not None:
            self.fc0 = nn.Linear(input_dims[0], emb_dim)
        if input_dims[1] is not None:
            self.fc1 = nn.Linear(input_dims[1], emb_dim)
        self.fc2 = nn.Linear(emb_dim*2, emb_dim)
        self.attention_pooling = AttentionPooling3(emb_dim=emb_dim, emb_num=2)
        self.fp16 = fp16

    def forward(self, emb_list, tb_tools=None, is_train=True):
        # text_emb, images_emb
        with torch.cuda.amp.autocast(self.fp16):
            text_emb = emb_list[0]
            video_emb = emb_list[1]
            if text_emb is not None:
                text_emb = self.fc0(text_emb)
            if video_emb is not None:
                video_emb = self.fc1(video_emb)
            if text_emb is not None and video_emb is not None:
                embs = torch.stack([text_emb, video_emb], 1)
                emb = self.attention_pooling(embs, tb_tools, is_train)

            else:
                emb = None
        if self.fp16:
            if emb is not None:
                emb = emb.float()
            if text_emb is not None:
                text_emb = text_emb.float()
            if video_emb is not None:
                video_emb = video_emb.float()
        return emb, text_emb, video_emb
