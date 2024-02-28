import torch
import torch.nn as nn
# Last Change:  2022-09-15 17:25:02
from transformers import BertModel
from collections import OrderedDict


__all__ = [
    'TextRelevanceModel',
    'Bert',
    'Roberta',
]


class TextRelevanceModel(nn.Module):

    def __init__(self, embedding_size=512, fp16=False):
        super(TextRelevanceModel, self).__init__()

        self.fp16 = fp16
        self.embedding_size = embedding_size

        self.bert_model = BertModel.from_pretrained(
            'bert-base-chinese',
            cache_dir='/share/ad/baixuehan03/pretrained/bert-base-chinese',
            output_hidden_states=False)

        self.output_size = self.bert_model.config.pooler_fc_size
        if self.embedding_size > 0:
            self.fc = nn.Linear(self.output_size, self.embedding_size)
            self.output_size = self.embedding_size

    def forward(self, bert_input):
        with torch.cuda.amp.autocast(self.fp16):
            bert_output = self.bert_model(**bert_input)

        bert_emb = bert_output[1].float() if self.fp16 else bert_output[1]

        if self.embedding_size > 0:
            bert_emb = self.fc(bert_emb)

        return bert_emb


class Bert(nn.Module):

    def __init__(self, embedding_size=512, fp16=False):
        super(Bert, self).__init__()

        self.fp16 = fp16
        self.embedding_size = embedding_size

        self.bert_model = BertModel.from_pretrained(
            'bert-base-chinese',
            cache_dir='/share/ad/baixuehan03/pretrained/bert-base-chinese',
            output_hidden_states=False)

        self.output_size = self.bert_model.config.pooler_fc_size
        if self.embedding_size > 0:
            self.fc = nn.Linear(self.output_size, self.embedding_size)
            self.output_size = self.embedding_size

    def forward(self, bert_input):
        with torch.cuda.amp.autocast(self.fp16):
            bert_output = self.bert_model(**bert_input)

        bert_emb = bert_output[1].float() if self.fp16 else bert_output[1]

        if self.embedding_size > 0:
            bert_emb = self.fc(bert_emb)

        return bert_emb


class Roberta(nn.Module):
    def __init__(self, model_type='rbt6', fp16=False):
        super(Roberta, self).__init__()
        self.fp16 = fp16
        if model_type == 'rbt3':
            self.bert_model = BertModel.from_pretrained(
                '/share/ad/baixuehan03/pretrained/rbt3',
#                 cache_dir='/share/ad/baixuehan03/pretrained/hfl-rbt3',
                local_files_only=True,
                output_hidden_states=False)
            # zrx_modify
            # checkpoint_path = '/data/phd/SPU/shark/experiments_zrx/cross_domain_emb-master/partial_spu_independent_domain/checkpoints/checkpoint_70.pth.tar'
            # checkpoint = torch.load(checkpoint_path, map_location='cpu')
            # model_state_dict = OrderedDict()
            # for k, v in checkpoint['state_dict'].items():
            #     if 'text_encoder' in k:
            #         if k.startswith('module.'):
            #             k = k[7:]
            #         k = k.split('text_encoder.bert_model.')[-1]
            #         model_state_dict[k] = v
            # self.bert_model.load_state_dict(model_state_dict, strict=True)
            # print('using pretrained text encoder checkpoint')
        elif model_type == 'rbt6':
            self.bert_model = BertModel.from_pretrained(
                '/share/ad/baixuehan03/pretrained/rbt6',
                # cache_dir='/share/ad/baixuehan03/pretrained/hfl-rbt6',
                local_files_only=True,
                output_hidden_states=False)
            # checkpoint_path = '/data/phd/SPU/shark/experiments_zrx/cross_domain_emb_rbt6-master/partial_spu_pretrain_cross-domain-in-modal-align/checkpoints/checkpoint_70.pth.tar'
            # checkpoint = torch.load(checkpoint_path, map_location='cpu')
            # model_state_dict = OrderedDict()
            # for k, v in checkpoint['state_dict'].items():
            #     if 'text_encoder' in k:
            #         if k.startswith('module.'):
            #             k = k[7:]
            #         k = k.split('text_encoder.bert_model.')[-1]
            #         model_state_dict[k] = v
            # self.bert_model.load_state_dict(model_state_dict, strict=True)
            # print('using pretrained text encoder checkpoint')
        self.output_size = self.bert_model.config.pooler_fc_size

    def forward(self, bert_input):
        with torch.cuda.amp.autocast(self.fp16):
            bert_output = self.bert_model(**bert_input)

        bert_cls_emb = bert_output['pooler_output'].float() if self.fp16 else bert_output['pooler_output']
        bert_word_emb = bert_output['last_hidden_state'].float() if self.fp16 else bert_output['last_hidden_state']

        return bert_cls_emb, bert_word_emb

class ChineseCLIP(nn.Module):
    def __init__(self, model_type='ViT-B-16', fp16=False):
        super(ChineseCLIP, self).__init__()
        self.fp16 = fp16

        import sys
        sys.path.append('/data/phd/SPU/shark/experiments_zrx/ChineseCLIP')
        from clip_utils import load_from_name
        self.clip_model, _ = load_from_name(model_type, device='cpu')

        self.output_size = self.clip_model.text_projection.shape[1]

    @torch.no_grad()
    def forward(self, text_token):
        with torch.cuda.amp.autocast(self.fp16):
            clip_text_features = self.clip_model.encode_text(text_token.view(-1, text_token.shape[1]))
        return clip_text_features
