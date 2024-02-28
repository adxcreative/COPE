from models.Qformer import BertConfig, BertLMHeadModel
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from models.module_fusion_layer import TextVideoTransformerFusionModel, TextVideoAttentionFusionModel
from models.text_relevance_model import Roberta
from models.module_xclip import xclip_vision
from models.cross_domain_matching_head import CrossDomainMatchingHead
from losses.text_relevance_loss import TextRelevanceLoss
from losses.partial_fc import CombinedMarginLoss, PartialFC_V2
from models.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
import numpy as np
from models.util_module import AllGather

allgather = AllGather.apply

class shark(nn.Module):
    def __init__(
        self,
        emb_dim = 512,
        qformer_cross_attention_freq = 1,
        num_qformer_hidden_layer = 4,
        num_fusion_hidden_layer = 4, 
        mixed_precision_training = True,
    ):
        super(shark, self).__init__()

        self.emb_dim = emb_dim
        print('create text encoder...')
        self.text_encoder = Roberta(model_type='rbt6', fp16=mixed_precision_training)
        print('create visual encoder...')
        self.visual_encoder = xclip_vision(pretrained=True, fp16=mixed_precision_training)
        
        self.item_fusion_model = TextVideoAttentionFusionModel(input_dims=[self.text_encoder.output_size, self.visual_encoder.output_size], emb_dim=emb_dim, fp16=mixed_precision_training)
        self.photo_fusion_model = TextVideoAttentionFusionModel(input_dims=[None, self.visual_encoder.output_size], emb_dim=emb_dim, fp16=mixed_precision_training)
        self.live_fusion_model = TextVideoAttentionFusionModel(input_dims=[None, self.visual_encoder.output_size], emb_dim=emb_dim, fp16=mixed_precision_training)
        
        self.temp = 0.05

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.alpha = 0.3


    def forward(self, text_data, visual_data):
        # backbone
        item_text_emb, item_word_emb = self.text_encoder(text_data['item'])
        # photo_text_emb, photo_word_emb = self.text_encoder(text_data['photo'])
        # live_text_emb, live_word_emb = self.text_encoder(text_data['live'])

        item_video_emb, _, item_patch_emb, item_frame_emb = self.visual_encoder(visual_data['item'])
        photo_video_emb, _, photo_patch_emb, photo_frame_emb = self.visual_encoder(visual_data['photo'])
        live_video_emb, _, live_patch_emb, live_frame_emb = self.visual_encoder(visual_data['live'])
        
        # text and visual fusion layer
        item_emb_logit, item_t_emb_logit, item_v_emb_logit = self.item_fusion_model([item_text_emb, item_video_emb])
        _, _, photo_v_emb_logit = self.photo_fusion_model([None, photo_video_emb])
        _, _, live_v_emb_logit = self.live_fusion_model([None, live_video_emb])

        emb_dict = {}
        emb_dict['item_emb_logit'] = item_emb_logit
        emb_dict['photo_emb_logit'] = photo_v_emb_logit
        emb_dict['live_emb_logit'] = live_v_emb_logit

        loss_dict = {}
        ###============== Cross-domain Contrastive ===============###
        item_emb = F.normalize(item_emb_logit, dim=-1) # [batch_size, dim]
        item_t_emb = F.normalize(item_t_emb_logit, dim=-1)
        item_v_emb = F.normalize(item_v_emb_logit, dim=-1)
        photo_v_emb = F.normalize(photo_v_emb_logit, dim=-1)
        live_v_emb = F.normalize(live_v_emb_logit, dim=-1)

        item_emb = allgather(item_emb) # [batch_size*num_gpu, dim]
        item_t_emb = allgather(item_t_emb)
        item_v_emb = allgather(item_v_emb)
        photo_v_emb = allgather(photo_v_emb)
        live_v_emb = allgather(live_v_emb)
        torch.distributed.barrier()

        sim_iv2pv = (item_v_emb @ photo_v_emb.t()) / self.temp
        sim_iv2lv = (item_v_emb @ live_v_emb.t()) / self.temp
        sim_pv2lv = (photo_v_emb @ live_v_emb.t()) / self.temp

        sim_i2pv = (item_emb @ photo_v_emb.t()) / self.temp # [batch_size*num_gpu, batch_size*num_gpu]
        sim_i2lv = (item_emb @ live_v_emb.t()) / self.temp # [batch_size*num_gpu, batch_size*num_gpu]

        sim_it2pv = (item_t_emb @ photo_v_emb.t()) / self.temp
        sim_it2lv = (item_t_emb @ live_v_emb.t()) / self.temp

        bs = item_emb.size(0)
        targets = torch.arange(bs).cuda().to(item_emb.device)

        loss_dict['loss_ivpvc'] = (F.cross_entropy(sim_iv2pv, targets, label_smoothing=0.1) + F.cross_entropy(sim_iv2pv.t(), targets, label_smoothing=0.1)) / 2
        loss_dict['loss_ivlvc'] = (F.cross_entropy(sim_iv2lv, targets, label_smoothing=0.1) + F.cross_entropy(sim_iv2lv.t(), targets, label_smoothing=0.1)) / 2
        loss_dict['loss_pvlvc'] = (F.cross_entropy(sim_pv2lv, targets, label_smoothing=0.1) + F.cross_entropy(sim_pv2lv.t(), targets, label_smoothing=0.1)) / 2

        loss_dict['loss_ipvc'] = (F.cross_entropy(sim_i2pv, targets, label_smoothing=0.1) + F.cross_entropy(sim_i2pv.t(), targets, label_smoothing=0.1)) / 2
        loss_dict['loss_ilvc'] = (F.cross_entropy(sim_i2lv, targets, label_smoothing=0.1) + F.cross_entropy(sim_i2lv.t(), targets, label_smoothing=0.1)) / 2

        loss_dict['loss_itpvc'] = (F.cross_entropy(sim_it2pv, targets, label_smoothing=0.1) + F.cross_entropy(sim_it2pv.t(), targets, label_smoothing=0.1)) / 2
        loss_dict['loss_itlvc'] = (F.cross_entropy(sim_it2lv, targets, label_smoothing=0.1) + F.cross_entropy(sim_it2lv.t(), targets, label_smoothing=0.1)) / 2

        return loss_dict, emb_dict

    def extract_features(self, text_data, visual_data):
        item_emb = None
        item_t_emb = None
        item_v_emb = None
        photo_emb = None
        photo_t_emb = None
        photo_v_emb = None
        live_emb = None
        live_t_emb = None
        live_v_emb = None
        if visual_data['item'] is not None:
            # backbone
            item_text_emb, item_word_emb = self.text_encoder(text_data['item'])
            item_video_emb, _, item_patch_emb, item_frame_emb = self.visual_encoder(visual_data['item'])

            # text and visual fusion layer
            item_emb_logit, item_t_emb_logit, item_v_emb_logit = self.item_fusion_model([item_text_emb, item_video_emb])

            item_emb = F.normalize(item_emb_logit, dim=-1) # [batch_size, dim]
            item_t_emb = F.normalize(item_t_emb_logit, dim=-1)
            item_v_emb = F.normalize(item_v_emb_logit, dim=-1)

        if visual_data['photo'] is not None:
            photo_video_emb, _, photo_patch_emb, photo_frame_emb = self.visual_encoder(visual_data['photo'])
            _, _, photo_v_emb_logit = self.photo_fusion_model([None, photo_video_emb])
            
            photo_v_emb = F.normalize(photo_v_emb_logit, dim=-1) # [batch_size, dim]

        if visual_data['live'] is not None:
            live_video_emb, _, live_patch_emb, live_frame_emb = self.visual_encoder(visual_data['live'])
            _, _, live_v_emb_logit = self.live_fusion_model([None, live_video_emb])
            
            live_v_emb = F.normalize(live_v_emb_logit, dim=-1)

        return item_emb, photo_v_emb, live_v_emb


if __name__ == "__main__":
    model = shark()
