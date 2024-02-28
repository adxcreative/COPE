# -*- coding: utf-8 -*-
# liyan26@kuaishou.com 李岩 @2022-09-19 21:45:58
# Last Change:  2022-10-19 15:18:20

from typing import Tuple, Union, OrderedDict
from timm.models.layers import trunc_normal_
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from torch.utils.checkpoint import checkpoint_sequential
import math

__all__ = ['xclip_vision']

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        # orig_type = x.dtype
        # ret = super().forward(x.type(torch.float32))
        # return ret.type(orig_type)
        return super().forward(x)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class CrossFramelAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, droppath = 0., T=0, ):
        super().__init__()
        self.T = T

        self.message_fc = nn.Linear(d_model, d_model)
        self.message_ln = LayerNorm(d_model)
        self.message_attn = nn.MultiheadAttention(d_model, n_head,)

        self.attn = nn.MultiheadAttention(d_model, n_head,)
        self.ln_1 = LayerNorm(d_model)

        self.drop_path = DropPath(droppath) if droppath > 0. else nn.Identity()
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]


    def forward(self, x):
        l, b, seq, d = x.size()

        msg_token = self.message_fc(x[0,:,:,:])
        msg_token = msg_token.view(b, seq, 1, d)

        msg_token = msg_token.permute(1,2,0,3).view(seq, b, d)
        msg_token = msg_token + self.drop_path(self.message_attn(self.message_ln (msg_token),self.message_ln(msg_token),self.message_ln(msg_token),need_weights=False)[0])
        msg_token = msg_token.view(seq, 1, b, d).permute(1,2,0,3)

        x = torch.cat([x, msg_token], dim=0)

        x = x.view(l+1, -1, d)
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x[:l,:,:]
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        x = x.view(l, b, seq, d)
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, droppath=None, use_checkpoint=False, T=8):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        if droppath is None:
            droppath = [0.0 for i in range(layers)]
        self.width = width
        self.layers = layers

        self.resblocks = nn.Sequential(*[CrossFramelAttentionBlock(width, heads, attn_mask, droppath[i], T) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        if not self.use_checkpoint:
            return self.resblocks(x)
        else:
            return checkpoint_sequential(self.resblocks, 3, x)


class CrossFrameCommunicationTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 droppath = None, T = 8, use_checkpoint = False,):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        ## Attention Blocks
        self.transformer = Transformer(width, layers, heads, droppath=droppath, use_checkpoint=use_checkpoint, T=T,)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))


    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, seq:int):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        x = self.ln_pre(x)
                

        x = x.permute(1, 0, 2)
        l, bt, d = x.size()
        b = bt//seq
        x = x.view(l, b, seq, d)
        x = self.transformer(x)
        x = x.view(l, -1, d)
        x = x.permute(1, 0, 2)

        x = self.ln_post(x)
        cls_x = x[:, 0, :]
        cls_x_old = cls_x

        if self.proj is not None:
            cls_x = cls_x @ self.proj

        return cls_x, x[:,1:,:], cls_x_old

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MultiframeIntegrationTransformer(nn.Module):
    def __init__(self, T, embed_dim=512, layers=1,):
        super().__init__()
        self.T = T
        transformer_heads = embed_dim // 64
        self.positional_embedding = nn.Parameter(torch.empty(1, T, embed_dim))
        trunc_normal_(self.positional_embedding, std=0.02)
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(d_model=embed_dim, n_head=transformer_heads) for _ in range(layers)])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear,)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x, return_hidden=False):
        ori_x = x
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.resblocks(x)
        x = x.permute(1, 0, 2)
        x = x.type(ori_x.dtype) + ori_x

        if return_hidden:
            return x.mean(dim=1, keepdim=False), x

        return x.mean(dim=1, keepdim=False)


class VISIONXCLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 fp16 : bool,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # video
                 T=8,
                 droppath=0.,
                 mit_layers=1,
                 # prompt
                 prompts_alpha=1e-4,
                 prompts_layers=1,
                 # other
                 use_cache=True,
                 use_checkpoint=False,
                 ):
        super().__init__()

        self.fp16 = fp16
        self.output_size = embed_dim
        self.use_cache=use_cache
        self.mit = MultiframeIntegrationTransformer(T=T, embed_dim=embed_dim, layers=mit_layers,)

        dpr = [x.item() for x in torch.linspace(0, droppath, vision_layers)] if droppath > 0. else None

        vision_heads = vision_width // 64
        self.visual = CrossFrameCommunicationTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            droppath=dpr,
            T=T,
            use_checkpoint=use_checkpoint
        )

        #self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.cache_text_features = None
    
    def encode_image(self, image, seq):
        return self.visual(image, seq)

    def encode_video(self, image):
        b,t,c,h,w = image.size()
        image = image.reshape(-1,c,h,w)
        seq = t

        # cls_features: [b*t, dim], img_features: [b*t, patch_num, dim]
        cls_features, img_features, cls_features_old = self.encode_image(image, seq)
        #img_features = self.prompts_visual_ln(img_features)
        #img_features = img_features @ self.prompts_visual_proj

        cls_features = cls_features.view(b, t, -1).contiguous()
        cls_features_old = cls_features_old.view(b, t, -1).contiguous()
        img_features = img_features.view(b, t, img_features.shape[-2], img_features.shape[-1]).contiguous()

        video_features, cls_features_new = self.mit(cls_features, return_hidden=True)

        return video_features, cls_features_new, img_features, cls_features_old

    def forward(self, image):
        with torch.cuda.amp.autocast(self.fp16):
            b = image.shape[0]
        #video_features, cls_features, img_features = self.encode_video(image)
            video_features, cls_features, img_features, cls_features_old = self.encode_video(image)
        #img_features = img_features.mean(dim=1, keepdim=False)
        #cls_features = cls_features.mean(dim=1, keepdim=False)
#             video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        if self.fp16:
            video_features, cls_features, img_features, cls_features_old = video_features.float(), cls_features.float(), img_features.float(), cls_features_old.float()
        return video_features, cls_features, img_features, cls_features_old

def build_model(state_dict: dict, pretrained = False, fp16=True,  T=8, droppath=0., use_checkpoint=False, logger=None, prompts_alpha=1e-1, prompts_layers=2, use_cache=True, mit_layers=4,):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)

        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]

    model = VISIONXCLIP(
        embed_dim, fp16,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        T=T, droppath=droppath, mit_layers=mit_layers,
        prompts_alpha=prompts_alpha, prompts_layers=prompts_layers,
        use_checkpoint=use_checkpoint, use_cache=use_cache,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]
    if pretrained:
        msg = model.load_state_dict(state_dict,strict=False)
    #logger.info(f"load pretrained CLIP: {msg}")

    return model

def xclip_vision(pretrained=False, fp16=True):
    if pretrained:
        path = "/share/ad/baixuehan03/pretrained/x-clip/k600_16_8.pth"
        model_state_dict = torch.load(path)['model']
    else:
        model_state_dict = None
    model = build_model(model_state_dict, pretrained, fp16)
    # zrx_modify
    # checkpoint_path = '/data/phd/SPU/shark/experiments_zrx/cross_domain_emb_rbt6-master/partial_spu_pretrain_cross-domain-in-modal-align/checkpoints/checkpoint_70.pth.tar'
    # checkpoint = torch.load(checkpoint_path, map_location='cpu')
    # model_state_dict = OrderedDict()
    # for k, v in checkpoint['state_dict'].items():
    #     if 'visual_encoder' in k:
    #         if k.startswith('module.'):
    #             k = k[7:]
    #         k = k.split('visual_encoder.')[-1]
    #         model_state_dict[k] = v
    # model.load_state_dict(model_state_dict, strict=True)
    # print('using pretrained visual encoder checkpoint')
    return model