import os
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil

import torch
from torch import nn
import torch.nn.functional as F
from models.file_utils import cached_path
from models.util_config import PretrainedConfig
from models.util_module import PreTrainedModel, LayerNorm, ACT2FN
from collections import OrderedDict

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {}
CONFIG_NAME = 'cross_config.json'
WEIGHTS_NAME = 'cross_pytorch_model.bin'


class CrossConfig(PretrainedConfig):
    """Configuration class to store the configuration of a `CrossModel`.
    """
    pretrained_model_archive_map = PRETRAINED_MODEL_ARCHIVE_MAP
    config_name = CONFIG_NAME
    weights_name = WEIGHTS_NAME
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        """Constructs CrossConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `CrossModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `CrossModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    
class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob=0.1):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def build_attention_mask(self, attention_mask):
        assert attention_mask.dtype == torch.bool
        attention_mask = attention_mask.type(torch.int)  # bool to int
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        #extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = extended_attention_mask.to(dtype=attention_mask.dtype)  # fp16 compatibility
        #extended_attention_mask = (1.0 - extended_attention_mask) * -1000000.0  # 0 means mask, 1 means not to mask
        extended_attention_mask = extended_attention_mask * -1000000.0  # 1 means mask, 0 means not to mask
        return extended_attention_mask
    
    def forward(self, q_states, k_states, v_states, attention_mask, return_attn_weights=True):
        q_states = q_states.permute(1, 0, 2)  # LND -> NLD
        k_states = k_states.permute(1, 0, 2)  # LND -> NLD
        v_states = v_states.permute(1, 0, 2)  # LND -> NLD
        attention_mask = self.build_attention_mask(attention_mask)
        
        mixed_query_layer = self.query(q_states)
        mixed_key_layer = self.key(k_states)
        mixed_value_layer = self.value(v_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        #import pdb;pdb.set_trace()
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        if return_attn_weights:
            # attn_weights = attention_probs.mean(dim=1)  # mean along num_heads
            attn_weights = attention_probs

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        context_layer = context_layer.permute(1, 0, 2)  # NLD -> LND
        return context_layer, attn_weights
    
class DiffCrossAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob=0.1):
        super(DiffCrossAttention, self).__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.drop = nn.Dropout(0.3)
        
    def forward(self, args_tuple):
        assert len(args_tuple) == 4
        q_tensor, kv_tensor, q_mask, kv_mask = args_tuple
        
        self_output, attn_weights = self.self(q_tensor, kv_tensor, kv_tensor, kv_mask)  # (L bs d)
        
        # Difference
        diff = q_tensor - self_output # (L bs d)
        diff = diff.permute(1, 0, 2) # (bs L d)
        diff_keep = 1.0 - q_mask.type(torch.int) 
        diff = diff * diff_keep.unsqueeze(-1)  # (bs L d)
        diff = torch.unbind(diff, dim=1) # split across dim=1, keepdim=False
        diff = sum(diff)/diff_keep.sum(dim=-1, keepdim=True)  # (bs d)
        
        diff = torch.pow(diff, 2)
        diff = self.drop(self.bn(diff))
        return diff, attn_weights
    
class ResidualCrossAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        #self.attn = nn.MultiheadAttention(d_model, n_head)
        self.attn = SelfAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.n_head = n_head

    def forward(self, args_tuple: tuple):
        assert len(args_tuple)==4
        x, kv, q_mask, kv_mask = args_tuple
        ln_kv = self.ln_1(kv)
        attn_output, attn_output_weights = self.attn(self.ln_1(x), ln_kv, ln_kv, attention_mask=kv_mask, return_attn_weights=True)
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return x, attn_output_weights
    
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.n_head = n_head

    def attention(self, query, key, value, kv_mask, attn_mask=None, need_weights=False):
        '''
        query: (bs L d)
        kv: (bs S d)
        attn_mask: (L S) (bs 1 L S), 只用来限定对k的attention范围，不管q，因为q如果有padding，会在输出时用q的mask过滤。
        kv_mask: (bs S) 扩展为 (bs 1 1 S)
        kv_mask和attn_mask区别：
            最终与attention_scores相加的attn_mask=kv_mask if attn_mask is None else attn_mask+kv_mask
            1) self-attention. 只需要提供kv_mask.
            2）cross-attention. 如果attn_mask是None，表示q全部参与attention，只需要kv_mask. 如果attn_mask不是None，
               表示q不能全部可见，如翻译的时候。
        '''
        return self.attn(query, key, value, key_padding_mask=kv_mask, attn_mask=attn_mask, need_weights=need_weights)

    def forward(self, args_tuple: tuple):
        # x: torch.Tensor, q_mask: torch.Tensor
        # print(args_tuple)
        assert (len(args_tuple)==2 or len(args_tuple)==4)
        if 2==len(args_tuple):  # self-attention
            x, q_mask = args_tuple
            ln_x = self.ln_1(x)
            attn_output, _ = self.attention(query=ln_x, key=ln_x, value=ln_x, kv_mask=q_mask)
            #attn_output, _ = self.attn(query=ln_x, key=ln_x, value=ln_x, key_padding_mask=q_mask, attn_mask=None, need_weights=False)
            x = x + attn_output
            x = x + self.mlp(self.ln_2(x))
            return (x, q_mask)
        else:  # cross-attention
            x, kv, q_mask, kv_mask = args_tuple
            ln_kv = self.ln_1(kv)
            attn_output, attn_output_weights = self.attention(query=self.ln_1(x), key=ln_kv, value=ln_kv, kv_mask=kv_mask, need_weights=True)
            x = x + attn_output
            x = x + self.mlp(self.ln_2(x))
            return x, attn_output_weights
        
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        
        # Opt1
        #self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers)])
        
        # Opt2
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers-1)])
        #self.cross_attention = DiffCrossAttention(width, heads)
        #self.cross_attention = ResidualAttentionBlock(width, heads)
        self.cross_attention = ResidualCrossAttention(width, heads)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, q_mask: torch.Tensor, kv_mask: torch.Tensor):
        # Opt1
        #x, _ = self.resblocks[0]((q, q_mask))  # self-attention
        #x, attn_weights = self.resblocks[1]((x, kv, q_mask, kv_mask)) # cross-attention
        
        # Opt2
        x, _ = self.resblocks((q, q_mask))  # self-attention
        x, attn_weights = self.cross_attention((x, kv, q_mask, kv_mask)) # cross-attention
        return x, attn_weights

class CrossEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(CrossEmbeddings, self).__init__()

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        # self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, concat_embeddings, concat_type=None):

        batch_size, seq_length = concat_embeddings.size(0), concat_embeddings.size(1)
        # if concat_type is None:
        #     concat_type = torch.zeros(batch_size, concat_type).to(concat_embeddings.device)

        position_ids = torch.arange(seq_length, dtype=torch.long, device=concat_embeddings.device)
        position_ids = position_ids.unsqueeze(0).expand(concat_embeddings.size(0), -1)

        # token_type_embeddings = self.token_type_embeddings(concat_type)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = concat_embeddings + position_embeddings # + token_type_embeddings
        # embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class CrossPooler(nn.Module):
    def __init__(self, hidden_size):
        super(CrossPooler, self).__init__()
        self.ln_pool = LayerNorm(hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = QuickGELU()

    def forward(self, hidden_states, hidden_mask):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        hidden_states = self.ln_pool(hidden_states)
        pooled_output = hidden_states[:, 0]
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class CrossModel(PreTrainedModel):
    def initialize_parameters(self):
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def __init__(self, config):
        super(CrossModel, self).__init__(config)

        self.embeddings = CrossEmbeddings(config)

        transformer_width = config.hidden_size
        transformer_layers = config.num_hidden_layers
        transformer_heads = config.num_attention_heads
        self.transformer = Transformer(width=transformer_width, layers=transformer_layers, heads=transformer_heads,)
        self.pooler = CrossPooler(config.hidden_size)
        self.apply(self.init_weights)

    def build_attention_mask(self, attention_mask):
        #extended_attention_mask = attention_mask.unsqueeze(1)
        #extended_attention_mask = attention_mask
        #extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        #extended_attention_mask = (1.0 - extended_attention_mask) * -1000000.0
        #extended_attention_mask = extended_attention_mask.expand(-1, attention_mask.size(1), -1)
        
        extended_attention_mask = attention_mask == 0  # mask==True 
        return extended_attention_mask

    def forward(self, q, kv, q_mask, kv_mask):
        '''
        q: (bs L d)
        kv: (bs S d)
        q_mask: (bs L) 
        kv_mask: (bs S)
        '''
        if q_mask is None:
            q_mask = torch.ones(q.size(0), q.size(1)).to(q.device)
        if kv_mask is None:
            kv_mask = torch.ones(kv.size(0), kv.size(1)).to(kv.device)

        extended_q_mask = self.build_attention_mask(q_mask)
        extended_kv_mask = self.build_attention_mask(kv_mask)
        
        embedding_q = self.embeddings(q)  # Plus position embedding
        embedding_q = embedding_q.permute(1, 0, 2)  # NLD -> LND
        embedding_kv = kv.permute(1, 0, 2)  # NLD -> LND
        
        embedding_output, attn_weights = self.transformer(embedding_q, embedding_kv, extended_q_mask, extended_kv_mask)
        
        if embedding_output.dim()==3:
            embedding_output = embedding_output.permute(1, 0, 2)  # LND -> NLD
            pooled_output = self.pooler(embedding_output, hidden_mask=q_mask)  # [CLS] token (bs d)
        else:
            assert embedding_output.dim()==2, f"Expect dim=2, but got dim={embedding_output.dim()}"
            pooled_output = embedding_output
        #import pdb; pdb.set_trace()
        # attn_weights, (bs L S)
        return pooled_output, attn_weights

if __name__=="__main__":
    model = ResidualAttentionBlock(512, 8)
    q = torch.randn(2, 10, 128)
    kv = torch.randn(2, 20, 128)
    kv_mask = torch.zeros(2, 20)
    q_mask = torch.zeros(2, 10)
    out = model((q, kv, q_mask, kv_mask))