""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer

DeiT model defs and weights from https://github.com/facebookresearch/deit,
paper `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020 Ross Wightman
"""
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from collections import OrderedDict
from copy import deepcopy
from enum import Enum

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import build_model_with_cfg, overlay_external_default_cfg
from timm.models.layers import DropPath, to_2tuple, trunc_normal_, lecun_normal_
from timm.models.registry import register_model

_logger = logging.getLogger(__name__)
from . import layer_count


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'vit_base_patch16_224': _cfg(
        url='',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    )
}

class LSH:
    def __init__(self, n_tables, n_hyperplane, input_dim, threshold=0.85):
        self.n_tables = n_tables
        self.n_hyperplane = n_hyperplane
        self.init_hash_tables()
        self.hyperplanes = [np.random.randn(self.n_hyperplane, input_dim) for _ in range(self.n_tables)]
        self.hash_table_cache = {}
        self.threshold = threshold

    def init_hash_tables(self):
        self.raw_embeddings = [{} for _ in range(self.n_tables)]
        self.processed_embeddings = {}

    def reset_for_new_episode(self):
        self.init_hash_tables()

    def make_or_get_table(self, house_id):
        if house_id not in self.hash_table_cache:
            self.reset_for_new_episode()
            self.hash_table_cache[house_id] = (self.raw_embeddings, self.processed_embeddings)
        else:
            self.raw_embeddings, self.processed_embeddings = self.hash_table_cache[house_id]

    def hash(self, table_idx, vector):
        vector = vector.detach().cpu().numpy()
        sgn = self.hyperplanes[table_idx].dot(vector)
        return tuple((sgn > 0).astype(int))

    def add_view(self, panorama_id, token_index, raw_vector, processed_vector):        
        view_id = f"{panorama_id}_{token_index}"
        flattened = raw_vector.reshape(-1)
        for i in range(self.n_tables):
            table = self.raw_embeddings[i]
            hash_key = self.hash(i, flattened)

            if hash_key not in table:
                table[hash_key] = []

            table[hash_key].append((view_id, raw_vector))
            self.processed_embeddings[view_id] = processed_vector

    def get_similar_processed_embedding(self, raw):
        candidates = set()
        max_similarity = -1
        best_embedding = None
        best_view_id = None

        raw_reshaped = raw.reshape(-1)

        for i in range(self.n_tables):
            hash_key = self.hash(i, raw_reshaped)
            bucket = self.raw_embeddings[i].get(hash_key, [])
            for view_id, candidate_vector in bucket:
                candidates.add((view_id, candidate_vector))

        for view_id, candidate_vector in candidates:
            similarity = torch.cosine_similarity(F.normalize(raw.clone().contiguous().view(1, -1)),
                                    F.normalize(candidate_vector.clone().contiguous().view(1, -1)))  
                                
            if similarity > max_similarity:
                max_similarity = similarity
                best_embedding = self.processed_embeddings[view_id]
                best_view_id = view_id

        if max_similarity >= self.threshold:
            return best_embedding
        else:
            return None

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn # return attention


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()


        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    # return last layer attention
    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_grid = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.patch_grid[0] * self.patch_grid[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Mode(str, Enum):
    EFFICIENT = "efficient"
    BASELINE = "baseline"
    K_ONLY = "k_only"
    K_LSH = "k_lsh"
    K_EE = "k_ee"
    MUE = "mue"
    LSH_ONLY = "lsh_only"
    EE_ONLY = "ee_only"
    EE_LSH = "ee_lsh"


class VisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(
        self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
        num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None, distilled=False,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
        act_layer=None, weight_init='',
        mode=Mode.BASELINE, k_ext=4, ee_decay=9e-4, lsh_threshold=0.85
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()

        ### LSH ###
        self.lsh = LSH(n_tables=1, n_hyperplane=10, input_dim=3*224*224, threshold=lsh_threshold)
        self.instr_id_store = 0
        ### LSH ###

        ### brute force ###
        self.raw = {}
        self.processed = {}
        ### brute force ###

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule


        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        assert weight_init in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in weight_init else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if weight_init.startswith('jax'):
            # leave cls token as zeros to match jax impl
            for n, m in self.named_modules():
                _init_vit_weights(m, n, head_bias=head_bias, jax_impl=True)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)
        
        table = {
            Mode.EFFICIENT: self.forward_efficient,
            Mode.BASELINE: self.forward_baseline,
            Mode.K_ONLY: self.forward_k_only,
            Mode.K_LSH: self.forward_k_lsh,
            Mode.K_EE: self.forward_ee_only,
            Mode.MUE: self.forward_MuE,
            Mode.LSH_ONLY: self.forward_lsh_only,
            Mode.EE_ONLY: self.forward_ee_only,
            Mode.EE_LSH: self.forward_ee_lsh,
        }
        try:
            self._forward_impl = table[Mode(mode)]
        except KeyError:
            raise KeyError(f"Mode {mode} not recognized. Valid modes are {[m.value for m in Mode]}")

        self.k_ext = k_ext
        self.ee_decay = ee_decay

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def forward(self, x, house_id, nav_idx, viewpointId, instr_id, *_, **__):
        return self._forward_impl(x, house_id, nav_idx, viewpointId, instr_id)

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def _extend_nav_idx(self, nav_idx, k=4):
        assert k > 0 and k <= 36, "k should be between 1 and 36"
        extended_dict = {}
        for idx in nav_idx:
            for offset in range(-k, k+1):  # Extend by 4
                new_idx = idx + offset
                if 0 <= new_idx < 36:
                    extension_level = abs(offset)
                    if new_idx not in extended_dict or extension_level < extended_dict[new_idx]:
                        extended_dict[new_idx] = extension_level
        return extended_dict 
    
    def _prep_tokens(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)   # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1) if self.dist_token is None \
            else torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        return self.pos_drop(x + self.pos_embed)

    def _ee_thresholds(self, ee_decay=9e-4):
        # thresholds[rank] = exp(-ee_decay * rank)
        return {i: math.e ** (-ee_decay * i) for i in range(36)}

    def _run_blocks_with_ee(self, sb, exit_threshold):
        for li, block in enumerate(self.blocks):
            prev = sb
            sb = block(sb)
            sim = torch.cosine_similarity(
                F.normalize(sb.view(1, -1)),
                F.normalize(prev.view(1, -1))
            )

            if sim > exit_threshold or li == len(self.blocks) - 1:
                layer_count.global_layer_count[li] += 1
                return sb, li
        
        return sb, li

    def _check_lsh_for_reset(self, instr_id):
        if self.instr_id_store != instr_id:
            self.lsh.reset_for_new_episode()
        self.instr_id_store = instr_id

    def forward_baseline(self, x, house_id, nav_idx, viewpointId, instr_id):
        x = self._prep_tokens(x)
        x = self.blocks(x)
        x = self.norm(x)
        return self.pre_logits(x[:, 0]) if self.dist_token is None else (x[:, 0], x[:, 1])
    
    def forward_MuE(self, x, house_id, nav_idx, viewpointId, instr_id):
        x = self._prep_tokens(x)
        for idx, block in enumerate(self.blocks):
            prev = x
            x = block(x)
            sim = torch.cosine_similarity(
                F.normalize(x.view(1, -1)),
                F.normalize(prev.view(1, -1))
            )

            if sim > 0.99 or idx == len(self.blocks) - 1:
                layer_count.global_layer_count[idx] += 1
                break
        
        x = self.norm(x)
        return self.pre_logits(x[:, 0]) if self.dist_token is None else (x[:, 0], x[:, 1])

    def forward_efficient(self, x, house_id, nav_idx, viewpointId, instr_id):
        self._check_lsh_for_reset(instr_id)
        
        panorama = x.clone()
        x = self._prep_tokens(x)

        processed_x = torch.zeros_like(x)
        nav_idx_dict = self._extend_nav_idx(nav_idx, k=self.k_ext)
        thresholds = self._ee_thresholds(self.ee_decay)

        for i in range(36):
            sb = x[i:i+1, :, :]
            raw_vec = panorama[i]

            is_nav = i in nav_idx
            is_ext = i in nav_idx_dict

            if is_nav:
                sb = self.blocks(sb); li = 11
            elif is_ext:
                exit_thr = thresholds[nav_idx_dict[i]]
                cached = self.lsh.get_similar_processed_embedding(raw_vec)
                if cached is not None:
                    sb = cached; li = 0
                else:
                    sb, li = self._run_blocks_with_ee(sb, exit_threshold=exit_thr)
                    self.lsh.add_view(viewpointId, i, raw_vec, sb)
            else:
                cached = self.lsh.get_similar_processed_embedding(raw_vec)
                sb = cached if cached is not None else torch.zeros_like(sb)
                li = 0
            
            layer_count.global_layer_count[li] += 1
            processed_x[i:i+1, :, :] = sb
        
        x = self.norm(processed_x)
        return self.pre_logits(x[:, 0]) if self.dist_token is None else (x[:, 0], x[:, 1])

    def forward_k_only(self, x, house_id, nav_idx, viewpointId, instr_id):
        x = self._prep_tokens(x)

        processed_x = torch.zeros_like(x)
        nav_idx_dict = self._extend_nav_idx(nav_idx, k=self.k_ext)

        for i in range(36):
            sb = x[i:i+1, :, :]

            if i in nav_idx_dict:
                sb = self.blocks(sb)
                layer_count.global_layer_count[11] += 1
            else:
                sb = torch.zeros_like(sb)
                layer_count.global_layer_count[0] += 1
            
            processed_x[i:i+1, :, :] = sb
        
        x = self.norm(processed_x)
        return self.pre_logits(x[:, 0]) if self.dist_token is None else (x[:, 0], x[:, 1])

    def forward_ee_only(self, x, house_id, nav_idx, viewpointId, instr_id):
        x = self._prep_tokens(x)

        processed_x = torch.zeros_like(x)
        nav_idx_dict = self._extend_nav_idx(nav_idx, k=36)
        thresholds = self._ee_thresholds(self.ee_decay)

        for i in range(36):
            sb = x[i:i+1, :, :]

            exit_thr = thresholds[nav_idx_dict[i]]
            sb, li = self._run_blocks_with_ee(sb, exit_threshold=exit_thr)
            layer_count.global_layer_count[li] += 1
        
            processed_x[i:i+1, :, :] = sb
        
        x = self.norm(processed_x)
        return self.pre_logits(x[:, 0]) if self.dist_token is None else (x[:, 0], x[:, 1])

    def forward_lsh_only(self, x, house_id, nav_idx, viewpointId, instr_id):
        self._check_lsh_for_reset(instr_id)

        panorama = x.clone()
        x = self._prep_tokens(x)

        processed_x = torch.zeros_like(x)

        for i in range(36):
            if i in nav_idx:
                sb = x[i:i+1, :, :]
                sb = self.blocks(sb); li = 11
            else:
                raw_vec = panorama[i]
                cached = self.lsh.get_similar_processed_embedding(raw_vec)
                if cached is not None:
                    sb = cached; li = 0
                else:
                    sb = x[i:i+1, :, :]
                    sb = self.blocks(sb); li = 11
                    self.lsh.add_view(viewpointId, i, raw_vec, sb)
            
            layer_count.global_layer_count[li] += 1
            processed_x[i:i+1, :, :] = sb

        x = self.norm(processed_x)
        return self.pre_logits(x[:, 0]) if self.dist_token is None else (x[:, 0], x[:, 1])

    def forward_k_lsh(self, x, house_id, nav_idx, viewpointId, instr_id):
        self._check_lsh_for_reset(instr_id)

        panorama = x.clone()
        x = self._prep_tokens(x)

        processed_x = torch.zeros_like(x)
        nav_idx_dict = self._extend_nav_idx(nav_idx, k=self.k_ext)

        for i in range(36):
            sb = x[i:i+1, :, :]
            raw_vec = panorama[i]

            if i in nav_idx:
                sb = self.blocks(sb); li = 11
                self.lsh.add_view(viewpointId, i, panorama[i], sb)
            elif i in nav_idx_dict:
                cached = self.lsh.get_similar_processed_embedding(raw_vec)
                if cached is not None:
                    sb = cached; li = 0
                else:
                    sb = self.blocks(sb); li = 11
                    self.lsh.add_view(viewpointId, i, raw_vec, sb)
            else:
                cached = self.lsh.get_similar_processed_embedding(raw_vec)
                if cached is not None:
                    sb = cached; li = 0
                else:
                    sb = torch.zeros_like(sb); li = 0
                    
            layer_count.global_layer_count[li] += 1
            processed_x[i:i+1, :, :] = sb
        
        x = self.norm(processed_x)
        return self.pre_logits(x[:, 0]) if self.dist_token is None else (x[:, 0], x[:, 1])

    def forward_k_ee(self, x, house_id, nav_idx, viewpointId, instr_id):
        x = self._prep_tokens(x)

        processed_x = torch.zeros_like(x)
        nav_idx_dict = self._extend_nav_idx(nav_idx, k=self.k_ext)
        thresholds = self._ee_thresholds(self.ee_decay)

        for i in range(36):
            sb = x[i:i+1, :, :]

            if i in nav_idx_dict:
                exit_thr = thresholds[nav_idx_dict[i]]
                sb, li = self._run_blocks_with_ee(sb, exit_threshold=exit_thr)
            else:
                sb = torch.zeros_like(sb); li = 0
            
            layer_count.global_layer_count[li] += 1
            processed_x[i:i+1, :, :] = sb
        
        x = self.norm(processed_x)
        return self.pre_logits(x[:, 0]) if self.dist_token is None else (x[:, 0], x[:, 1])
    
    def forward_ee_lsh(self, x, house_id, nav_idx, viewpointId, instr_id):
        self._check_lsh_for_reset(instr_id)

        panorama = x.clone()
        x = self._prep_tokens(x)

        processed_x = torch.zeros_like(x)
        nav_idx_dict = self._extend_nav_idx(nav_idx, k=36)
        thresholds = self._ee_thresholds(self.ee_decay)

        for i in range(36):
            sb = x[i:i+1, :, :]
            raw_vec = panorama[i]

            if i in nav_idx:
                sb = self.blocks(sb); li = 11
                self.lsh.add_view(viewpointId, i, raw_vec, sb)
            else:
                exit_thr = thresholds[nav_idx_dict[i]]
                cached = self.lsh.get_similar_processed_embedding(raw_vec)
                if cached is not None:
                    sb = cached; li = 0
                else:
                    sb, li = self._run_blocks_with_ee(sb, exit_threshold=exit_thr)
                    self.lsh.add_view(viewpointId, i, raw_vec, sb)
            
            layer_count.global_layer_count[li] += 1
            processed_x[i:i+1, :, :] = sb
        
        x = self.norm(processed_x)
        return self.pre_logits(x[:, 0]) if self.dist_token is None else (x[:, 0], x[:, 1])
    
    ## Last self attention
    def get_last_selfattention(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        for idx, block in enumerate(self.blocks):
            if idx < len(self.blocks) - 1:
                prev_x = x
                x = block(prev_x)
                similarity = torch.cosine_similarity(F.normalize(x.clone().contiguous().view(1, -1)),
                                        F.normalize(prev_x.clone().contiguous().view(1, -1)))

                if similarity > 0.985:
                    print("idx: ", idx)
                    return block(prev_x, return_attention=True)

            else:
                # return attention of the last block
                return block(x, return_attention=True)


def _init_vit_weights(m, n: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(m, nn.Linear):
        if n.startswith('head'):
            nn.init.zeros_(m.weight)
            nn.init.constant_(m.bias, head_bias)
        elif n.startswith('pre_logits'):
            lecun_normal_(m.weight)
            nn.init.zeros_(m.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    if 'mlp' in n:
                        nn.init.normal_(m.bias, std=1e-6)
                    else:
                        nn.init.zeros_(m.bias)
            else:
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    elif jax_impl and isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def resize_pos_embed(posemb, posemb_new, num_tokens=1):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(v, model.pos_embed, getattr(model, 'num_tokens', 1))
        out_dict[k] = v
    return out_dict


def _create_vision_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    if default_cfg is None:
        default_cfg = deepcopy(default_cfgs[variant])
    overlay_external_default_cfg(default_cfg, kwargs)
    default_num_classes = default_cfg['num_classes']
    default_img_size = default_cfg['input_size'][-2:]

    num_classes = kwargs.pop('num_classes', default_num_classes)
    img_size = kwargs.pop('img_size', default_img_size)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        # Remove representation layer if fine-tuning. This may not always be the desired action,
        # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
        _logger.warning("Removing representation layer for fine-tuning.")
        repr_size = None

    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    vit_path = kwargs.pop('vit_path')
    model = build_model_with_cfg(
    VisionTransformer, variant, pretrained,
    default_cfg=default_cfg,
    img_size=img_size,
    num_classes=num_classes,
    representation_size=repr_size,
    pretrained_filter_fn=checkpoint_filter_fn,
    **kwargs
    )

    model.load_state_dict(torch.load(vit_path, map_location=lambda storage, loc: storage))
    model.eval()

    return model


@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model