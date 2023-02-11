# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2020 Ross Wightman
# Modified Model definition

import torch
import torch.nn as nn
from functools import partial
import math
import warnings
import torch.nn.functional as F
import numpy as np
import copy

from imtt.models.vit_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from imtt.models.helpers import load_pretrained
from imtt.models.vit_utils import DropPath, to_2tuple, trunc_normal_
from timm.models.layers import trunc_normal_ as img_trunc_normal_

from .build import MODEL_REGISTRY
from torch import einsum
from einops import rearrange, reduce, repeat
from .transformer_block import Block as Images_Block
from timm.models.vision_transformer import VisionTransformer as Images_VIT, _cfg
from .clip import LayerNorm, Transformer
# from .layers_ours import *
from .resPromptVit import resPromptVisionTransformer
from .basicVit import BasicVIT


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}

def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    # all_layer_matrices = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
    #                       for i in range(len(all_layer_matrices))]
    joint_attention = all_layer_matrices[start_layer]
    for i in range(start_layer+1, len(all_layer_matrices)):
        joint_attention = all_layer_matrices[i].bmm(joint_attention)
    return joint_attention

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
    
    def relprop(self, cam, **kwargs):
        cam = self.drop.relprop(cam, **kwargs)
        cam = self.fc2.relprop(cam, **kwargs)
        cam = self.act.relprop(cam, **kwargs)
        cam = self.fc1.relprop(cam, **kwargs)
        return cam

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        self.attn_cam = None
        self.attn = None
        self.v = None
        self.v_cam = None
        self.attn_gradients = None

    def get_attn(self):
        return self.attn

    def save_attn(self, attn):
        self.attn = attn

    def save_attn_cam(self, cam):
        self.attn_cam = cam

    def get_attn_cam(self):
        return self.attn_cam

    def get_v(self):
        return self.v

    def save_v(self, v):
        self.v = v

    def save_v_cam(self, cam):
        self.v_cam = cam

    def get_v_cam(self):
        return self.v_cam

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
           q, k, v = qkv[0], qkv[1], qkv[2]
        else:
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v  = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x
    
    def relprop(self, cam, **kwargs):
        cam = self.proj_drop.relprop(cam, **kwargs)
        cam = self.proj.relprop(cam, **kwargs)
        cam = rearrange(cam, 'b n (h d) -> b h n d', h=self.num_heads)

        # attn = A*V
        (cam1, cam_v)= self.matmul2.relprop(cam, **kwargs)
        cam1 /= 2
        cam_v /= 2

        self.save_v_cam(cam_v)
        self.save_attn_cam(cam1)

        cam1 = self.attn_drop.relprop(cam1, **kwargs)
        cam1 = self.softmax.relprop(cam1, **kwargs)

        # A = Q*K^T
        (cam_q, cam_k) = self.matmul1.relprop(cam1, **kwargs)
        cam_q /= 2
        cam_k /= 2

        cam_qkv = rearrange([cam_q, cam_k, cam_v], 'qkv b h n d -> b n (qkv h d)', qkv=3, h=self.num_heads)

        return self.qkv.relprop(cam_qkv, **kwargs)

class Block(nn.Module):

    def __init__(self,extra_tokens,dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type='divided_space_time'):
        super().__init__()
        self.attention_type = attention_type
        assert(attention_type in ['divided_space_time', 'space_only','joint_space_time'])

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
           dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        ## Temporal Attention Parameters
        if self.attention_type == 'divided_space_time':
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn = Attention(
              dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.temporal_fc = nn.Linear(dim, dim)

        ## drop path
        self.drop_path = nn.DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        ## view token
        self.extra_tokens = extra_tokens

        self.add1 = Add()
        self.add2 = Add()
        self.clone1 = Clone()
        self.clone2 = Clone()


    def forward(self, x, B, T, W,is_cls=True):
        if not is_cls: num_spatial_tokens = x.size(1) // T
        else: num_spatial_tokens = (x.size(1) - (1 + self.extra_tokens)) // T
        H = num_spatial_tokens // W

        if self.attention_type in ['space_only', 'joint_space_time']:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        elif self.attention_type == 'divided_space_time':
            ## Temporal
            if is_cls: xt = x[:,(1 + self.extra_tokens):,:]
            else: xt = x[:,:,:]
            xt = rearrange(xt, 'b (h w t) m -> (b h w) t m',b=B,h=H,w=W,t=T)
            res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
            res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m',b=B,h=H,w=W,t=T)
            res_temporal = self.temporal_fc(res_temporal)
            if is_cls: xt = x[:,(1 + self.extra_tokens):,:] + res_temporal
            else: xt = x[:,:,:] + res_temporal

            ## Spatial
            if is_cls:
                init_cls_token = x[:,self.extra_tokens,:].unsqueeze(1)
                init_view_token = x[:,0,:].unsqueeze(1)
                cls_token = init_cls_token.repeat(1, T, 1)
                view_token = init_view_token.repeat(1,T,1)
                cls_token = rearrange(cls_token, 'b t m -> (b t) m',b=B,t=T).unsqueeze(1)
                view_token = rearrange(view_token, 'b t m -> (b t) m',b=B,t=T).unsqueeze(1)
            xs = xt
            xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m',b=B,h=H,w=W,t=T)
            if is_cls:
                if self.extra_tokens == 1:
                    xs = torch.cat((view_token,cls_token, xs), 1)
                else:
                    xs = torch.cat((cls_token, xs), 1)
            res_spatial = self.drop_path(self.attn(self.norm1(xs)))

            if is_cls:
                ### Taking care of CLS token
                cls_token = res_spatial[:,self.extra_tokens,:]
                view_token = res_spatial[:,0,:]
                cls_token = rearrange(cls_token, '(b t) m -> b t m',b=B,t=T)
                view_token = rearrange(view_token, '(b t) m -> b t m',b=B,t=T)
                cls_token = torch.mean(cls_token,1,True) ## averaging for every frame
                view_token = torch.mean(view_token,1,True) ## averaging for every frame
                res_spatial = res_spatial[:,(1 + self.extra_tokens):,:]
                
            res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m',b=B,h=H,w=W,t=T)
            res = res_spatial
            x = xt

            ## Mlp
            if is_cls:
                if self.extra_tokens == 1:
                    x = torch.cat((init_view_token,init_cls_token, x), 1) + torch.cat((view_token,cls_token, res), 1)
                else:
                    x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
            else: x = x + res
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
    
    def relprop(self, cam, **kwargs):
        (cam1, cam2) = self.add2.relprop(cam, **kwargs)
        cam2 = self.mlp.relprop(cam2, **kwargs)
        cam2 = self.norm2.relprop(cam2, **kwargs)
        cam = self.clone2.relprop((cam1, cam2), **kwargs)

        (cam1, cam2) = self.add1.relprop(cam, **kwargs)
        cam2 = self.attn.relprop(cam2, **kwargs)
        cam2 = self.norm1.relprop(cam2, **kwargs)
        cam = self.clone1.relprop((cam1, cam2), **kwargs)
        return cam

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.proj(x)
        W = x.size(-1)
        x = x.flatten(2).transpose(1, 2)
        return x, T, W

    def relprop(self, cam, **kwargs):
        cam = cam.transpose(1,2)
        cam = cam.reshape(cam.shape[0], cam.shape[1],
                     (self.img_size[0] // self.patch_size[0]), (self.img_size[1] // self.patch_size[1]))
        return self.proj.relprop(cam, **kwargs)


class VisionTransformer(nn.Module):
    """ Vision Transformer
    """
    def __init__(self, extra_tokens=0, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, hybrid_backbone=None, norm_layer=nn.LayerNorm, num_frames=8, attention_type='divided_space_time', dropout=0.):
        super().__init__()
        print("Model has ", extra_tokens, " extra token(s)")
        self.attention_type = attention_type
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.extra_tokens = extra_tokens

        ## Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        if self.attention_type != 'space_only':
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            self.time_drop = nn.Dropout(p=drop_rate)
            
        ## View Token
        self.view_token = nn.Parameter(torch.zeros(1,1,embed_dim))

        ## Attention Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                extra_tokens=self.extra_tokens, dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, attention_type=self.attention_type)
            for i in range(self.depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        self.view_head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.view_token, std=.02)
        self.apply(self._init_weights)

        ## initialization of temporal attention weights
        if self.attention_type == 'divided_space_time':
            i = 0
            for m in self.blocks.modules():
                m_str = str(m)
                if 'Block' in m_str:
                    if i > 0:
                      nn.init.constant_(m.temporal_fc.weight, 0)
                      nn.init.constant_(m.temporal_fc.bias, 0)
                    i += 1

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        if self.training:
            return self.view_head
        else:
            return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        if self.training:
            self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        else:
            self.view_head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x, T, W = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        view_tokens = self.view_token.expand(x.size(0),-1,-1)
        x = torch.cat((cls_tokens, x), dim=1)

        ## resizing the positional embeddings in case they don't match the input at inference
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed
            cls_pos_embed = pos_embed[0,0,:].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0,1:,:].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed
        
        x = self.pos_drop(x)

        # print(x[0][1])
        ## Time Embeddings
        if self.attention_type != 'space_only':
            cls_tokens = x[:B, 0, :].unsqueeze(1)
            view_tokens = view_tokens[:B,:,:]
            x = x[:,1:]
            x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)
            ## Resizing time embeddings in case they don't match
            if T != self.time_embed.size(1):
                time_embed = self.time_embed.transpose(1, 2)
                new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
                new_time_embed = new_time_embed.transpose(1, 2)
                x = x + new_time_embed
            else:
                x = x + self.time_embed
            x = self.time_drop(x)
            x = rearrange(x, '(b n) t m -> b (n t) m',b=B,t=T)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.extra_tokens == 1:
            x = torch.cat((view_tokens, x), dim=1)
    
        ## Attention blocks
        for i,blk in enumerate(self.blocks):
            x = blk(x, B, T, W)


        ### Predictions for space-only baseline
        if self.attention_type == 'space_only':
            x = rearrange(x, '(b t) n m -> b t n m',b=B,t=T)
            x = torch.mean(x, 1) # averaging predictions for every frame

        x = self.norm(x)
        if self.training:
            return x[:, 0]
        elif self.extra_tokens == 1:
            return x[:,1]
        else:
            return x[:,0]
            

    def forward(self, x):
        x = self.forward_features(x)
        if self.training and self.extra_tokens == 1:
            x = self.view_head(x)
        else:
            x = self.head(x)
        return x

# class resPromptVisionTransformer(Images_VIT):
#     def __init__(self, *args,actual_num_classes=400,qk_scale=None, num_prompts=1, num_frames=8, img_size=224, attention_type='divided_space_time', **kwargs):
#         super().__init__(*args,**kwargs)

#         self.patch_size = kwargs['patch_size']
#         embed_dim = kwargs['embed_dim']
#         self.depth = kwargs['depth']
#         num_heads = kwargs['num_heads']
#         mlp_ratio = kwargs['mlp_ratio']
#         qkv_bias = kwargs['qkv_bias']
#         norm_layer = kwargs['norm_layer']
#         num_classes = actual_num_classes
#         drop_rate = kwargs['drop_rate']
#         drop_path_rate = kwargs['drop_path_rate']
#         attn_drop_rate = kwargs['attn_drop_rate']
#         self.num_prompts = num_prompts
#         self.attention_type = attention_type
#         num_frames = num_frames
#         img_size = img_size

#         act_layer = None
#         act_layer = act_layer or nn.GELU

#         self.patch_embed = PatchEmbed(
#             img_size=img_size, patch_size=self.patch_size, in_chans=3, embed_dim=embed_dim)

#         if self.attention_type != 'space_only':
#             self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
#             self.time_drop = nn.Dropout(p=drop_rate)

#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule

#         self.blocks = nn.Sequential(*[
#             Images_Block(
#                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
#                 attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
#             for i in range(self.depth)])

#         self.transformation = Block(extra_tokens=0,dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,attention_type=self.attention_type)
#         self.resPrompt_token = nn.Parameter(torch.zeros(1, self.num_prompts, self.embed_dim))
#         self.head_resPrompt = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

#         ## initialize the temporal attention weights
#         # if self.attention_type == 'divided_space_time':
#         #     nn.init.constant_(self.transformation.temporal_fc.weight, 0)
#         #     nn.init.constant_(self.transformation.temporal_fc.bias, 0)


#         trunc_normal_(self.resPrompt_token, std=.02)
#         self.head_resPrompt.apply(self._init_weights)

#         self.pool = IndexSelect()
#         self.add = Add()

#         self.inp_grad = None

#     # def _init_weights(self, m):
#     #     if isinstance(m, nn.Linear):
#     #         trunc_normal_(m.weight, std=.02)
#     #         if isinstance(m, nn.Linear) and m.bias is not None:
#     #             nn.init.constant_(m.bias, 0)
#     #     elif isinstance(m, nn.LayerNorm):
#     #         nn.init.constant_(m.bias, 0)
#     #         nn.init.constant_(m.weight, 1.0)

#     def save_inp_grad(self,grad):
#         self.inp_grad = grad

#     def get_inp_grad(self):
#         return self.inp_grad

#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'pos_embed', 'cls_token', 'time_embed'}

#     def interpolate_pos_encoding(self, x, w, h):
#         npatch = x.shape[1] - 1
#         N = self.pos_embed.shape[1] - 1
#         if npatch == N and w == h:
#             return self.pos_embed
#         class_pos_embed = self.pos_embed[:, 0]
#         patch_pos_embed = self.pos_embed[:, 1:]
#         dim = x.shape[-1]
#         w0 = w // self.patch_embed.patch_size
#         h0 = h // self.patch_embed.patch_size
#         # we add a small number to avoid floating point error in the interpolation
#         # see discussion at https://github.com/facebookresearch/dino/issues/8
#         w0, h0 = w0 + 0.1, h0 + 0.1
#         patch_pos_embed = nn.functional.interpolate(
#             patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
#             scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
#             mode='bicubic',
#         )
#         assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
#         patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
#         return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

#     def time_embedding(self,x,B,T):
#         cls_tokens = x[:B, 0, :].unsqueeze(1)
#         x = x[:,1:]
#         x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)
#         ## Resizing time embeddings in case they don't match
#         if T != self.time_embed.size(1):
#             time_embed = self.time_embed.transpose(1, 2)
#             new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
#             new_time_embed = new_time_embed.transpose(1, 2)
#             x = x + new_time_embed
#         else:
#             x = x + self.time_embed
#         x = self.time_drop(x)
#         x = rearrange(x, '(b n) t m -> b (n t) m',b=B,t=T)
#         x = torch.cat((cls_tokens, x), dim=1)
#         return x

#     def spatial_embedding(self,x,W):
#         ## resizing the positional embeddings in case they don't match the input at inference
#         if x.size(1) != self.pos_embed.size(1):
#             pos_embed = self.pos_embed
#             cls_pos_embed = pos_embed[0,0,:].unsqueeze(0).unsqueeze(1)
#             other_pos_embed = pos_embed[0,1:,:].unsqueeze(0).transpose(1, 2)
#             P = int(other_pos_embed.size(2) ** 0.5)
#             H = x.size(1) // W
#             other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
#             new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='bicubic')
#             new_pos_embed = new_pos_embed.flatten(2)
#             new_pos_embed = new_pos_embed.transpose(1, 2)
#             new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
#             x = x + new_pos_embed
#         else:
#             x = x + self.pos_embed
        
#         x = self.pos_drop(x)
#         return x

#     def forward_features(self, x):
#         if len(list(x.shape)) == 4:
#             x = x.unsqueeze(2)
#         B = x.shape[0]
#         x,T,W = self.patch_embed(x)

#         cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
#         # x = torch.cat((cls_tokens,x),dim=1)

#         ## transformation
#         x = rearrange(x[:,:,:], '(b t) n m -> b (n t) m',t=T)
#         transformation = self.transformation(x[:,:,:],B,T,W,is_cls=False).mean(dim=1, keepdim=True)
#         x = rearrange(x[:,:,:], 'b (n t) m -> (b t) n m',t=T)
#         x = torch.cat((cls_tokens,x),dim=1)

#         ## Spatial embedding
#         x = self.spatial_embedding(x,W)

#         ## Time Embeddings
#         if self.attention_type != 'space_only':
#             x = self.time_embedding(x,B,T)

#         ## prompt token
#         resPrompt_token = self.resPrompt_token.expand(B, -1, -1) + transformation
#         x = torch.cat((resPrompt_token, x), dim=1)

#         layer_wise_tokens = []
#         attention_maps = []
#         for i,blk in enumerate(self.blocks):
#             x, attn = blk(x, return_attention=True)
#             layer_wise_tokens.append(x)
#             attention_maps.append(attn)

#         return layer_wise_tokens, attention_maps

#     def forward(self, x, all_tokens=True):
#         size = x.size(0)
#         layer_wise_tokens, attention_maps = self.forward_features(x)
#         layer_wise_tokens_norm = [self.norm(x) for x in layer_wise_tokens]
#         x = [self.head(x[:, self.num_prompts]) for x in layer_wise_tokens_norm]
#         x_resPrompt = [self.head_resPrompt(x[:, 0:self.num_prompts].mean(dim=1)) for x in layer_wise_tokens_norm]
#         return x_resPrompt, layer_wise_tokens, attention_maps  # , x
    
    

class timeAvgVisionTransformer(Images_VIT):
    def __init__(self, *args,actual_num_classes=400,qk_scale=None, num_prompts=1, num_frames=8, img_size=224, attention_type='divided_space_time', **kwargs):
        super().__init__(*args,**kwargs)

        self.patch_size = kwargs['patch_size']
        embed_dim = kwargs['embed_dim']
        self.depth = kwargs['depth']
        num_heads = kwargs['num_heads']
        mlp_ratio = kwargs['mlp_ratio']
        qkv_bias = kwargs['qkv_bias']
        norm_layer = kwargs['norm_layer']
        num_classes = actual_num_classes
        drop_rate = kwargs['drop_rate']
        drop_path_rate = kwargs['drop_path_rate']
        attn_drop_rate = kwargs['attn_drop_rate']
        self.num_prompts = num_prompts
        self.attention_type = attention_type
        num_frames = num_frames
        img_size = img_size

        act_layer = None
        act_layer = act_layer or nn.GELU

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=self.patch_size, in_chans=3, embed_dim=embed_dim)

        if self.attention_type != 'space_only':
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            self.time_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule

        self.blocks = nn.Sequential(*[
            Images_Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(self.depth)])

        self.transformation = Block(extra_tokens=0,dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,attention_type=self.attention_type)
        self.resPrompt_token = nn.Parameter(torch.zeros(1, self.num_prompts, self.embed_dim))
        self.head_new = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        ## initialize the temporal attention weights
        # if self.attention_type == 'divided_space_time':
        #     nn.init.constant_(self.transformation.temporal_fc.weight, 0)
        #     nn.init.constant_(self.transformation.temporal_fc.bias, 0)


        trunc_normal_(self.resPrompt_token, std=.02)
        self.head_new.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def time_embedding(self,x,B,T):
        # cls_tokens = x[:B, 0, :].unsqueeze(1)
        # x = x[:,1:]
        x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)
        ## Resizing time embeddings in case they don't match
        if T != self.time_embed.size(1):
            time_embed = self.time_embed.transpose(1, 2)
            new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
            new_time_embed = new_time_embed.transpose(1, 2)
            x = x + new_time_embed
        else:
            x = x + self.time_embed
        x = self.time_drop(x)
        x = rearrange(x, '(b n) t m -> b (n t) m',b=B,t=T)
        return x

    def forward_features(self, x):
        B = x.shape[0]
        x,T,W = self.patch_embed(x)

        # cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        # x = torch.cat((cls_tokens,x),dim=1)

        ## Time Embeddings
        if self.attention_type != 'space_only':
            x = self.time_embedding(x,B,T)

        
        x = self.transformation(x,B,T,W,is_cls=False)
        # x_cls = x[:,0,:].unsqueeze(1)
        # x = rearrange(x, 'b (n t) m -> b n t m', t=T).mean(dim=2, keepdim=False)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # x = x + self.interpolate_pos_encoding(x, w, h)
        ## resizing the positional embeddings in case they don't match the input at inference
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed
            cls_pos_embed = pos_embed[0,0,:].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0,1:,:].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed
        
        x = self.pos_drop(x)


        resPrompt_token = self.resPrompt_token.expand(B, -1, -1)
        x = torch.cat((resPrompt_token, x), dim=1)

        # x = self.pos_drop(x)

        layer_wise_tokens = []
        attention_maps = []
        for i,blk in enumerate(self.blocks):
            x, attn = blk(x, return_attention=True)
            layer_wise_tokens.append(x)
            attention_maps.append(attn)

        # return layer_wise_tokens, attention_maps
        x = self.norm(x)
        return x[:,0:self.num_prompts], x[:,self.num_prompts]

    def forward(self, x, all_tokens=True):
        # size = x.size(0)
        # layer_wise_tokens, attention_maps = self.forward_features(x)
        # layer_wise_tokens_norm = [self.norm(x) for x in layer_wise_tokens]
        # x = [self.head(x[:, self.num_prompts]) for x in layer_wise_tokens_norm]
        # x_resPrompt = [self.head_resPrompt(x[:, 0:self.num_prompts].mean(dim=1)) for x in layer_wise_tokens_norm]
        # return x_resPrompt, layer_wise_tokens, attention_maps  # , x
        size = x.size(0)
        x_resPrompt,x_cls = self.forward_features(x)
        x_resPrompt = x_resPrompt.mean(dim=1)
        x_final = self.head_new(x_resPrompt)
        return x_final


class BaselineVisionTransformer(Images_VIT):
    def __init__(self, *args,actual_num_classes=400,qk_scale=None, num_prompts=1, num_frames=8, img_size=224, attention_type='divided_space_time', **kwargs):
        super().__init__(*args,**kwargs)

        self.patch_size = kwargs['patch_size']
        embed_dim = kwargs['embed_dim']
        self.depth = kwargs['depth']
        num_heads = kwargs['num_heads']
        mlp_ratio = kwargs['mlp_ratio']
        qkv_bias = kwargs['qkv_bias']
        norm_layer = kwargs['norm_layer']
        num_classes = actual_num_classes
        drop_rate = kwargs['drop_rate']
        drop_path_rate = kwargs['drop_path_rate']
        attn_drop_rate = kwargs['attn_drop_rate']
        num_frames = num_frames
        img_size = img_size

        act_layer = None
        act_layer = act_layer or nn.GELU

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=self.patch_size, in_chans=3, embed_dim=embed_dim)


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule

        self.blocks = nn.Sequential(*[
            Images_Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(self.depth)])

        self.new_head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.new_head.apply(self._init_weights)


    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward_features(self, x):
        B = x.shape[0]
        x,T,W = self.patch_embed(x)

        # cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        # x = torch.cat((cls_tokens,x),dim=1)
        x = rearrange(x, '(b t) n m -> b (n t) m',b=B,t=T)

        # x_cls = x[:,0,:].unsqueeze(1)
        # x = rearrange(x[:,1:,:], 'b (n t) m -> b n t m', t=T).mean(dim=2, keepdim=False)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # x = x + self.interpolate_pos_encoding(x, w, h)
        ## resizing the positional embeddings in case they don't match the input at inference
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed
            cls_pos_embed = pos_embed[0,0,:].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0,1:,:].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed
        
        x = self.pos_drop(x)


        layer_wise_tokens = []
        attention_maps = []
        for blk in self.blocks:
            x, attn = blk(x, return_attention=True)
            layer_wise_tokens.append(x)
            attention_maps.append(attn)

        x = self.norm(x)
        return x[:,0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.new_head(x)
        return x

class resPromptDino(Images_VIT):
    def __init__(self, *args,actual_num_classes=400,qk_scale=None, num_prompts=1, num_frames=8, img_size=224, attention_type='divided_space_time', **kwargs):
        super().__init__(*args,**kwargs)

        self.patch_size = kwargs['patch_size']
        embed_dim = kwargs['embed_dim']
        self.depth = kwargs['depth']
        num_heads = kwargs['num_heads']
        mlp_ratio = kwargs['mlp_ratio']
        qkv_bias = kwargs['qkv_bias']
        norm_layer = kwargs['norm_layer']
        num_classes = actual_num_classes
        drop_rate = kwargs['drop_rate']
        drop_path_rate = kwargs['drop_path_rate']
        attn_drop_rate = kwargs['attn_drop_rate']
        self.num_prompts = num_prompts
        self.attention_type = attention_type
        num_frames = num_frames
        img_size = img_size

        act_layer = None
        act_layer = act_layer or nn.GELU

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=self.patch_size, in_chans=3, embed_dim=embed_dim)

        self.head = nn.Identity()
        self.linear = nn.Linear(embed_dim*2, self.num_classes)

        if self.attention_type != 'space_only':
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            self.time_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule

        self.blocks = nn.Sequential(*[
            Images_Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(self.depth)])

        self.transformation = Block(extra_tokens=0,dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,attention_type=self.attention_type)
        self.resPrompt_token = nn.Parameter(torch.zeros(1, self.num_prompts, embed_dim))
        self.head_resPrompt = nn.Linear(embed_dim*2, num_classes) if num_classes > 0 else nn.Identity()

        ## initialize the temporal attention weights
        # if self.attention_type == 'divided_space_time':
        #     nn.init.constant_(self.transformation.temporal_fc.weight, 0)
        #     nn.init.constant_(self.transformation.temporal_fc.bias, 0)


        trunc_normal_(self.resPrompt_token, std=.02)
        self.head_resPrompt.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def time_embedding(self,x,B,T):
        cls_tokens = x[:B, 0, :].unsqueeze(1)
        x = x[:,1:]
        x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)
        ## Resizing time embeddings in case they don't match
        if T != self.time_embed.size(1):
            time_embed = self.time_embed.transpose(1, 2)
            new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
            new_time_embed = new_time_embed.transpose(1, 2)
            x = x + new_time_embed
        else:
            x = x + self.time_embed
        x = self.time_drop(x)
        x = rearrange(x, '(b n) t m -> b (n t) m',b=B,t=T)
        x = torch.cat((cls_tokens, x), dim=1)
        return x

    def spatial_embedding(self,x,W):
        ## resizing the positional embeddings in case they don't match the input at inference
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed
            cls_pos_embed = pos_embed[0,0,:].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0,1:,:].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='bicubic')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed
        
        x = self.pos_drop(x)
        return x

    def forward_features(self, x):
        B = x.shape[0]
        x,T,W = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        # x = torch.cat((cls_tokens,x),dim=1)

        ## transformation
        x = rearrange(x[:,:,:], '(b t) n m -> b (n t) m',t=T)
        transformation = self.transformation(x[:,:,:],B,T,W,is_cls=False).mean(dim=1, keepdim=True)
        x = rearrange(x[:,:,:], 'b (n t) m -> (b t) n m',t=T)
        x = torch.cat((cls_tokens,x),dim=1)

        ## Spatial embedding
        x = self.spatial_embedding(x,W)

        ## Time Embeddings
        if self.attention_type != 'space_only':
            x = self.time_embedding(x,B,T)


        resPrompt_token = self.resPrompt_token.expand(B, -1, -1) + transformation
        # resPrompt_token = self.resPrompt_token.expand(B, -1, -1)
        x = torch.cat((resPrompt_token, x), dim=1)

        layer_wise_tokens = []
        attention_maps = []
        for i,blk in enumerate(self.blocks):
            x, attn = blk(x, return_attention=True)
            layer_wise_tokens.append(x)
            attention_maps.append(attn)

        # return layer_wise_tokens, attention_maps
        x = self.norm(x)
        return x[:, 0:self.num_prompts], x[:, self.num_prompts], x[:, self.num_prompts+1:]

    def forward(self, x, all_tokens=True):
        size = x.size(0)

        x_resPrompt, x_cls, x_patches = self.forward_features(x)

        x_cls = torch.cat((x_cls.unsqueeze(-1), torch.mean(x_patches, dim=1).unsqueeze(-1)), dim=-1)
        x_cls = x_cls.reshape(size, -1)
        x_cls = self.linear(x_cls)

        x_resPrompt = x_resPrompt.mean(dim=1)
        x_resPrompt = torch.cat((x_resPrompt.unsqueeze(-1), torch.mean(x_patches, dim=1).unsqueeze(-1)), dim=-1)
        x_resPrompt = x_resPrompt.reshape(size, -1)
        x_resPrompt = self.head_resPrompt(x_resPrompt)
        return x_resPrompt, x_cls   # , x

class resPromptClip(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, num_prompts: int, num_classes:int, num_frames: int , actual_num_classes: int, attention_type, qk_scale=None):
        super().__init__()
        self.num_prompts = num_prompts
        self.num_classes = num_classes
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.width = width
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        drop_path_rate = 0.1
        self.attention_type = attention_type

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate , 12)] 

        self.transformation = Block(extra_tokens=0,dim=self.width, num_heads=heads, mlp_ratio=4, qkv_bias=True, qk_scale=qk_scale, drop=0., attn_drop=0., drop_path=dpr[0], norm_layer=partial(nn.LayerNorm, eps=1e-6),attention_type=self.attention_type) 
        self.resPrompt_token = nn.Parameter(torch.zeros(1, self.num_prompts, self.width))
        self.head_resPrompt = nn.Linear(self.width, actual_num_classes) if actual_num_classes > 0 else nn.Identity()

        if self.attention_type != 'space_only':
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, width))
            self.time_drop = nn.Dropout(p=0.)

        trunc_normal_(self.resPrompt_token, std=0.02)

    def time_embedding(self,x,B,T):
        cls_tokens = x[:B, 0, :].unsqueeze(1)
        x = x[:,1:]
        x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)
        ## Resizing time embeddings in case they don't match
        if T != self.time_embed.size(1):
            time_embed = self.time_embed.transpose(1, 2)
            new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
            new_time_embed = new_time_embed.transpose(1, 2)
            x = x + new_time_embed
        else:
            x = x + self.time_embed
        x = self.time_drop(x)
        x = rearrange(x, '(b n) t m -> b (n t) m',b=B,t=T)
        x = torch.cat((cls_tokens, x), dim=1)
        return x
    
    def forward(self,x: torch.Tensor, text_project=False):
        B,C,T,H,W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.conv1(x)
        W = x.size(-1)
        x = x.flatten(2).transpose(1,2)

        # x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        x = rearrange(x[:,:,:], '(b t) n m -> b (n t) m',t=T)
        transform = self.transformation(x[:,:,:],B,T,W,is_cls=False).mean(dim=1, keepdim=True)
        x = rearrange(x[:,:,:], 'b (n t) m -> (b t) n m',t=T)

        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        
        if x.size(1) != self.positional_embedding.size(1):
            pos_embed = self.positional_embedding
            cls_pos_embed = pos_embed[0,:].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[1:,:].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            x = x + self.positional_embedding

        ## Time embeddings
        if self.attention_type != 'space_only':
            x = self.time_embedding(x,B,T)

        resPrompt_token = self.resPrompt_token.expand(B, -1, -1) + transform
        x = torch.cat((resPrompt_token, x), dim=1)
        
        x = self.ln_pre(x)

        x = x.permute(1,0,2)
        x = self.transformer(x)
        x = x.permute(1,0,2)

        x_resPrompt, x = x[:, 0:self.num_prompts].mean(dim=1), x[:, self.num_prompts]

        x_resPrompt, x = self.ln_post(x_resPrompt), self.ln_post(x)
        if self.proj is not None and text_project:
            # projects prompts
            x_resPrompt = x_resPrompt @ self.proj
            x = x @ self.proj
        
        x_resPrompt = self.head_resPrompt(x_resPrompt)

        return x_resPrompt, x

def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            if v.shape[-1] != patch_size:
                patch_size = v.shape[-1]
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict

@MODEL_REGISTRY.register()
class vit_1p(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=400, num_frames=8, attention_type='divided_space_time',  pretrained_model='', **kwargs):
        super(vit_1p, self).__init__()
        self.pretrained=True
        patch_size = 16
        self.model = resPromptVisionTransformer(img_size=img_size,num_classes=1000, actual_num_classes=num_classes, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=num_frames, attention_type=attention_type, **kwargs)

        self.attention_type = attention_type
        self.model.default_cfg = default_cfgs['vit_base_patch16_224']
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        pretrained_model=pretrained_model
        if self.pretrained:
            # checkpoint = torch.hub.load_state_dict_from_url (
            #     url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth",
            #     map_location="cpu", check_hash=True
            # )
            # self.model.load_state_dict(checkpoint,strict=False)
            checkpoint = torch.load(pretrained_model)["model_state"]
            changed_ckpt = {}
            for key, value in checkpoint.items():
                changed_ckpt[key[6:]] = value
            self.model.load_state_dict(changed_ckpt,strict=True)
        
        


    def forward(self, x):
        x_resPrompt, layer_wise_tokens, attention_maps, is_image = self.model(x)
        return x_resPrompt[-1]

    def relprop(self,x,method="transformer_attribution", is_ablation=False, start_layer=0,num_prompts=1, is_image=False, **kwargs ):
        return self.model.relprop(x,method=method, is_ablation=is_ablation, start_layer=start_layer, num_prompts=num_prompts, is_image=is_image, **kwargs)

class vit_10p(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=400, num_frames=8, attention_type='divided_space_time',  pretrained_model='', **kwargs):
        super(vit_10p, self).__init__()
        self.pretrained=True
        patch_size = 16
        self.model = resPromptVisionTransformer(img_size=img_size,num_classes=1000, actual_num_classes=num_classes, num_prompts=10, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=num_frames, attention_type=attention_type, **kwargs)

        self.attention_type = attention_type
        self.model.default_cfg = default_cfgs['vit_base_patch16_224']
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        pretrained_model=pretrained_model
        if self.pretrained:
            # checkpoint = torch.hub.load_state_dict_from_url (
            #     url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth",
            #     map_location="cpu", check_hash=True
            # )
            # self.model.load_state_dict(checkpoint,strict=False)
            checkpoint = torch.load(pretrained_model)["model_state"]
            changed_ckpt = {}
            for key, value in checkpoint.items():
                changed_ckpt[key[6:]] = value
            self.model.load_state_dict(changed_ckpt,strict=True)
        
        


    def forward(self, x):
        x_resPrompt, layer_wise_tokens, attention_maps, is_image = self.model(x)
        return x_resPrompt[-1], is_image

    def relprop(self,x,method="transformer_attribution", is_ablation=False, start_layer=0,num_prompts=1, which_prompt=0, is_image=False, **kwargs ):
        return self.model.relprop(x,method=method, is_ablation=is_ablation, start_layer=start_layer, num_prompts=num_prompts, is_image=is_image, which_prompt=which_prompt, **kwargs)
    

@MODEL_REGISTRY.register()
class vit_basic(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=400, num_frames=8, attention_type='divided_space_time',  pretrained_model='', **kwargs):
        super(vit_basic, self).__init__()
        self.pretrained=True
        patch_size = 16
        self.model = BasicVIT(img_size=img_size,num_classes=1000, actual_num_classes=num_classes, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=num_frames, attention_type=attention_type, **kwargs)

        self.attention_type = attention_type
        self.model.default_cfg = default_cfgs['vit_base_patch16_224']
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        pretrained_model=pretrained_model
        if self.pretrained:
            checkpoint = torch.hub.load_state_dict_from_url (
                url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth",
                map_location="cpu", check_hash=True
            )
            self.model.load_state_dict(checkpoint,strict=False)
            # checkpoint = torch.load(pretrained_model)["model_state"]
            # changed_ckpt = {}
            # for key, value in checkpoint.items():
            #     changed_ckpt[key[6:]] = value
            # self.model.load_state_dict(changed_ckpt,strict=True)
        
        


    def forward(self, x):
        x_resPrompt, layer_wise_tokens, attention_maps, is_image = self.model(x)
        return x_resPrompt[-1], is_image

    def relprop(self,x,method="transformer_attribution", is_ablation=False, start_layer=0,num_prompts=1, is_image=False, **kwargs ):
        return self.model.relprop(x,method=method, is_ablation=is_ablation, start_layer=start_layer, **kwargs)

