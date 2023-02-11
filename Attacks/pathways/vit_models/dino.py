# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math
from functools import partial

import torch
import torch.nn as nn
import os
from timm.models import create_model

from timm.models.vision_transformer import VisionTransformer, _cfg
# from timm.models.layers import PatchEmbed
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from .transformer_block import *


__all__ = [
    'dino_tiny_patch16_224', 'dino_small_patch16_224', 'dino_base_patch16_224','dino_base_patch8_224',
]


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class DINOVisionTransformer(VisionTransformer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.patch_size=kwargs['patch_size']
        embed_dim=kwargs['embed_dim']
        self.depth=kwargs['depth']
        num_heads=kwargs['num_heads']
        mlp_ratio=kwargs['mlp_ratio']
        qkv_bias=kwargs['qkv_bias']
        norm_layer=kwargs['norm_layer']
        # num_classes=kwargs['num_classes']
        drop_rate=kwargs['drop_rate']
        drop_path_rate=kwargs['drop_path_rate']
        attn_drop_rate=kwargs['attn_drop_rate']

        act_layer = None
        act_layer = act_layer or nn.GELU

        self.patch_embed = PatchEmbed(
            img_size=224, patch_size=self.patch_size, in_chans=3, embed_dim=embed_dim)

        self.head = nn.Identity()
        # print(self.patch_embed)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(self.depth)])

        self.linear = nn.Linear(1536, 1000)
        # self.depth=1

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
        print(class_pos_embed.unsqueeze(0).shape, patch_pos_embed.shape)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)


    def forward(self, x, nh=None,  only_last=False,  all_tokens=False):
        assert x.shape[2] == 1
        # x = x[:,:,4,:,:]
        x = x.squeeze(2)
        B, nc, w, h = x.shape
        x = self.prepare_tokens(x)
        layer_wise_tokens = []
        attention_maps = []
        for idx, blk in enumerate(self.blocks):
            x, attn = blk(x, return_attention=True, nh=nh)
            layer_wise_tokens.append(x)
            attention_maps.append(attn)

        layer_wise_tokens = [self.norm(x) for x in layer_wise_tokens]
        # intermediate_output = [layer_wise_tokens[-1]]
        # output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
        # print(output.shape)
        # avgpool = True
        # if avgpool:
        #     output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)),
        #                        dim=-1)
        #     print(output.shape)
        #     output = output.reshape(output.shape[0], -1)
        #     print(output.shape)

        x = [self.linear(torch.cat((x[:, 0].unsqueeze(-1), torch.mean(x[:,1:], dim=1).unsqueeze(-1)), dim=-1).reshape(B, -1)) for x in layer_wise_tokens]
        # x = [x[:, 0] for x in layer_wise_tokens]
        # x = self.linear(output)
        if only_last:
            return x[11:] # return last class token
        elif all_tokens:
            return x, x, attention_maps
        else:
            return x


@register_model
def dino_tiny_patch16_224(pretrained=False, **kwargs):
    model = DINOVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def dino_small_patch16_224(pretrained=False, **kwargs):
    model = DINOVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def dino_small_patch8_224(pretrained=False, **kwargs):
    model = DINOVisionTransformer(
        patch_size=8, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def dino_base_patch16_224(pretrained=False, **kwargs):
    model = DINOVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load('pretrained_models/dino_vitbase16_full.pth',
                                map_location="cpu"
                                )
        msg = model.load_state_dict(checkpoint)
        print(msg)
        # checkpoint = torch.load('pretrained_models/dino_vitbase16_linearweights.pth',
        #                         map_location="cpu"
        #                         )
        # checkpoint = checkpoint['state_dict']
        # checkpoint = {k.replace("module.linear.", ""): v for k, v in checkpoint.items()}
        # model.linear.load_state_dict(checkpoint)
        # checkpoint = {k for k, v in checkpoint.items()}
        # print(checkpoint)
        # checkpoint_linear = {k for k, v in checkpoint_linear['state_dict'].items() }
        # print(checkpoint_linear)
        # torch.save(model.state_dict(), 'pretrained_models/dino_vitbase16_full.pth')

    return model

@register_model
def dino_base_patch8_224(pretrained=False, **kwargs):
    model = DINOVisionTransformer(
        patch_size=8, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load('pretrained_models/dino_vitbase8_pretrain.pth',
                                map_location="cpu"
                                )
        msg = model.load_state_dict(checkpoint, strict=False)
        print(msg)
        checkpoint = torch.load('pretrained_models/dino_vitbase8_linearweights.pth',
                                map_location="cpu"
                                )
        checkpoint = checkpoint['state_dict']
        checkpoint = {k.replace("module.linear.", ""): v for k, v in checkpoint.items()}
        model.linear.load_state_dict(checkpoint)
        # checkpoint = {k for k, v in checkpoint.items()}
        # print(checkpoint)
        # checkpoint_linear = {k for k, v in checkpoint_linear['state_dict'].items() }
        # print(checkpoint_linear)
        torch.save(model.state_dict(), 'pretrained_models/dino_vitbase8_full.pth')

    return model

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


if __name__=="__main__":
    model = create_model(
        'dino_small_patch16_224',
        pretrained=True,
        num_classes=0,
        drop_rate=0.0,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        drop_block_rate=None,
    )
    model(torch.randn(1,3,56,56))