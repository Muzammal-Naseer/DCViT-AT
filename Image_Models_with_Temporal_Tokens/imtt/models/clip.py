from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import math
import torch
import torch.nn.functional as F
from torch import nn
from timm.models import create_model
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
# from timm.models.vision_transformer import _init_weights
from .transformer_block import *

__all__ = [
    'clip_base_patch16_224_resP_all_1', 'clip_base_patch16_224_resP_all_10', 'clip_resnet50_224_resP_all_1', 'clip_resnet50_224_resP_all_10',
]


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
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

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, num_prompts: int, num_classes:int):
        super().__init__()
        self.num_prompts = num_prompts
        self.num_classes = num_classes
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.width = width
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        # Hard Coded for resolutions: 56, 96, 120, 160, 224
        self.resPrompt_token_56 = nn.Parameter(torch.zeros(1, self.num_prompts, self.width))
        self.head_resPrompt_56 = nn.Linear(self.width, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.resPrompt_token_96 = nn.Parameter(torch.zeros(1, self.num_prompts, self.width))
        self.head_resPrompt_96 = nn.Linear(self.width, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.resPrompt_token_120 = nn.Parameter(torch.zeros(1, self.num_prompts, self.width))
        self.head_resPrompt_120 = nn.Linear(self.width, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.resPrompt_token_160 = nn.Parameter(torch.zeros(1, self.num_prompts, self.width))
        self.head_resPrompt_160 = nn.Linear(self.width, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.resPrompt_token_224 = nn.Parameter(torch.zeros(1, self.num_prompts, self.width))
        self.head_resPrompt_224 = nn.Linear(self.width, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.resPrompt_token_56, std=.02)
        # self.head_resPrompt_56.apply(_init_weights)
        trunc_normal_(self.resPrompt_token_96, std=.02)
        # self.head_resPrompt_96.apply(_init_weights)
        trunc_normal_(self.resPrompt_token_120, std=.02)
        # self.head_resPrompt_120.apply(_init_weights)
        trunc_normal_(self.resPrompt_token_160, std=.02)
        # self.head_resPrompt_160.apply(_init_weights)
        trunc_normal_(self.resPrompt_token_224, std=.02)
        # self.head_resPrompt_224.apply(_init_weights)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.positional_embedding.shape[0] - 1
        if npatch == N and w == h:
            return self.positional_embedding
        class_pos_embed = self.positional_embedding[0]
        patch_pos_embed = self.positional_embedding[1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
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
        return torch.cat((class_pos_embed.unsqueeze(0).unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, x: torch.Tensor,  which_prompt=224, text_project=False):
        B, nc, w, h = x.shape
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        # x = x + self.positional_embedding.to(x.dtype)
        x = x + self.interpolate_pos_encoding(x, w, h)

        if which_prompt == 56:
            resPrompt_token = self.resPrompt_token_56.expand(B, -1, -1)
            x = torch.cat((resPrompt_token, x), dim=1)
        elif which_prompt == 96:
            resPrompt_token = self.resPrompt_token_96.expand(B, -1, -1)
            x = torch.cat((resPrompt_token, x), dim=1)
        elif which_prompt == 120:
            resPrompt_token = self.resPrompt_token_120.expand(B, -1, -1)
            x = torch.cat((resPrompt_token, x), dim=1)
        elif which_prompt == 160:
            resPrompt_token = self.resPrompt_token_160.expand(B, -1, -1)
            x = torch.cat((resPrompt_token, x), dim=1)
        elif which_prompt == 224:
            resPrompt_token = self.resPrompt_token_224.expand(B, -1, -1)
            x = torch.cat((resPrompt_token, x), dim=1)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x_resPrompt, x = x[:, 0:self.num_prompts].mean(dim=1), x[:, self.num_prompts]

        # x = self.ln_post(x[:, 0, :])
        x_resPrompt, x = self.ln_post(x_resPrompt), self.ln_post(x)

        if self.proj is not None and text_project:
            # projects prompts
            x_resPrompt = x_resPrompt @ self.proj
            x = x @ self.proj

        if which_prompt == 56:
            x_resPrompt = self.head_resPrompt_56(x_resPrompt)
        elif which_prompt == 96:
            x_resPrompt = self.head_resPrompt_96(x_resPrompt)
        elif which_prompt == 120:
            x_resPrompt = self.head_resPrompt_120(x_resPrompt)
        elif which_prompt == 160:
            x_resPrompt = self.head_resPrompt_160(x_resPrompt)
        elif which_prompt == 224:
            x_resPrompt = self.head_resPrompt_224(x_resPrompt)
        return x_resPrompt, x


@register_model
def clip_resnet50_224_resP_all_1(pretrained=False, **kwargs):
    model = ModifiedResNet(
                layers = (3,4,6,3),
                output_dim = 1024,
                heads = 64 * 32 // 64,
                input_resolution = 224,
                width = 64,
                num_prompts=1,
                num_classes=1000,
    )
    return model

@register_model
def clip_resnet50_224_resP_all_10(pretrained=False, **kwargs):
    model = ModifiedResNet(
                layers = (3,4,6,3),
                output_dim = 1024,
                heads = 64 * 32 // 64,
                input_resolution = 224,
                width = 64,
                num_prompts=10,
                num_classes=1000,
    )
    return model

@register_model
def clip_base_patch16_224_resP_all_1(pretrained=False, **kwargs):
    model = VisionTransformer(
                input_resolution = 224,
                patch_size = 16,
                width = 768,
                layers = 12,
                heads = 768 // 64,
                output_dim = 512,
                num_prompts=1,
                num_classes = 1000,
        )
    return model

@register_model
def clip_base_patch16_224_resP_all_10(pretrained=False, **kwargs):
    model = VisionTransformer(
                input_resolution = 224,
                patch_size = 16,
                width = 768,
                layers = 12,
                heads = 768 // 64,
                output_dim = 512,
                num_prompts=10,
                num_classes = 1000,
        )
    return model

if __name__=="__main__":
    model = create_model(
        'clip_base_patch16_224',
        pretrained=True,
        num_classes=0,
        drop_rate=0.0,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        drop_block_rate=None,
    )
    model(torch.randn(8,3,224,224))

# self.visual = ModifiedResNet(
#                 layers=vision_layers, (3,4,6,3))
#                 output_dim=embed_dim,1024
#                 heads=vision_heads, vision_width * 32 // 64
#                 input_resolution=image_resolution,224
#                 width=vision_width, 64
#
#
# self.visual = VisionTransformer(
#     input_resolution=image_resolution,224
#     patch_size=vision_patch_size,16
#     width=vision_width,768
#     layers=vision_layers,12
#     heads=vision_heads, vision_width // 64
#     output_dim=embed_dim, 512
# )
