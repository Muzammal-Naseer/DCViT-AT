# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import math
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from .transformer_block import *
from timm.models import create_model

from einops import reduce, rearrange
import torch.nn.functional as F


class ConvHead(nn.Module):
    expansion = 1

    def __init__(self, token_dim, num_patches=196, num_classes=1000, stride=1):
        super(ConvHead, self).__init__()

        self.token_dim = token_dim
        self.num_patches = num_patches
        self.num_classes = num_classes

        # To process patches
        self.conv = nn.Conv2d(self.token_dim, self.token_dim, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(self.token_dim)

        self.shortcut = nn.Sequential()
        if stride != 1 or self.token_dim != self.expansion * self.token_dim:
            self.shortcut = nn.Sequential(
                nn.Conv2d(self.token_dim, self.expansion * self.token_dim, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * self.token_dim)
            )

    def forward(self, x):
        """
            x : (B, num_patches, D) -> (B, C=num_classes)
        """
        size = int(math.sqrt(x.shape[1]))
        x = rearrange(x, 'b (h w) d -> b d h w', h=size, w=size)  # B, D, H, W
        features = F.relu(self.bn(self.conv(x)))
        features += self.shortcut(x)
        features = F.relu(features)
        x = F.avg_pool2d(features, 14).view(-1, 1, self.token_dim)

        return x


__all__ = [
    'deit_tiny_patch16_224_resPC_1', 'deit_small_patch16_224_resPC_1', 'deit_base_patch16_224_resPC_1',
    'deit_tiny_patch16_224_resPC_10', 'deit_small_patch16_224_resPC_10', 'deit_base_patch16_224_resPC_10',
    'deit_tiny_patch16_224_resPC_100', 'deit_small_patch16_224_resPC_100', 'deit_base_patch16_224_resPC_100',
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

class resPromptVisionTransformer(VisionTransformer):
    def __init__(self, *args, num_prompts=1, **kwargs):
        super().__init__(*args, **kwargs)

        self.patch_size = kwargs['patch_size']
        embed_dim = kwargs['embed_dim']
        self.depth = kwargs['depth']
        num_heads = kwargs['num_heads']
        mlp_ratio = kwargs['mlp_ratio']
        qkv_bias = kwargs['qkv_bias']
        norm_layer = kwargs['norm_layer']
        num_classes = kwargs['num_classes']
        drop_rate = kwargs['drop_rate']
        drop_path_rate = kwargs['drop_path_rate']
        attn_drop_rate = kwargs['attn_drop_rate']
        self.num_prompts = num_prompts

        act_layer = None
        act_layer = act_layer or nn.GELU

        self.patch_embed = PatchEmbed(
            img_size=224, patch_size=self.patch_size, in_chans=3, embed_dim=embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule

        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(self.depth)])

        self.transformation = ConvHead(embed_dim)
        self.resPrompt_token = nn.Parameter(torch.zeros(1, self.num_prompts, self.embed_dim))
        self.head_resPrompt = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.resPrompt_token, std=.02)
        self.head_resPrompt.apply(self._init_weights)


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
        B, nc, w, h = x.shape
        x = self.patch_embed(x)

        transformation = self.transformation(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.interpolate_pos_encoding(x, w, h)

        resPrompt_token = self.resPrompt_token.expand(B, -1, -1) + transformation
        x = torch.cat((resPrompt_token, x), dim=1)

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0:self.num_prompts], x[:, self.num_prompts]

    def forward(self, x):
        size = x.size(0)
        x_resPrompt, x  = self.forward_features(x)
        x = self.head(x)
        x_resPrompt = self.head_resPrompt(x_resPrompt.mean(dim=1))
        return x_resPrompt, x



#######
# Models for 1, 10, and 100 prompts
######

# Only with 1 prompt
@register_model
def deit_tiny_patch16_224_resPC_1(pretrained=False, **kwargs):
    model = resPromptVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_prompts=1, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def deit_small_patch16_224_resPC_1(pretrained=False, **kwargs):
    model = resPromptVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),num_prompts=1, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_224_resPC_1(pretrained=False, **kwargs):
    model = resPromptVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_prompts=1,  **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


# Only with 10 prompts
@register_model
def deit_tiny_patch16_224_resPC_10(pretrained=False, **kwargs):
    model = resPromptVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_prompts=10, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_patch16_224_resPC_10(pretrained=False, **kwargs):
    model = resPromptVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),num_prompts=10, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_224_resPC_10(pretrained=False, **kwargs):
    model = resPromptVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_prompts=10,  **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model



# Only with 100 prompts
@register_model
def deit_tiny_patch16_224_resPC_100(pretrained=False, **kwargs):
    model = resPromptVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_prompts=100, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_patch16_224_resPC_100(pretrained=False, **kwargs):
    model = resPromptVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),num_prompts=100, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_224_resPC_100(pretrained=False, **kwargs):
    model = resPromptVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_prompts=100,  **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

if __name__=="__main__":
    model = create_model(
        'deit_base_patch16_224_resPC_100',
        pretrained=False,
        num_classes=1000,
        drop_rate=0.0,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        drop_block_rate=None,
    )
    model(torch.randn(2,3,224,224))
    print(sum(p.numel() for p in model.parameters() if p.requires_grad)/1000000)