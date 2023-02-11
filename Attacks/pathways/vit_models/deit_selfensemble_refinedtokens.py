'''
Deit Models with Self-Ensemble and Refined Tokens proposed by

Naseer et.al "Improving Adversarial Transferability of Vision Transformers"

'''

from functools import partial

import torch
import torch.nn as nn
import math
from einops import reduce, rearrange
from timm.models.registry import register_model
from timm.models.vision_transformer import VisionTransformer, _cfg

from .transformer_block import *
from timm.models import create_model

import random

import torch.nn.functional as F

__all__ = [
    "tiny_patch16_224_SelfEnsemble_RefinedTokens", "small_patch16_224_SelfEnsemble_RefinedTokens", "base_patch16_224_SelfEnsemble_RefinedTokens"
]


class TransformerHead(nn.Module):
    '''
    This is the token refinement module added between the output
    of each block and the final classifier (norm+head)
    '''
    expansion = 1

    def __init__(self, token_dim, num_patches=196, num_classes=1000, stride=1):
        super(TransformerHead, self).__init__()

        self.token_dim = token_dim
        self.num_patches = num_patches
        self.num_classes = num_classes

        # To process patches
        self.conv = nn.Conv2d(self.token_dim, self.token_dim, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(self.token_dim)
        self.conv = nn.Conv2d(self.token_dim, self.token_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(self.token_dim)

        self.shortcut = nn.Sequential()
        if stride != 1 or self.token_dim != self.expansion * self.token_dim:
            self.shortcut = nn.Sequential(
                nn.Conv2d(self.token_dim, self.expansion * self.token_dim, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * self.token_dim)
            )

        self.token_fc = nn.Linear(self.token_dim, self.token_dim)

    def forward(self, x):
        """
            x : (B, num_patches + 1, D) -> (B, C=num_classes)
        """
        cls_token, patch_tokens = x[:, 0], x[:, 1:]
        size = int(math.sqrt(x.shape[1]))

        patch_tokens = rearrange(patch_tokens, 'b (h w) d -> b d h w', h=size, w=size)  # B, D, H, W
        features = F.relu(self.bn(self.conv(patch_tokens)))
        features = self.bn(self.conv(features))
        features += self.shortcut(patch_tokens)
        features = F.relu(features)
        patch_tokens = F.avg_pool2d(features, 14).view(-1, self.token_dim)
        cls_token = self.token_fc(cls_token)

        out = patch_tokens + cls_token

        return out


class VisionTransformer_SelfEnsemble_RefinedTokens(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        patch_size = kwargs['patch_size']
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

        act_layer = None
        act_layer = act_layer or nn.GELU

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule

        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(self.depth)])

        # Token Refinement Module
        self.transformerheads = nn.Sequential(*[
            TransformerHead(self.embed_dim)
            for i in range(self.depth-1)])

    def forward_features(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        layer_wise_tokens = []
        attention_maps = []
        layer_wise_refined_outputs = []
        for idx, blk in enumerate(self.blocks):
            x, attn = blk(x, return_attention=True)
            layer_wise_tokens.append(x)
            attention_maps.append(attn)

        layer_wise_tokens = [self.norm(x) for x in layer_wise_tokens]
        layer_wise_refined_outputs = [self.transformerheads[idx](x) for idx, x in enumerate(layer_wise_tokens) if idx < self.depth-1]
        return [x[:, 0] for x in layer_wise_tokens], layer_wise_tokens, attention_maps, layer_wise_refined_outputs

    def forward(self, x, only_last=False,  all_tokens=False):
        token_out, layer_wise_tokens, attention_maps, layer_wise_refined_outputs = self.forward_features(x)

        x = [self.head(x) for x in token_out] # Cls Tokens without refinement moduls
        x_refined = [self.head(x) for x in layer_wise_refined_outputs]
        x_refined.append(x[-1])

        if only_last:
            return x[-1]  # return classification score of the last token
        elif all_tokens:
            return x_refined, layer_wise_tokens, attention_maps # all cls token w and w/o refinement, block outputs and attention maps.
        else:
            return x_refined # return classification scores of all the tokens (refined and the last cls token)


@register_model
def tiny_patch16_224_SelfEnsemble_RefinedTokens(pretrained=False, **kwargs):
    model = VisionTransformer_SelfEnsemble_RefinedTokens(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://github.com/Muzammal-Naseer/Improving-Adversarial-Transferability-of-Vision-Transformers"
                "/releases/download/v0/deit_tiny_trm.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["state_dict"])
    return model


@register_model
def small_patch16_224_SelfEnsemble_RefinedTokens(pretrained=False, **kwargs):
    model = VisionTransformer_SelfEnsemble_RefinedTokens(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://github.com/Muzammal-Naseer/Improving-Adversarial-Transferability-of-Vision-Transformers"
                "/releases/download/v0/deit_small_trm.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["state_dict"])
    return model


@register_model
def base_patch16_224_SelfEnsemble_RefinedTokens(pretrained=False, **kwargs):
    model = VisionTransformer_SelfEnsemble_RefinedTokens(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://github.com/Muzammal-Naseer/Improving-Adversarial-Transferability-of-Vision-Transformers"
                "/releases/download/v0/deit_base_trm.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["state_dict"])
    return model

if __name__=="__main__":
    model = create_model(
        'tiny_patch16_224_SelfEnsemble_RefinedTokens',
        pretrained=False,
        num_classes=1000,
        drop_rate=0.0,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        drop_block_rate=None,
    )
    model(torch.randn(2,3,224,224))
    print(sum(p.numel() for p in model.parameters() if p.requires_grad)/1000000)