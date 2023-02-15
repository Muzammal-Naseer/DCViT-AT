# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2020 Ross Wightman
# Modified Model definition

import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F

from imtt.models.vit_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from imtt.models.helpers import load_pretrained
from imtt.models.vit_utils import trunc_normal_

from .build import MODEL_REGISTRY
from einops import rearrange, reduce, repeat
BASE=False

from .prompt_models_196_cat_prompt import resPromptVisionTransformer, resPromptDino, resPromptClip
# BASE=True
# from .vid_layers import PatchEmbed, Block
from .resnet import generate_model

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


##########################
#        DINO+CLIP       #
##########################
@MODEL_REGISTRY.register()
class dino_base_patch16_224_1P(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(dino_base_patch16_224_1P, self).__init__()
        self.pretrained=True
        patch_size = 16
        self.model = resPromptDino(img_size=cfg.DATA.TRAIN_CROP_SIZE,num_classes=1000, num_prompts=1,actual_num_classes=cfg.MODEL.NUM_CLASSES, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=cfg.DATA.NUM_FRAMES, attention_type=cfg.TIMESFORMER.ATTENTION_TYPE, **kwargs)

        self.attention_type = cfg.TIMESFORMER.ATTENTION_TYPE
        self.model.default_cfg = default_cfgs['vit_base_patch16_224']
        self.num_patches = (cfg.DATA.TRAIN_CROP_SIZE // patch_size) * (cfg.DATA.TRAIN_CROP_SIZE // patch_size)
        pretrained_model=cfg.TIMESFORMER.PRETRAINED_MODEL
        if self.pretrained:
            checkpoint = torch.load("timesformer/models/pretrained/dino_vitbase16_full.pth")
            self.model.load_state_dict(checkpoint,strict=False)
        
        # self.model.resPrompt_token.data = self.model.cls_token.data.clone()
        
    def forward(self, x):
        x_resPrompt, _ = self.model(x)
        return x_resPrompt

@MODEL_REGISTRY.register()
class clip_base_patch16_224_1P(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(clip_base_patch16_224_1P, self).__init__()
        self.pretrained=True
        patch_size = 16
        self.model = resPromptClip(
                input_resolution = 224,
                patch_size = 16,
                width = 768,
                layers = 12,
                heads = 768 // 64,
                output_dim = 512,
                num_prompts=1,
                num_classes = 1000,
                actual_num_classes = cfg.MODEL.NUM_CLASSES,
                num_frames=cfg.DATA.NUM_FRAMES,
                attention_type=cfg.TIMESFORMER.ATTENTION_TYPE)

        self.attention_type = cfg.TIMESFORMER.ATTENTION_TYPE
        self.model.default_cfg = default_cfgs['vit_base_patch16_224']
        self.num_patches = (cfg.DATA.TRAIN_CROP_SIZE // patch_size) * (cfg.DATA.TRAIN_CROP_SIZE // patch_size)
        pretrained_model=cfg.TIMESFORMER.PRETRAINED_MODEL
        if self.pretrained:
            checkpoint = torch.load("timesformer/models/pretrained/clip_ViTB16_vision_only.pth")
            self.model.load_state_dict(checkpoint,strict=False)
        
        # self.model.resPrompt_token.data = self.model.class_embedding.data.unsqueeze(0).unsqueeze(0).clone()
        
    def forward(self, x):
        x_resPrompt, x_cls = self.model(x)
        return x_resPrompt

##########################
#          Deit          #
##########################
@MODEL_REGISTRY.register()
class deit_base_patch16_224_timeP_1(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(deit_base_patch16_224_timeP_1, self).__init__()
        self.pretrained=True
        patch_size = 16
        self.model = resPromptVisionTransformer( img_size=cfg.DATA.TRAIN_CROP_SIZE,num_prompts=1,num_classes=1000, actual_num_classes=cfg.MODEL.NUM_CLASSES, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=cfg.DATA.NUM_FRAMES, attention_type=cfg.TIMESFORMER.ATTENTION_TYPE, **kwargs)

        self.attention_type = cfg.TIMESFORMER.ATTENTION_TYPE
        self.model.default_cfg = default_cfgs['vit_base_patch16_224']
        self.num_patches = (cfg.DATA.TRAIN_CROP_SIZE // patch_size) * (cfg.DATA.TRAIN_CROP_SIZE // patch_size)
        pretrained_model=cfg.TIMESFORMER.PRETRAINED_MODEL
        if self.pretrained:
            checkpoint = torch.load(
                "imtt/models/pretrained/deit_base_patch16_224-b5f2ef4d.pth"
            )
            ## remove module from keys
            # checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
            self.model.load_state_dict(checkpoint['model'],strict=False)

            # self.load_extras(0)
            # self.model.resPrompt_token.data = self.model.cls_token.data.clone()

    def load_extras(self,layer_ind):
        # self.model.resPrompt_token.data = self.model.cls_token.data.clone()
            
        self.model.transformation.norm1.weight.data = self.model.blocks[layer_ind].norm1.weight.data.clone()
        self.model.transformation.norm1.bias.data = self.model.blocks[layer_ind].norm1.bias.data.clone()
        self.model.transformation.attn.qkv.weight.data = self.model.blocks[layer_ind].attn.qkv.weight.data.clone()
        self.model.transformation.attn.qkv.bias.data = self.model.blocks[layer_ind].attn.qkv.bias.data.clone()
        self.model.transformation.attn.proj.weight.data = self.model.blocks[layer_ind].attn.proj.weight.data.clone()
        self.model.transformation.attn.proj.bias.data = self.model.blocks[layer_ind].attn.proj.bias.data.clone()
        self.model.transformation.norm2.weight.data = self.model.blocks[layer_ind].norm2.weight.data.clone()
        self.model.transformation.norm2.bias.data = self.model.blocks[layer_ind].norm2.bias.data.clone()
        self.model.transformation.mlp.fc1.weight.data = self.model.blocks[layer_ind].mlp.fc1.weight.data.clone()
        self.model.transformation.mlp.fc1.bias.data = self.model.blocks[layer_ind].mlp.fc1.bias.data.clone()
        self.model.transformation.mlp.fc2.weight.data = self.model.blocks[layer_ind].mlp.fc2.weight.data.clone()
        self.model.transformation.mlp.fc2.bias.data = self.model.blocks[layer_ind].mlp.fc2.bias.data.clone()

    def forward(self, x, individual_token=None):
        x_resPrompt, layer_wise_tokens, attention_maps = self.model(x, individual_token=individual_token)
        return x_resPrompt[-1]

@MODEL_REGISTRY.register()
class deit_small_patch16_224_timeP_1(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(deit_small_patch16_224_timeP_1, self).__init__()
        self.pretrained=True
        patch_size = 16
        self.model = resPromptVisionTransformer( img_size=cfg.DATA.TRAIN_CROP_SIZE,num_prompts=1,num_classes=1000, actual_num_classes=cfg.MODEL.NUM_CLASSES, patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=cfg.DATA.NUM_FRAMES, attention_type=cfg.TIMESFORMER.ATTENTION_TYPE, **kwargs)

        self.attention_type = cfg.TIMESFORMER.ATTENTION_TYPE
        self.model.default_cfg = default_cfgs['vit_base_patch16_224']
        self.num_patches = (cfg.DATA.TRAIN_CROP_SIZE // patch_size) * (cfg.DATA.TRAIN_CROP_SIZE // patch_size)
        pretrained_model=cfg.TIMESFORMER.PRETRAINED_MODEL
        if self.pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
            )
            self.model.load_state_dict(checkpoint["model"], strict=False)
            # self.load_extras(0)
            # self.model.resPrompt_token.data = self.model.cls_token.data.clone()

    def forward(self, x, individual_token=None):
        x_resPrompt, layer_wise_tokens, attention_maps = self.model(x, individual_token=individual_token)
        return x_resPrompt[-1]

@MODEL_REGISTRY.register()
class deit_tiny_patch16_224_timeP_1(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(deit_tiny_patch16_224_timeP_1, self).__init__()
        self.pretrained=True
        patch_size = 16
        self.model = resPromptVisionTransformer( img_size=cfg.DATA.TRAIN_CROP_SIZE,num_prompts=1,num_classes=1000, actual_num_classes=cfg.MODEL.NUM_CLASSES, patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=cfg.DATA.NUM_FRAMES, attention_type=cfg.TIMESFORMER.ATTENTION_TYPE, **kwargs)

        self.attention_type = cfg.TIMESFORMER.ATTENTION_TYPE
        self.model.default_cfg = default_cfgs['vit_base_patch16_224']
        self.num_patches = (cfg.DATA.TRAIN_CROP_SIZE // patch_size) * (cfg.DATA.TRAIN_CROP_SIZE // patch_size)
        pretrained_model=cfg.TIMESFORMER.PRETRAINED_MODEL
        if self.pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
            )
            self.model.load_state_dict(checkpoint["model"], strict=False)
            # self.load_extras(0)
            # self.model.resPrompt_token.data = self.model.cls_token.data.clone()
            

    def forward(self, x, individual_token=None):
        x_resPrompt, layer_wise_tokens, attention_maps = self.model(x, individual_token=individual_token)
        return x_resPrompt[-1]

##########################
#         Resnet         #
##########################
@MODEL_REGISTRY.register()
class resnet_50(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(resnet_50, self).__init__()
        self.model = generate_model(
                        model_depth=50,
                        n_classes=cfg.MODEL.NUM_CLASSES,
                        n_input_channels=3,
                        shortcut_type='B',
                        conv1_t_size=7,
                        conv1_t_stride=1,
                        no_max_pool=False,
                        widen_factor=1.0,
                        )

        ckpt = torch.load("timesformer/models/pretrained/r3d50_K_200ep.pth")
        ## remove module from keys in state dict
        ckpt = ckpt['state_dict']
        ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
        self.model.load_state_dict(ckpt, strict=False)
        
    def forward(self, x):
        x = self.model(x)
        return x