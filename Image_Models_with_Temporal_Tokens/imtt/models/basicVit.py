import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
import math
import torch.nn.functional as F

from imtt.models.vit_utils import DropPath, to_2tuple, trunc_normal_
from .ViT_LRP import VisionTransformer, Block as Images_Block
# from .vit import Block, PatchEmbed
from .vid_layers import Block as Video_Block, PatchEmbed
from .layers_relprop import *

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


class BasicVIT(VisionTransformer):
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
        act_layer = act_layer or GELU

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=self.patch_size, in_chans=3, embed_dim=embed_dim)

        if self.attention_type != 'space_only':
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            self.time_drop = Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule

        self.blocks = Sequential(*[
            Images_Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(self.depth)])

        # self.transformation = Video_Block(extra_tokens=0,dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,attention_type=self.attention_type)
        # self.resPrompt_token = nn.Parameter(torch.zeros(1, self.num_prompts, self.embed_dim))
        # self.head_resPrompt = Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        ## initialize the temporal attention weights
        # if self.attention_type == 'divided_space_time':
        #     nn.init.constant_(self.transformation.temporal_fc.weight, 0)
        #     nn.init.constant_(self.transformation.temporal_fc.bias, 0)


        # trunc_normal_(self.resPrompt_token, std=.02)
        # self.head_resPrompt.apply(self._init_weights)

        self.pool = IndexSelect()
        self.add = Add()

        self.inp_grad = None

        self.is_image = False

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)

    def save_inp_grad(self,grad):
        self.inp_grad = grad

    def get_inp_grad(self):
        return self.inp_grad

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
        # if T != self.time_embed.size(1):
        #     time_embed = self.time_embed.transpose(1, 2)
        #     new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
        #     new_time_embed = new_time_embed.transpose(1, 2)
        #     x = x + new_time_embed
        # else:
        #     x = x + self.time_embed
        # x = self.time_drop(x)
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
        
        # x = self.pos_drop(x)
        return x

    def forward_features(self, x):
        if len(list(x.shape)) == 4:
            print("GOT AN IMAGE")
            x = x.unsqueeze(2)
            self.is_image = True
            self.num_prompts = 0
        B = x.shape[0]
        x,T,W = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        # x = torch.cat((cls_tokens,x),dim=1)

        # if not self.is_image:
        #     ## transformation
        #     x = rearrange(x[:,:,:], '(b t) n m -> b (n t) m',t=T)
        #     transformation = self.transformation(x[:,:,:],B,T,W,is_cls=False).mean(dim=1, keepdim=True)
        #     x = rearrange(x[:,:,:], 'b (n t) m -> (b t) n m',t=T)
        x = torch.cat((cls_tokens,x),dim=1)

        ## Spatial embedding
        x = self.spatial_embedding(x,W)

        if not self.is_image:
            ## Time Embeddings
            if self.attention_type != 'space_only':
                x = self.time_embedding(x,B,T)

        #     ## prompt token
        #     resPrompt_token = self.resPrompt_token.expand(B, -1, -1) + transformation
        #     x = torch.cat((resPrompt_token, x), dim=1)
        
        x.register_hook(self.save_inp_grad)

        layer_wise_tokens = []
        attention_maps = []
        for i,blk in enumerate(self.blocks):
            x = blk(x)
            layer_wise_tokens.append(x)

        return layer_wise_tokens, attention_maps

    def forward(self, x, all_tokens=True):
        size = x.size(0)
        layer_wise_tokens, attention_maps = self.forward_features(x)
        layer_wise_tokens_norm = [self.norm(x) for x in layer_wise_tokens]
        x = [self.head(x[:, 0]) for x in layer_wise_tokens_norm]
        return x, layer_wise_tokens, attention_maps
    
    def relprop(self, cam=None,method="transformer_attribution", is_ablation=False, start_layer=0, **kwargs):
        cam = self.head.relprop(cam, **kwargs)
        cam = cam.unsqueeze(1)
        # cam = self.pool.relprop(cam, **kwargs)
        cam = self.norm.relprop(cam, **kwargs)
        for blk in reversed(self.blocks):
            cam = blk.relprop(cam, **kwargs)
        
        # if not self.is_image:
        #     cam = self.transformation.relprop(cam[:,(self.num_prompts+1):], **kwargs)

        # print("conservation 2", cam.sum())
        # print("min", cam.min())

        if method == "full":
            (cam, _) = self.add.relprop(cam, **kwargs)
            cam = cam[:, 1:]
            cam = self.patch_embed.relprop(cam, **kwargs)
            # sum on channels
            cam = cam.sum(dim=1)
            return cam

        elif method == "rollout":
            # cam rollout
            attn_cams = []
            for blk in self.blocks:
                attn_heads = blk.attn.get_attn_cam().clamp(min=0)
                avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
                attn_cams.append(avg_heads)
            cam = compute_rollout_attention(attn_cams, start_layer=start_layer)
            cam = cam[:, 0, 1:]
            return cam
        
        # our method, method name grad is legacy
        elif method == "transformer_attribution" or method == "grad":
            cams = []
            for blk in self.blocks:
                grad = blk.attn.get_attn_gradients()
                cam = blk.attn.get_attn_cam()
                cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
                cam = cam.clamp(min=0).mean(dim=0)
                cams.append(cam.unsqueeze(0))
            rollout = compute_rollout_attention(cams, start_layer=start_layer)
            # print("ROLLOUT: ",rollout.shape)
            cam = rollout[:, 0, 1:]
            return cam
            
        elif method == "last_layer":
            cam = self.blocks[-1].attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            if is_ablation:
                grad = self.blocks[-1].attn.get_attn_gradients()
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam

        elif method == "last_layer_attn":
            cam = self.blocks[-1].attn.get_attn()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam

        elif method == "second_layer":
            cam = self.blocks[1].attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            if is_ablation:
                grad = self.blocks[1].attn.get_attn_gradients()
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam