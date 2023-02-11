import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F

from timesformer.models.vit_utils import trunc_normal_

from .build import MODEL_REGISTRY
from torch import einsum
from einops import rearrange, reduce, repeat
from .transformer_block import Block as Images_Block
from .timesformer_block import Block, PatchEmbed
from timm.models.vision_transformer import VisionTransformer as Images_VIT, _cfg
from .clip import LayerNorm, Transformer

class resPromptVisionTransformer(Images_VIT):
    def __init__(self, *args,actual_num_classes=400,qk_scale=None, num_prompts=1, num_frames=8, img_size=224, attention_type='divided_space_time', num_cross=1, **kwargs):
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
        self.num_frames = num_frames

        act_layer = None
        act_layer = act_layer or nn.GELU

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=self.patch_size, in_chans=3, embed_dim=embed_dim)

        # if self.attention_type != 'space_only':
        #     self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        #     self.time_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule

        self.blocks = nn.Sequential(*[
            Images_Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(self.depth)])

        # self.transformation = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,attention_type=self.attention_type)
        self.resPrompt_token = nn.Parameter(torch.zeros(1, self.num_prompts, self.embed_dim))
        self.head_resPrompt = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        # self.prompt_norm = nn.LayerNorm(self.embed_dim)
        # self.aggr_linear = nn.Linear(num_frames, 1)

        trunc_normal_(self.resPrompt_token, std=.02)
        self.apply(self._init_weights)

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
        x = x[:,:,self.num_frames // 2, :, :].unsqueeze(2)
        B = x.shape[0]
        x,T,W = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)

        ## transformation
        x = rearrange(x[:,:,:], '(b t) n m -> b (n t) m',t=T)
        # transformation = self.transformation(x[:,:,:],B,T,W,is_cls=False)
        x = rearrange(x[:,:,:], 'b (n t) m -> (b t) n m',t=T)
        
        x = torch.cat((cls_tokens, x), dim=1)
        

        ## Spatial embedding
        x = self.spatial_embedding(x,W)

        ## Time Embeddings
        # if self.attention_type != 'space_only':
        #     x = self.time_embedding(x,B,T)


        # cls_tokens = x[:B, 0, :].unsqueeze(1)
        # x = rearrange(x[:,1:,:], '(b t) n m -> b n t m',b=B,t=T).mean(dim=2)
        # x = torch.cat((cls_tokens, x), dim=1)

        ## prompt token
        # resPrompt_token = self.resPrompt_token.expand(B, -1, -1) + transformation.mean(dim=1)
        resPrompt_token = self.resPrompt_token.expand(B, -1, -1)
        x = torch.cat((resPrompt_token, x), dim=1)

        layer_wise_tokens = []
        attention_maps = []
        for i,blk in enumerate(self.blocks):
            x, attn = blk(x, return_attention=True)
            layer_wise_tokens.append(x)
            attention_maps.append(attn)

        return layer_wise_tokens, attention_maps

    def forward(self, x, all_tokens=True):
        size = x.size(0)
        layer_wise_tokens, attention_maps = self.forward_features(x)
        layer_wise_tokens_norm = [self.norm(x) for x in layer_wise_tokens]
        x = [self.head(x[:, self.num_prompts]) for x in layer_wise_tokens_norm]
        x_resPrompt = [self.head_resPrompt(x[:, 0:self.num_prompts].mean(dim=1)) for x in layer_wise_tokens_norm]
        return x_resPrompt, [xi[:,self.num_prompts] for xi in layer_wise_tokens], attention_maps  # , x

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
        self.num_frames = num_frames

        act_layer = None
        act_layer = act_layer or nn.GELU

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=self.patch_size, in_chans=3, embed_dim=embed_dim)

        self.head = nn.Identity()
        self.linear = nn.Linear(embed_dim*2, self.num_classes)

        # if self.attention_type != 'space_only':
        #     self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        #     self.time_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule

        self.blocks = nn.Sequential(*[
            Images_Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(self.depth)])

        self.transformation = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,attention_type=self.attention_type)
        self.resPrompt_token = nn.Parameter(torch.zeros(1, self.num_prompts, embed_dim))
        self.head_resPrompt = nn.Linear(embed_dim*2, num_classes) if num_classes > 0 else nn.Identity()


        trunc_normal_(self.resPrompt_token, std=.02)
        self.apply(self._init_weights)

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
        x = x[:,:,self.num_frames // 2, :, :].unsqueeze(2)
        B = x.shape[0]
        x,T,W = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)

        ## transformation
        x = rearrange(x[:,:,:], '(b t) n m -> b (n t) m',t=T)
        # transformation = self.transformation(x[:,:,:],B,T,W,is_cls=False).mean(dim=1, keepdim=True)
        x = rearrange(x[:,:,:], 'b (n t) m -> (b t) n m',t=T)
        x = torch.cat((cls_tokens,x),dim=1)

        ## Spatial embedding
        x = self.spatial_embedding(x,W)

        ## Time Embeddings
        # if self.attention_type != 'space_only':
        #     x = self.time_embedding(x,B,T)

        cls_tokens = x[:B, 0, :].unsqueeze(1)
        x = rearrange(x[:,1:,:], '(b t) n m -> b n t m',b=B,t=T).mean(dim=2)
        x = torch.cat((cls_tokens, x), dim=1)

        resPrompt_token = self.resPrompt_token.expand(B, -1, -1)
        x = torch.cat((resPrompt_token, x), dim=1)

        layer_wise_tokens = []
        attention_maps = []
        for i,blk in enumerate(self.blocks):
            x, attn = blk(x, return_attention=True)
            layer_wise_tokens.append(x)
            attention_maps.append(attn)

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
        self.num_frames = num_frames

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate , 12)] 

        self.transformation = Block(dim=self.width, num_heads=heads, mlp_ratio=4, qkv_bias=True, qk_scale=qk_scale, drop=0., attn_drop=0., drop_path=dpr[0], norm_layer=partial(nn.LayerNorm, eps=1e-6),attention_type=self.attention_type) 
        self.resPrompt_token = nn.Parameter(torch.zeros(1, self.num_prompts, self.width))
        self.head_resPrompt = nn.Linear(self.width, actual_num_classes) if actual_num_classes > 0 else nn.Identity()

        # if self.attention_type != 'space_only':
        #     self.time_embed = nn.Parameter(torch.zeros(1, num_frames, width))
        #     self.time_drop = nn.Dropout(p=0.)

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
        x = x[:,:,self.num_frames // 2, :, :].unsqueeze(2)
        B,C,T,H,W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.conv1(x)
        W = x.size(-1)
        x = x.flatten(2).transpose(1,2)

        x = rearrange(x[:,:,:], '(b t) n m -> b (n t) m',t=T)
        # transform = self.transformation(x[:,:,:],B,T,W,is_cls=False).mean(dim=1, keepdim=True)
        x = rearrange(x[:,:,:], 'b (n t) m -> (b t) n m',t=T)

        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        
        ## Spatial embedding
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
        # if self.attention_type != 'space_only':
        #     x = self.time_embedding(x,B,T)

        cls_tokens = x[:B, 0, :].unsqueeze(1)
        x = rearrange(x[:,1:,:], '(b t) n m -> b n t m',b=B,t=T).mean(dim=2)
        x = torch.cat((cls_tokens, x), dim=1)

        resPrompt_token = self.resPrompt_token.expand(B, -1, -1)
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