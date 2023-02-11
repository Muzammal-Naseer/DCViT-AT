from timm.models.layers import trunc_normal_ as img_trunc_normal_
import torch
from torch import nn

if __name__ == '__main__':
    x = nn.Parameter(torch.zeros(1, 10, 768))
    print(x)
    img_trunc_normal_(x, std=.02)
    print(x)


