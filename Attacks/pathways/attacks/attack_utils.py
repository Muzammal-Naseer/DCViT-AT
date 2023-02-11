import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats as st
from torch.nn.functional import interpolate


# Helper fuctions common to different attacks

def projection_operator(adv, img, eps):
    adv.data = torch.where(adv.data > img.data + eps, img.data + eps, adv.data)
    adv.data = torch.where(adv.data < img.data - eps, img.data - eps, adv.data)

# fix this?
# def input_diversity(img):
#     rnd = torch.randint(224, 257, (1,)).item()
#     rescaled = F.interpolate(img, (rnd, rnd), mode='nearest')
#     h_rem = 256 - rnd
#     w_hem = 256 - rnd
#     pad_top = torch.randint(0, h_rem + 1, (1,)).item()
#     pad_bottom = h_rem - pad_top
#     pad_left = torch.randint(0, w_hem + 1, (1,)).item()
#     pad_right = w_hem - pad_left
#     padded = F.pad(rescaled, pad=(pad_left, pad_right, pad_top, pad_bottom))
#     padded = F.interpolate(padded, (224, 224), mode='nearest')
#     return padded

def resize_input(img, size):
    if size <224:
        rescaled = F.interpolate(img, (size, size), mode='bicubic')
    else:
        rescaled = img
    return rescaled

def input_diversity(img):
    img_size = img.shape[-1]
    rnd = torch.randint(img_size, img_size+33, (1,)).item()
    rescaled = F.interpolate(img, (rnd, rnd), mode='nearest')
    h_rem = (img_size+32) - rnd
    w_hem = (img_size+32) - rnd
    pad_top = torch.randint(0, h_rem + 1, (1,)).item()
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(0, w_hem + 1, (1,)).item()
    pad_right = w_hem - pad_left
    padded = F.pad(rescaled, pad=(pad_left, pad_right, pad_top, pad_bottom))
    padded = F.interpolate(padded, (img_size, img_size), mode='nearest')
    return padded


def project_kern(kern_size):
    kern = np.ones((kern_size, kern_size), dtype=np.float32) / (kern_size ** 2 - 1)
    kern[kern_size // 2, kern_size // 2] = 0.0
    kern = kern.astype(np.float32)
    stack_kern = np.stack([kern, kern, kern])
    stack_kern = np.expand_dims(stack_kern, 1)
    stack_kern = torch.tensor(stack_kern).cuda()
    return stack_kern, kern_size // 2
stack_kern, padding_size = project_kern(3)
def project_noise(x, stack_kern, padding_size):
    x = F.conv2d(x, stack_kern, padding=(padding_size, padding_size), groups=3)
    return x


##define TI
def gkern(kernlen=15, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel
channels=3
kernel_size=5
kernel = gkern(kernel_size, 3).astype(np.float32)
gaussian_kernel = np.stack([kernel, kernel, kernel])
gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()