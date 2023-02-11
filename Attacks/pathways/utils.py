import json
import math
import os
from typing import Tuple, Any
from einops import rearrange
from matplotlib import pyplot as plt
import numpy as np
import copy


import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms, models
from timm.models import create_model
from torchvision import models as pytorch_models
from torchvision.datasets.folder import default_loader

from robustness import model_utils
from robustness.datasets import ImageNet

from . import vit_models
from . import cnn_models
from . import clip_models
from .vit_models.build import build_model

from .mvcnn_models.MVCNN import SVCNN, MVCNN

pytorch_model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))

cnn_model_names = sorted(name for name in cnn_models.__dict__
                         if not name.startswith("__")
                         and callable(cnn_models.__dict__[name]))

vit_model_names = sorted(name for name in vit_models.__dict__
                         if not name.startswith("__")
                         and callable(vit_models.__dict__[name]))

robust_model_names = ['resnet50_l2_eps0.01', 'resnet50_linf_eps8.0', 'resnet50_linf_eps0.5']

video_model_names = ['deit_base_patch16_224_timeP_1_cat_ens','dino_base_patch16_224_1P_ens','clip_base_patch16_224_1P_ens','dino_base_image','clip_base_image','deit_base_image','deit_small_image','deit_tiny_image','deit_small_patch16_224_timeP_1', 'deit_tiny_patch16_224_timeP_1','deit_base_patch16_224_timeP_1_base_prompt','deit_base_patch16_224_timeP_1_full_1568','vit_base_patch16_224_base_lin','deit_base_patch16_224_timeP_1_cat','deit_base_patch16_224_timeP_1_org','deit_base_patch16_224_base_lin', 'deit_base_patch16_224_base_prompt','deit_base_patch16_224_timeP_1','vit_base_patch16_224_timeP_1', 'dino_base_patch16_224_1P', 'clip_base_patch16_224_1P', 'timesformer_vit_base_patch16_224', 'resnet_50', 'i3d']

img3d_model_names = ['mvcnn']


def get_model(model_name, num_classes, args, pretrained=True, is_src=False):

    # models with different normalization
    model_names_diff_norm = ['vit_base_patch16_224',
                             'vit_large_patch16_224',
                             'tnt_s_patch16_224']


    if model_name in cnn_model_names:
        model = cnn_models.__dict__[model_name](pretrained=pretrained)
        model.depth = 1
        if model_name in ['cifar10', 'cifar100', 'BIT', 'BIT_160']:
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)
        else:
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)

    elif model_name in pytorch_model_names:
        model = pytorch_models.__dict__[model_name](pretrained=pretrained)
        model.depth = 1
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    elif model_name in vit_model_names:
        if 'dino' in model_name:
            model = create_model(model_name, pretrained=True,
                                 num_classes=0,
                                 drop_rate=0.0,
                                 drop_path_rate=0.1,
                                 attn_drop_rate=0.0,
                                 drop_block_rate=None)

        else:
            model = create_model(model_name, pretrained=pretrained,
                                 num_classes=num_classes,
                                 drop_rate=0.0,
                                 drop_path_rate=0.1,
                                 attn_drop_rate=0.0,
                                 drop_block_rate=None)

        if model_name in model_names_diff_norm:
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)
        else:
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
    elif model_name in robust_model_names:
        imagenet_ds = ImageNet('/')
        model, _ = model_utils.make_and_restore_model(arch=model_name.split('_')[0], dataset=imagenet_ds,
                                                      resume_path=f'pretrained_models/{model_name}.ckpt', parallel=False, add_custom_forward=True)

        model.depth = 1
        mean = (0.0, 0.0, 0.0)
        std = (1.0, 1.0, 1.0)

    elif 'clip' in model_name and model_name not in ['clip_base_patch16_224_1P','clip_base_image', 'clip_base_patch16_224_1P_ens']:
        model, preprocess = clip_models.clip.load(model_name)
        model.depth = 1
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
    
    elif model_name in video_model_names:
        if is_src:
            args2 = copy.deepcopy(args)
            args2.num_frames = args.src_frames
        else:
            args2 = copy.deepcopy(args)
        args2.num_classes = num_classes
        model = build_model(args2, model_name)

        ## normalisation for video models is done in dataloader so no need to do it again
        mean = (0, 0, 0)
        std = (1, 1, 1)

    elif model_name in img3d_model_names:
        model_s = SVCNN("mvcnn", 40, True, 'vgg11')
        model = MVCNN("mvcnn", model_s, 40, 'vgg11',8)

        mean = (0, 0, 0)
        std = (1, 1, 1)

    else:
        raise NotImplementedError(f"Please provide correct model names: {model_names}")

    return model, mean, std



#  Test Samples
def get_data_loader(args, verbose=True):

    test_dir = args.test_dir

    data_transform = transforms.Compose([
        transforms.Resize(args.scale_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
    ])

    if args.data_type == 'IN':
        if args.image_list == "":
            print("Loading ImageNet validation set")
            test_dir = test_dir + '/IN/val'
            test_set = torchvision.datasets.ImageFolder(test_dir, data_transform)
        elif args.image_list == "common_corruptions":
            print("Loading ImageNet validation set with common corruption: {}".format(corruption))
            test_dir = test_dir + '/IN/val'
            test_set = torchvision.datasets.ImageFolder(test_dir, data_transform)
        elif "image_list" in args.image_list:
            print("Loading ImageNet validation subset")
            test_dir = test_dir + '/IN/val'
            test_set =  AdvImageNet(image_list=f"data/{args.image_list}", root=test_dir, transform=data_transform)

    elif args.data_type == 'CIFAR10':
        test_dir = test_dir+'/cifar10'
        test_set = torchvision.datasets.CIFAR10(root=test_dir, train=False, download=True, transform=data_transform)
    elif args.data_type == 'CIFAR100':
        test_dir = test_dir + '/cifar100'
        test_set = torchvision.datasets.CIFAR100(root=test_dir, train=False, download=True, transform=data_transform)
    elif args.data_type == 'voc':
        print("Loading Pascal Voc")
        test_dir = test_dir
        data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        def load_target(image):
            image = np.array(image)
            image = torch.from_numpy(image)
            return image

        target_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(load_target),
        ])

        test_size = torchvision.datasets.VOCSegmentation(root=test_dir, image_set="val", download=True,
                                                       transform=data_transform,
                                                       target_transform=target_transform)
        test_loader = torch.utils.data.DataLoader(test_size, batch_size=1, drop_last=False)

        return test_loader, test_size


    test_size = len(test_set)
    if verbose:
        print('Test data size:', test_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                              pin_memory=True)
    return test_loader, test_size



class AdvImageNet(torchvision.datasets.ImageFolder):

    def __init__(self, image_list, *args, **kwargs):

        self.image_list = set(json.load(open(image_list, "r"))["images"])
        super(AdvImageNet, self).__init__(is_valid_file=self.is_valid_file, *args, **kwargs)

    def is_valid_file(self, x: str) -> bool:
        return x[-38:] in self.image_list


class Flowers102Dataset(torchvision.datasets.VisionDataset):
    """
    Dataset class for Oxford Flowers102 Dataset
    """

    def __init__(self, image_root_path, split="train", *args, **kwargs):
        """

        Args:
            image_root_path:      path to dir containing images and lists folders
            split:                train / val / test / trainval
            *args:
            **kwargs:
        """
        self.loader = default_loader

        self.classes = list(range(1, 103))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        annotations = self.get_file_content(f"{image_root_path}/annotations/{split}.csv")
        self.samples = [[x.split(",")[0], int(x.split(",")[1].strip())] for x in annotations]
        self.targets = [self.class_to_idx[s[1]] for s in self.samples]

        super(Flowers102Dataset, self).__init__(root=f"{image_root_path}/images", *args, **kwargs)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        path = os.path.join(self.root, f"{path}")
        target = self.class_to_idx[target]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def get_file_content(file_path):
        with open(file_path) as fo:
            content = fo.readlines()
        return content

def normalize(t, mean, std):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]

    return t


def resize(images, size, mode="bilinear"):
    if isinstance(size, int):
        new_height, new_width = size, size
    else:
        new_height, new_width = size
    return torch.nn.functional.interpolate(
        images,
        size=(new_height, new_width),
        mode=mode,
        align_corners=False,
    )


def plot_overlays(image, attn_map_list, save_path=None):
    plt.ioff()
    for head in range(attn_map_list.shape[1]):
        cur_attn_map = attn_map_list[0, head]
        fig = plt.figure()
        plt.imshow(image.cpu())
        plt.imshow(cur_attn_map.cpu(), alpha=0.5)
        plt.axis("off")
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path.format(head))
        plt.close(fig)


def plot_joint_overlay(image, attn_map_list, c_attn_map_list, save_path=None):
    plt.ioff()
    nh = attn_map_list.shape[1]
    fig, axes = plt.subplots(2, nh)
    fig.set_size_inches(12, 5)

    for head in range(nh):
        cur_ax = axes[0, head]
        cur_attn_map = attn_map_list[0, head]
        cur_ax.imshow(image.detach().cpu())
        cur_ax.imshow(cur_attn_map.detach().cpu(), alpha=0.5)
        cur_ax.set_axis_off()

        cur_ax = axes[1, head]
        cur_attn_map = c_attn_map_list[0, head]
        cur_ax.imshow(image.detach().cpu())
        cur_ax.imshow(cur_attn_map.detach().cpu(), alpha=0.5)
        cur_ax.set_axis_off()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path.format(head))
    plt.close(fig)


def get_attn_maps(attn_mat, attn_thresh=None):
    attn_mat = F.softmax(attn_mat, dim=-1)
    cls_token_attn = attn_mat[:, :, 0, 1:]
    if attn_thresh is None:
        th_attn = cls_token_attn
        th_attn = rearrange(th_attn, "b nh (h w) -> b nh h w", h=int(math.sqrt(th_attn.shape[-1])))
        th_attn = resize(th_attn, size=224)
        return th_attn

    else:
        nh = cls_token_attn.shape[1]
        cls_token_attn = cls_token_attn[0]  # fix for larger batch sizes
        val, idx = torch.sort(cls_token_attn)
        val /= torch.sum(val, dim=1, keepdim=True)
        cum_val = torch.cumsum(val, dim=1)
        th_attn = cum_val > (1 - attn_thresh)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]

        th_attn = rearrange(th_attn, "nh (h w) -> nh h w", h=int(math.sqrt(th_attn.shape[-1])))
        th_attn = resize(th_attn.unsqueeze(0).to(float), size=224)

        return th_attn


class SoftTargetCrossEntropy(torch.nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    @staticmethod
    def forward(x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


# some helper functions for DINO Models
def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name, patch_size):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if model_name == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif model_name == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        elif model_name == "xcit_small_12_p16":
            url = "dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_pretrain.pth"
        elif model_name == "xcit_small_12_p8":
            url = "dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_pretrain.pth"
        elif model_name == "xcit_medium_24_p16":
            url = "dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain.pth"
        elif model_name == "xcit_medium_24_p8":
            url = "dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_pretrain.pth"
        elif model_name == "resnet50":
            url = "dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")
