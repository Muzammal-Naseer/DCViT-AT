import argparse
import datetime
import json
import logging
import os
import numpy as np
import random
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision
import torchvision.utils as vutils
from einops import reduce, rearrange

from pathways.utils import get_model, get_data_loader, normalize
from pathways.clip_models.clip_utils import imagenet_classes, imagenet_templates
from pathways.clip_models import clip
from itertools import product
from datasets.loader import construct_loader

# for video testing
from utils.meters import TestMeter

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation of Vision Models')
    parser.add_argument('--test_dir', default='', help='ImageNet Validation Data Set')
    parser.add_argument('--image_list', type=str, default="", help='Image List from Validation Data stored as json file in data folder')
    parser.add_argument('--data_type', default='IN', help='ImageNet, CIFAR10/100')
    parser.add_argument('--src_model', type=str, default='deit_small_patch16_224', help='Source Model Name')
    parser.add_argument('--scale_size', type=int, default=256, help='')
    parser.add_argument('--img_size', type=int, default=224, help='')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch Size')
    parser.add_argument('--pre_trained', default=None, help='Load given model weights')

    # Transformer specific parameters
    parser.add_argument('--num_classes', type=int, default=1000, help='Number of output classes')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')
    parser.add_argument('--attn-drop-rate', type=float, default=0.0, metavar='PCT', help='Attention Drop rate ')

    # Video specific parameters
    parser.add_argument('--num_frames', type=int, default=1, help='Number of frames in a video')
    parser.add_argument('--num_spatial_crops', type=int, default=3, help='Number of spatial crops in a video')
    parser.add_argument('--num_temporal_views', type=int, default=1, help='Number of temporal crops in a video')
    parser.add_argument('--video_sampling_rate', type=int, default=32, help='Sampling rate of a video')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--num_workers', type=int, default=6, help='Number of workers to use')
    parser.add_argument('--video_mean', type=list, default=([0.45, 0.45, 0.45]), help='Mean of video dataset')
    parser.add_argument('--video_std', type=list, default=([0.225, 0.225, 0.225]), help='Std of video dataset')

    return parser.parse_args()

def main():
    # setup run
    args = parse_args()
    args.scale_size = int((256 / 224) * args.img_size)
    print(args)
    args.dir = f"results/{args.src_model}"
    if not os.path.isdir(args.dir):
        os.makedirs(args.dir)
    json.dump(vars(args), open(f"{args.dir}/config.json", "w"), indent=4)

    video_data = False
    if args.data_type in ['hmdb51', 'ucf101', 'kinetics', 'ssv2']:
        video_data = True

    # GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    src_model, src_mean, src_std = get_model(args.src_model, args.num_classes, args)
    if args.pre_trained:
        checkpoint = torch.load(args.pre_trained)
        if 'model' in checkpoint:
            src_model.load_state_dict(checkpoint['model'])
        elif 'state_dict' in checkpoint:
            src_model.load_state_dict(checkpoint['state_dict'])
        elif 'model_state' in checkpoint:
            src_model.load_state_dict(checkpoint['model_state'])
        else:
            src_model.load_state_dict(checkpoint)


    src_model = src_model.to(device).eval()
    num_blocks = src_model.depth

    if 'clip' in args.src_model:
        print('Tokenizing')
        def zeroshot_classifier(classnames, templates):
            with torch.no_grad():
                zeroshot_weights = []
                for classname in tqdm(classnames):
                    texts = [template.format(classname) for template in templates]  # format with class
                    texts = clip.tokenize(texts).cuda()  # tokenize
                    class_embeddings = src_model.encode_text(texts)  # embed with text encoder
                    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                    class_embedding = class_embeddings.mean(dim=0)
                    class_embedding /= class_embedding.norm()
                    zeroshot_weights.append(class_embedding)
                zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
            return zeroshot_weights

        zeroshot_weights = zeroshot_classifier(imagenet_classes, imagenet_templates)

    if video_data:
        test_loader, test_size = construct_loader(args)

        # setup test_meter for video multiview testing
        test_meter = TestMeter(
            len(test_loader.dataset)
            // (args.num_temporal_views * args.num_spatial_crops),
            args.num_temporal_views * args.num_spatial_crops,
            args.num_classes,
            len(test_loader),
            False,
            "sum",
            src_model.depth,
        )

    else:
        test_loader, test_size = get_data_loader(args)

    acc = {}
    for block in range(num_blocks):
        acc[block] = 0

    with tqdm(enumerate(test_loader), total=len(test_loader)) as prog_bar:
        for idx, image_label in prog_bar: # for idx, (img , label) in prog_bar: doesnt work for video 
            img, label = image_label[0].to(device), image_label[1].to(device)

            if video_data:
                video_idx, meta = image_label[2].to(device), image_label[3]
                for key, val in meta.items():
                    if isinstance(val, (list,)):
                        for i in range(len(val)):
                            val[i] = val[i].to(device)
                    else:
                        meta[key] = val.to(device)

            if 'clip' not in args.src_model:
                clean_out = src_model(normalize(img.clone(), mean=src_mean, std=src_std))
                if video_data:
                    if isinstance(clean_out, (tuple)):
                        clean_out = clean_out[0]
                    for num_layer in range(len(clean_out)):
                        clean_out[num_layer] = clean_out[num_layer].cpu()
                        clean_out[num_layer] = clean_out[num_layer].detach()

                    label = label.cpu()
                    video_idx = video_idx.cpu()
                    test_meter.update_stats(
                        clean_out, label.detach(), video_idx.detach()
                    )
                    continue
            else:
                image_features = src_model.encode_image(normalize(img.clone(), mean=src_mean, std=src_std))
                image_features /= image_features.norm(dim=-1, keepdim=True)
                clean_out = 100. * image_features @ zeroshot_weights
            # for robust models: they return tuple
            if isinstance(clean_out, tuple):
                clean_out = clean_out[0]

            if not isinstance(clean_out, list):
                clean_out = [clean_out]

            for block in range(num_blocks):
                acc[block] += torch.sum(clean_out[block].argmax(dim=-1) == label).item()

            del clean_out

    stats = test_meter.finalize_metrics()

    for block in range(num_blocks):
        if video_data:
            acc[block] = float(stats["layer_{}: top1_acc".format(block+1)])
            args.image_list = args.data_type
        else:
            acc[block] = round(acc[block]/test_size * 100, 3)
    acc['mean'] = round(np.array(list(acc.values())).mean(), 3) # Average accuracy across blocks

    json.dump({"Accuracy": acc},
              open(f"{args.dir}/{args.image_list.split('.')[0]}_acc_{args.img_size}.json", "w"), indent=4)

if __name__ == '__main__':
    main()
