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
import torchvision.utils as vutils
from einops import reduce, rearrange
import torch.nn as nn

from pathways.utils import get_model, get_data_loader, normalize, get_attn_maps, plot_joint_overlay
from pathways.attacks import Adv_Attack

from pathways.clip_models.clip_utils import imagenet_classes, imagenet_templates
from pathways.clip_models import clip
from mvcnn.tools.ImgDataset import MultiviewImgDataset

import pytorch_ssim
import lpips

# for video testing
from utils.meters import TestMeter, CompareMeter
from datasets.loader import construct_loader
from utils.distributed import *
import utils.multiprocessing as mpu

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation of Vision Models')
    parser.add_argument('--test_dir', default='IN/val', help='ImageNet Validation Data Set')
    parser.add_argument('--image_list', default=None, help='Image List from Validation Data stored as json file in data folder')
    parser.add_argument('--data_type', default='IN', help='ImageNet, CIFAR10/100')
    parser.add_argument('--src_model', type=str, default='deit_small_patch16_224', help='Source Model Name')
    parser.add_argument('--tar_model', type=str, default='deit_small_patch16_224', help='Source Model Name')
    parser.add_argument('--scale_size', type=int, default=256, help='')
    parser.add_argument('--img_size', type=int, default=224, help='')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch Size')
    parser.add_argument('--pre_trained', default=None, help='Load given model weights')
    parser.add_argument('--tar_pre_trained', default=None, help='Load given model weights for target model')

    # Transformer specific parameters
    parser.add_argument('--num_classes', type=int, default=1000, help='Number of output classes')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')
    parser.add_argument('--attn-drop-rate', type=float, default=0.0, metavar='PCT', help='Attention Drop rate ')

    # Attack Parameters
    parser.add_argument('--attack_type', type=str, default='dim', help='Type of baseline attacks: fgsm, rfgsm, pgd, mifgsm, dim')
    parser.add_argument('--eps', type=int, default=16, help='Perturbation Budget')
    parser.add_argument('--iter', type=int, default=10, help='Attack iterations')
    parser.add_argument('--index', type=str, default='all', help='last(final classifier) or all(self-ensemble)')
    parser.add_argument('--target_label', type=int, default=-1, help='-1(untarget), 0,1,2...999')

    # Video specific parameters
    parser.add_argument('--num_frames', type=int, default=1, help='Number of frames in a video')
    parser.add_argument('--num_spatial_crops', type=int, default=3, help='Number of spatial crops in a video')
    parser.add_argument('--num_temporal_views', type=int, default=1, help='Number of temporal crops in a video')
    parser.add_argument('--video_sampling_rate', type=int, default=32, help='Sampling rate of a video')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of workers to use')
    parser.add_argument('--video_mean', type=list, default=([0.45, 0.45, 0.45]), help='Mean of video dataset')
    parser.add_argument('--video_std', type=list, default=([0.225, 0.225, 0.225]), help='Std of video dataset')
    parser.add_argument('--src_frames', type=int, default=1, help='Number of frames that src model takes')
    parser.add_argument('--num_div_gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--replicate_grad', type=bool, default=False, help='copy gradient of middle frame to all frames; required when using cat prompt as src model')
    parser.add_argument('--no_sup_loss', type=bool, default=False, help='ignore supervised loss')
    parser.add_argument('--no_unsup_loss', type=bool, default=False, help='ignore unsupervised loss')
    parser.add_argument('--variation', type=str, default='', help='')
    parser.add_argument('--add_grad', type=bool, default=False, help='add gradient of middle frame to all frames; required when using cat prompt as src model')
    parser.add_argument('--prod_grad', type=bool, default=False, help='multiply gradient of middle frame to all frames; required when using cat prompt as src model')
    

    return parser.parse_args()

def main():
    # setup run
    args = parse_args()

    print("STARTING ATTACK:", args.attack_type)

    if args.attack_type in ['fgsm', 'rfgsm']:
        args.iters = 1 # single step attacks

    print("pre_train: ", args.pre_trained)

    if args.variation == '':
        args.dir = f"results_adv/{args.attack_type}/{args.src_model}_{args.index}_{args.data_type}/{args.tar_model}"
    else:
        args.dir = f"results_adv/{args.attack_type}/{args.src_model}_{args.index}_{args.variation}_{args.data_type}/{args.tar_model}"
    if not os.path.isdir(args.dir):
        os.makedirs(args.dir)
    json.dump(vars(args), open(f"{args.dir}/config.json", "w"), indent=4)

    ## check if data is of videos
    video_data = False
    if args.data_type in ['hmdb51', 'ucf101', 'kinetics', 'ssv2']:
        video_data = True

    multi_data = False
    if 'img3d' in args.data_type:
        multi_data = True

    # GPU
    device1 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device2 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    src_model, src_mean, src_std = get_model(args.src_model, args.num_classes, args, is_src=True)
    if args.num_gpus > 1:
        src_model = src_model.module
    
    device = device1
    src_model = src_model.to(device).eval()
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

    tar_model, tar_mean, tar_std = get_model(args.tar_model, args.num_classes, args)

    if args.num_gpus > 1:
        tar_model = tar_model.module
    ## move to data parallel
    if args.num_div_gpus > 1:
        device = device2
    tar_model = tar_model.to(device).eval()
    if args.tar_pre_trained:
        checkpoint = torch.load(args.tar_pre_trained)
        if 'model' in checkpoint:
            tar_model.load_state_dict(checkpoint['model'])
        elif 'state_dict' in checkpoint:
            tar_model.load_state_dict(checkpoint['state_dict'])
        elif 'model_state' in checkpoint:
            tar_model.load_state_dict(checkpoint['model_state'])
        else:
            tar_model.load_state_dict(checkpoint)
    
    if args.tar_model == 'resnet_50' or args.tar_model == 'mvcnn':
        num_blocks = 1
    else:
        num_blocks = tar_model.depth

    if 'clip' in args.tar_model:
        def zeroshot_classifier(classnames, templates):
            with torch.no_grad():
                zeroshot_weights = []
                for classname in tqdm(classnames):
                    texts = [template.format(classname) for template in templates]  # format with class
                    texts = clip.tokenize(texts).cuda()  # tokenize
                    class_embeddings = tar_model.encode_text(texts)  # embed with text encoder
                    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                    class_embedding = class_embeddings.mean(dim=0)
                    class_embedding /= class_embedding.norm()
                    zeroshot_weights.append(class_embedding)
                zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
            return zeroshot_weights

        zeroshot_weights = zeroshot_classifier(imagenet_classes, imagenet_templates)
    
    if multi_data:
        test_dataset = MultiviewImgDataset(args.test_dir, scale_aug=False, rot_aug=False, num_views=args.num_frames)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_size = len(test_dataset)
        print("SIZE:", test_size)

    elif video_data:
        test_loader, test_size = construct_loader(args)

    # setup test_meter for video multiview testing
        clean_test_meter = TestMeter(
            len(test_loader.dataset)
            // (args.num_temporal_views * args.num_spatial_crops),
            args.num_temporal_views * args.num_spatial_crops,
            args.num_classes,
            len(test_loader),
            False,
            "sum",
            src_model.depth,
        )
        adv_test_meter = TestMeter(
            len(test_loader.dataset)
            // (args.num_temporal_views * args.num_spatial_crops),
            args.num_temporal_views * args.num_spatial_crops,
            args.num_classes,
            len(test_loader),
            False,
            "sum",
            src_model.depth,
        )
        fool_test_meter = CompareMeter(
            len(test_loader.dataset)
            // (args.num_temporal_views * args.num_spatial_crops),
            args.num_temporal_views * args.num_spatial_crops,
            args.num_classes,
            len(test_loader),
            "sum",
            src_model.depth,
        )
        tar_test_meter = TestMeter(
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

    loss_fn_alex = lpips.LPIPS(net='alex', verbose=False).to(device)  # best forward scores

    acc = {}
    for block in range(num_blocks):
        acc[block] = 0

    adv_acc = {}
    for block in range(num_blocks):
        adv_acc[block] = 0

    fool_rate = {}
    for block in range(num_blocks):
        fool_rate[block] = 0

    target_acc = {}
    if args.target_label != -1:
        for block in range(num_blocks):
            target_acc[block] = 0

    distance  = 0 # distance between clean and adversarial image
    ssim_d = 0 # structual similarity
    lpips_d = 0 # perceptual similarity


    with tqdm(enumerate(test_loader), total=len(test_loader)) as prog_bar:
        for idx, image_label in prog_bar:
            if args.num_div_gpus > 1:
                device = device2
            img, label = image_label[0].to(device), image_label[1].to(device)

            if video_data:
                video_idx, meta = image_label[2].to(device), image_label[3]
                for key, val in meta.items():
                    if isinstance(val, (list,)):
                        for i in range(len(val)):
                            val[i] = val[i].to(device)
                    else:
                        meta[key] = val.to(device)

            if 'clip' in args.tar_model:
                image_features = tar_model.encode_image(normalize(img.clone(), mean=src_mean, std=src_std))
                image_features /= image_features.norm(dim=-1, keepdim=True)
                clean_out = 100. * image_features @ zeroshot_weights
            else:
                if multi_data:
                    N,V,C,H,W = img.shape
                    img_v = img.view(-1,C,H,W).to(device)
                    clean_out = tar_model(normalize(img_v.clone(), mean=tar_mean, std=tar_std))
                else:
                    clean_out = tar_model(normalize(img.clone(), mean=tar_mean, std=tar_std))

                if args.num_gpus > 1:
                    clean_out, label, video_idx = all_gather(
                        [clean_out, label, video_idx]
                    )

                if video_data:
                    if isinstance(clean_out, (tuple)):
                        clean_out = clean_out[0]
                    for num_layer in range(len(clean_out)):
                        clean_out[num_layer] = clean_out[num_layer].cpu()
                        clean_out[num_layer] = clean_out[num_layer].detach()

                    label = label.cpu()
                    video_idx = video_idx.cpu()
                    clean_test_meter.update_stats(
                        clean_out, label.detach(), video_idx.detach()
                    )
                    label = label.to(device)

            # for robust models: they return tuple
            if isinstance(clean_out, tuple) and not video_data:
                clean_out = clean_out[0]
            if not isinstance(clean_out, list) and not video_data:
                clean_out = [clean_out]

            if not video_data:
                for block in range(num_blocks):
                    acc[block] += torch.sum(clean_out[block].argmax(dim=-1) == label).item()

            if args.target_label == -1:
                target = None
            else:
                target = torch.LongTensor(img.size(0))
                target.fill_(args.target_label)
                target = target.to(device)

            if args.num_div_gpus > 1:
                img, label = img.to(device1), label.to(device1)
                if target is not None:
                    target = target.to(device1)
            if multi_data:
                adv = Adv_Attack(args.attack_type)(src_model, src_mean, src_std, img.permute(0,2,1,3,4).to(device), label, target, args, True)
        
            else:
                adv = Adv_Attack(args.attack_type)(src_model, src_mean, src_std, img, label, target, args, video_data)

            if 'clip' in args.tar_model:
                image_features = tar_model.encode_image(normalize(adv.clone(), mean=src_mean, std=src_std))
                image_features /= image_features.norm(dim=-1, keepdim=True)
                adv_out = 100. * image_features @ zeroshot_weights
            else:
                if multi_data:
                    adv = adv.permute(0,2,1,3,4)
                    N,V,C,H,W = adv.shape
                    try:
                        adv_v = adv.view(-1,C,H,W).to(device)
                    except:
                        adv_v = adv.reshape(-1,C,H,W).to(device)
                    adv_out = tar_model(normalize(adv_v.clone(), mean=tar_mean, std=tar_std))
                else:
                    adv_out = tar_model(normalize(adv.clone(), mean=tar_mean, std=tar_std))

                if args.num_gpus > 1:
                    adv_out, label, video_idx = all_gather(
                        [adv_out, label, video_idx]
                    )

                if video_data:
                    if isinstance(adv_out, (tuple)):
                        adv_out = adv_out[0]
                    for num_layer in range(len(adv_out)):
                        adv_out[num_layer] = adv_out[num_layer].cpu()
                        adv_out[num_layer] = adv_out[num_layer].detach()

                    label = label.cpu()
                    video_idx = video_idx.cpu()
                    adv_test_meter.update_stats(
                        adv_out, label.detach(), video_idx.detach()
                    )
                    fool_test_meter.update_stats(
                        adv_out, clean_out, video_idx.detach()
                    )
                    if target is not None:
                        tar_test_meter.update_stats(
                            adv_out, target.detach(), video_idx.detach()
                        )
                    label = label.to(device)

            # for robust models: they return tuple
            if isinstance(adv_out, tuple) and not video_data:
                adv_out = adv_out[0]
            if not isinstance(adv_out, list) and not video_data:
                adv_out = [adv_out]
            
            if not video_data:
                for block in range(num_blocks):
                    adv_acc[block] += torch.sum(adv_out[block].argmax(dim=-1) == label).item()

                for block in range(num_blocks):
                    fool_rate[block] += torch.sum(adv_out[block].argmax(dim=-1) != clean_out[block].argmax(dim=-1)).item()

                if target is not None:
                    for block in range(num_blocks):
                        target_acc[block] += torch.sum(adv_out[block].argmax(dim=-1) == target).item()

            distance+=(img-adv).max().item()*255
            if not video_data and not multi_data:
                ssim_d += pytorch_ssim.ssim(img, adv).item()
                lpips_d += loss_fn_alex((2*img-1),(2*adv-1)).view(-1,).mean().item()

            del clean_out, adv_out

    distance = distance/(idx+1)
    ssim_d = ssim_d/(idx+1)
    lpips_d = lpips_d/(idx+1)

    if video_data:
        stats_acc = clean_test_meter.finalize_metrics()
        stats_adv_acc = adv_test_meter.finalize_metrics()
        stats_fool_rate = fool_test_meter.finalize_metrics()
        if target is not None:
            stats_tar_acc = tar_test_meter.finalize_metrics()

    for block in range(num_blocks):
        if video_data:
            acc[block] = float(stats_acc['layer_{}: top1_acc'.format(block + 1)])
        else:
            acc[block] = round(acc[block]/test_size * 100, 3)
    acc['mean'] = round(np.array(list(acc.values())).mean(), 3) # Average accuracy across blocks

    for block in range(num_blocks):
        if video_data:
            adv_acc[block] = float(stats_adv_acc['layer_{}: top1_acc'.format(block + 1)])
        else:
            adv_acc[block] = round(adv_acc[block] / test_size * 100, 3)
    adv_acc['mean'] = round(np.array(list(adv_acc.values())).mean(), 3)  # Average adversaral accuracy across blocks

    for block in range(num_blocks):
        if video_data:
            fool_rate[block] = float(stats_fool_rate['layer_{}: fooled'.format(block + 1)])
        else:
            fool_rate[block] = round(fool_rate[block] / test_size * 100, 3)
    fool_rate['mean'] = round(np.array(list(fool_rate.values())).mean(), 3)  # Average fool rate across blocks


    if target is not None:
        for block in range(num_blocks):
            if video_data:
                target_acc[block] = float(stats_tar_acc['layer_{}: top1_acc'.format(block + 1)])
            else:
                target_acc[block] = round(target_acc[block] / test_size * 100, 3)
        target_acc['mean'] = round(np.array(list(target_acc.values())).mean(), 3)  # Average target accuracy across blocks

    json.dump({"Accuracy": acc,
               "Adv Accuracy": adv_acc,
               "Fool Rate": fool_rate,
               "Target Accuracy": target_acc,
               "L_infty": distance,
               "SSIM": ssim_d,
               "LPIPS": lpips_d,
               },
              open(f"{args.dir}/{args.image_list.split('.')[0]}_results.json", "w"), indent=4)



if __name__ == '__main__':
    args = parse_args()
    if args.num_gpus > 1:
        torch.multiprocessing.spawn(
            mpu.run,
            nprocs=args.num_gpus,
            args=(
                args.num_gpus,
                main,
                "tcp://localhost:9999",
                0,
                1,
                "nccl",
                ()
            ),
            daemon=False,
        )
    main()
