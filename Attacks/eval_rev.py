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
    parser.add_argument('--src_num_cls', type=int, default=400, help='')
    

    return parser.parse_args()

def main():
    # setup run
    # args = parse_args()

    # print("STARTING ATTACK:", args.attack_type)

    # if args.attack_type in ['fgsm', 'rfgsm']:
    #     args.iters = 1 # single step attacks

    # # print(args.replicate_grad)
    # # print(args.no_sup_loss)
    # # print(args.no_unsup_loss)

    # if args.variation == '':
    #     args.dir = f"results_adv/{args.attack_type}/{args.src_model}_{args.index}_{args.data_type}/{args.tar_model}"
    # else:
    #     args.dir = f"results_adv/{args.attack_type}/{args.src_model}_{args.index}_{args.variation}_{args.data_type}/{args.tar_model}"
    # if not os.path.isdir(args.dir):
    #     os.makedirs(args.dir)
    # json.dump(vars(args), open(f"{args.dir}/config.json", "w"), indent=4)

    # ## check if data is of videos
    # video_data = False
    # if args.data_type in ['hmdb51', 'ucf101', 'kinetics', 'ssv2']:
    #     video_data = True

    # # GPU
    # device1 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device2 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # # src_model, src_mean, src_std = get_model(args.src_model, args.src_num_cls, args, is_src=True)
    # # src_model, src_mean, src_std = None, None , None
    # print("SRC Loaded")
    # # if args.num_gpus > 1:
    # #     src_model = src_model.module
    
    # device = device1
    # # src_model = src_model.to(device).eval()
    # # if args.pre_trained:
    # #     checkpoint = torch.load(args.pre_trained)
    # #     if 'model' in checkpoint:
    # #         src_model.load_state_dict(checkpoint['model'])
    # #     elif 'state_dict' in checkpoint:
    # #         src_model.load_state_dict(checkpoint['state_dict'])
    # #     elif 'model_state' in checkpoint:
    # #         src_model.load_state_dict(checkpoint['model_state'])
    # #     else:
    # #         src_model.load_state_dict(checkpoint)

    # tar_model, tar_mean, tar_std = get_model(args.tar_model, args.num_classes, args)
    # print("TAR Loaded")

    # if args.num_gpus > 1:
    #     tar_model = tar_model.module
    # ## move to data parallel
    # if args.num_div_gpus > 1:
    #     device = device2
    # tar_model = tar_model.to(device).eval()
    # if args.tar_pre_trained:
    #     checkpoint = torch.load(args.tar_pre_trained)
    #     if 'model' in checkpoint:
    #         ckpt = checkpoint['model']
    #         # tar_model.load_state_dict(checkpoint['model'])
    #     elif 'state_dict' in checkpoint:
    #         ckpt = checkpoint['state_dict']
    #         # tar_model.load_state_dict(checkpoint['state_dict'])
    #     elif 'model_state' in checkpoint:
    #         ckpt = checkpoint['model_state']
    #         # tar_model.load_state_dict(checkpoint['model_state'])
    #     else:
    #         ckpt = checkpoint
    #         # tar_model.load_state_dict(checkpoint)
    #     # ckpt = {k.replace('model.', ''): v for k, v in ckpt.items()}
    #     tar_model.load_state_dict(ckpt)
        
    # if args.tar_model == 'resnet_50':
    #     num_blocks = 1
    # else:
    #     num_blocks = tar_model.depth

    # if 'clip' in args.tar_model:
    #     def zeroshot_classifier(classnames, templates):
    #         with torch.no_grad():
    #             zeroshot_weights = []
    #             for classname in tqdm(classnames):
    #                 texts = [template.format(classname) for template in templates]  # format with class
    #                 texts = clip.tokenize(texts).cuda()  # tokenize
    #                 class_embeddings = tar_model.encode_text(texts)  # embed with text encoder
    #                 class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
    #                 class_embedding = class_embeddings.mean(dim=0)
    #                 class_embedding /= class_embedding.norm()
    #                 zeroshot_weights.append(class_embedding)
    #             zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    #         return zeroshot_weights

    #     zeroshot_weights = zeroshot_classifier(imagenet_classes, imagenet_templates)
    # for num_test in range(3):
    #     if video_data:
    #         test_loader, test_size = construct_loader(args)

    #         # setup test_meter for video multiview testing
    #         clean_test_meter = TestMeter(
    #             len(test_loader.dataset)
    #             // (args.num_temporal_views * args.num_spatial_crops),
    #             args.num_temporal_views * args.num_spatial_crops,
    #             args.num_classes,
    #             len(test_loader),
    #             False,
    #             "sum",
    #             12,
    #         )
    #         adv_test_meter = TestMeter(
    #             len(test_loader.dataset)
    #             // (args.num_temporal_views * args.num_spatial_crops),
    #             args.num_temporal_views * args.num_spatial_crops,
    #             args.num_classes,
    #             len(test_loader),
    #             False,
    #             "sum",
    #             12,
    #         )
    #         fool_test_meter = CompareMeter(
    #             len(test_loader.dataset)
    #             // (args.num_temporal_views * args.num_spatial_crops),
    #             args.num_temporal_views * args.num_spatial_crops,
    #             args.num_classes,
    #             len(test_loader),
    #             "sum",
    #             12,
    #         )
    #         tar_test_meter = TestMeter(
    #             len(test_loader.dataset)
    #             // (args.num_temporal_views * args.num_spatial_crops),
    #             args.num_temporal_views * args.num_spatial_crops,
    #             args.num_classes,
    #             len(test_loader),
    #             False,
    #             "sum",
    #             12,
    #         )

    #     else:
    #         test_loader, test_size = get_data_loader(args)

    #     loss_fn_alex = lpips.LPIPS(net='alex', verbose=False).to(device)  # best forward scores

    #     acc = {}
    #     for block in range(num_blocks):
    #         acc[block] = 0

    #     adv_acc = {}
    #     for block in range(num_blocks):
    #         adv_acc[block] = 0

    #     fool_rate = {}
    #     for block in range(num_blocks):
    #         fool_rate[block] = 0

    #     target_acc = {}
    #     if args.target_label != -1:
    #         for block in range(num_blocks):
    #             target_acc[block] = 0

    #     distance  = 0 # distance between clean and adversarial image
    #     ssim_d = 0 # structual similarity
    #     lpips_d = 0 # perceptual similarity


    #     with tqdm(enumerate(test_loader), total=len(test_loader)) as prog_bar:
    #         for idx, image_label in prog_bar:
    #             if args.num_div_gpus > 1:
    #                 device = device2

    #             img, label = image_label[0].to(device), image_label[1].to(device)
    #             if num_test >= 0:
    #                 idx_tmp = torch.randperm(img.shape[2])
    #                 img = img[:,:,idx_tmp,:,:]

    #             if video_data:
    #                 video_idx, meta = image_label[2].to(device), image_label[3]
    #                 for key, val in meta.items():
    #                     if isinstance(val, (list,)):
    #                         for i in range(len(val)):
    #                             val[i] = val[i].to(device)
    #                     else:
    #                         meta[key] = val.to(device)

    #             if 'clip' in args.tar_model:
    #                 image_features = tar_model.encode_image(normalize(img.clone(), mean=src_mean, std=src_std))
    #                 image_features /= image_features.norm(dim=-1, keepdim=True)
    #                 clean_out = 100. * image_features @ zeroshot_weights
    #             else:
    #                 clean_out = tar_model(normalize(img.clone(), mean=tar_mean, std=tar_std))

    #                 if args.num_gpus > 1:
    #                     clean_out, label, video_idx = all_gather(
    #                         [clean_out, label, video_idx]
    #                     )

    #                 if video_data:
    #                     if isinstance(clean_out, (tuple)):
    #                         clean_out = clean_out[0]
    #                     for num_layer in range(len(clean_out)):
    #                         clean_out[num_layer] = clean_out[num_layer].cpu()
    #                         clean_out[num_layer] = clean_out[num_layer].detach()

    #                     label = label.cpu()
    #                     video_idx = video_idx.cpu()
    #                     clean_test_meter.update_stats(
    #                         clean_out, label.detach(), video_idx.detach()
    #                     )
    #                     label = label.to(device)

    #             # for robust models: they return tuple
    #             if isinstance(clean_out, tuple) and not video_data:
    #                 clean_out = clean_out[0]
    #             if not isinstance(clean_out, list) and not video_data:
    #                 clean_out = [clean_out]

    #             if not video_data:
    #                 for block in range(num_blocks):
    #                     acc[block] += torch.sum(clean_out[block].argmax(dim=-1) == label).item()

    #             if args.target_label == -1:
    #                 target = None
    #             else:
    #                 target = torch.LongTensor(img.size(0))
    #                 target.fill_(args.target_label)
    #                 target = target.to(device)
    #             # if idx == 0:
    #             #     vutils.save_image(vutils.make_grid(img[:,:,0,:,:], normalize=True, scale_each=True), f'org.png')
    #             if args.num_div_gpus > 1:
    #                 img, label = img.to(device1), label.to(device1)
    #                 if target is not None:
    #                     target = target.to(device1)
    #             adv = img.clone()
    #             # adv = Adv_Attack(args.attack_type)(src_model, src_mean, src_std, img, label, target, args, video_data)
    #             # if idx == 0:
    #             #     vutils.save_image(vutils.make_grid(adv[:,:,0,:,:], normalize=True, scale_each=True), f'adv.png')
    #             #     break

    #             if 'clip' in args.tar_model:
    #                 image_features = tar_model.encode_image(normalize(adv.clone(), mean=src_mean, std=src_std))
    #                 image_features /= image_features.norm(dim=-1, keepdim=True)
    #                 adv_out = 100. * image_features @ zeroshot_weights
    #             else:
    #                 adv_out = tar_model(normalize(adv.clone(), mean=tar_mean, std=tar_std))
    #                 # adv_out = adv.clone()
    #                 # adv_out = clean

    #                 if args.num_gpus > 1:
    #                     adv_out, label, video_idx = all_gather(
    #                         [adv_out, label, video_idx]
    #                     )

    #                 if video_data:
    #                     if isinstance(adv_out, (tuple)):
    #                         adv_out = adv_out[0]
    #                     for num_layer in range(len(adv_out)):
    #                         adv_out[num_layer] = adv_out[num_layer].cpu()
    #                         adv_out[num_layer] = adv_out[num_layer].detach()

    #                     label = label.cpu()
    #                     video_idx = video_idx.cpu()
    #                     adv_test_meter.update_stats(
    #                         adv_out, label.detach(), video_idx.detach()
    #                     )
    #                     fool_test_meter.update_stats(
    #                         adv_out, clean_out, video_idx.detach()
    #                     )
    #                     if target is not None:
    #                         tar_test_meter.update_stats(
    #                             adv_out, target.detach(), video_idx.detach()
    #                         )
    #                     label = label.to(device)

    #             # for robust models: they return tuple
    #             if isinstance(adv_out, tuple) and not video_data:
    #                 adv_out = adv_out[0]
    #             if not isinstance(adv_out, list) and not video_data:
    #                 adv_out = [adv_out]
                
    #             if not video_data:
    #                 for block in range(num_blocks):
    #                     adv_acc[block] += torch.sum(adv_out[block].argmax(dim=-1) == label).item()

    #                 for block in range(num_blocks):
    #                     fool_rate[block] += torch.sum(adv_out[block].argmax(dim=-1) != clean_out[block].argmax(dim=-1)).item()

    #                 if target is not None:
    #                     for block in range(num_blocks):
    #                         target_acc[block] += torch.sum(adv_out[block].argmax(dim=-1) == target).item()

    #             distance+=(img-adv).max().item()*255
    #             if not video_data:
    #                 ssim_d += pytorch_ssim.ssim(img, adv).item()
    #                 lpips_d += loss_fn_alex((2*img-1),(2*adv-1)).view(-1,).mean().item()

    #             if idx==20:
    #                 if not video_data:
    #                     vutils.save_image(vutils.make_grid(adv, normalize=True, scale_each=True), f'{args.dir}/adv.png')
    #                 else: 
    #                     print(adv.shape, img.shape)
    #                     vutils.save_image(vutils.make_grid(adv[:,:,0,:,:], normalize=True, scale_each=True), f'{args.dir}/adv.png')
    #                     vutils.save_image(vutils.make_grid(img[:,:,0,:,:], normalize=True, scale_each=True), f'{args.dir}/org.png')
    #             del clean_out, adv_out

    #     distance = distance/(idx+1)
    #     ssim_d = ssim_d/(idx+1)
    #     lpips_d = lpips_d/(idx+1)

    #     if video_data:
    #         stats_acc = clean_test_meter.finalize_metrics()
    #         stats_adv_acc = adv_test_meter.finalize_metrics()
    #         stats_fool_rate = fool_test_meter.finalize_metrics()
    #         if target is not None:
    #             stats_tar_acc = tar_test_meter.finalize_metrics()

    #     for block in range(num_blocks):
    #         if video_data:
    #             acc[block] = float(stats_acc['layer_{}: top1_acc'.format(block + 1)])
    #         else:
    #             acc[block] = round(acc[block]/test_size * 100, 3)
    #     acc['mean'] = round(np.array(list(acc.values())).mean(), 3) # Average accuracy across blocks

    #     for block in range(num_blocks):
    #         if video_data:
    #             adv_acc[block] = float(stats_adv_acc['layer_{}: top1_acc'.format(block + 1)])
    #         else:
    #             adv_acc[block] = round(adv_acc[block] / test_size * 100, 3)
    #     adv_acc['mean'] = round(np.array(list(adv_acc.values())).mean(), 3)  # Average adversaral accuracy across blocks

    #     for block in range(num_blocks):
    #         if video_data:
    #             fool_rate[block] = float(stats_fool_rate['layer_{}: fooled'.format(block + 1)])
    #         else:
    #             fool_rate[block] = round(fool_rate[block] / test_size * 100, 3)
    #     fool_rate['mean'] = round(np.array(list(fool_rate.values())).mean(), 3)  # Average fool rate across blocks


    #     if target is not None:
    #         for block in range(num_blocks):
    #             if video_data:
    #                 target_acc[block] = float(stats_tar_acc['layer_{}: top1_acc'.format(block + 1)])
    #             else:
    #                 target_acc[block] = round(target_acc[block] / test_size * 100, 3)
    #         target_acc['mean'] = round(np.array(list(target_acc.values())).mean(), 3)  # Average target accuracy across blocks
    #     print("Accuracy:", acc)
    #     # json.dump({"Accuracy": acc,
    #     #         "Adv Accuracy": adv_acc,
    #     #         "Fool Rate": fool_rate,
    #     #         "Target Accuracy": target_acc,
    #     #         "L_infty": distance,
    #     #         "SSIM": ssim_d,
    #     #         "LPIPS": lpips_d,
    #     #         },
    #     #         open(f"{args.dir}/{args.image_list.split('.')[0]}_results.json", "w"), indent=4)
    
    # load model
    tar_model, tar_mean, tar_std = get_model(args.tar_model, args.num_classes, args)
    print("TAR Loaded")

    if args.num_gpus > 1:
        tar_model = tar_model.module
    ## move to data parallel
    # if args.num_div_gpus > 1:
    #     device = device2
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tar_model = tar_model.to(device).eval()
    if args.tar_pre_trained:
        checkpoint = torch.load(args.tar_pre_trained)
        if 'model' in checkpoint:
            ckpt = checkpoint['model']
            # tar_model.load_state_dict(checkpoint['model'])
        elif 'state_dict' in checkpoint:
            ckpt = checkpoint['state_dict']
            # tar_model.load_state_dict(checkpoint['state_dict'])
        elif 'model_state' in checkpoint:
            ckpt = checkpoint['model_state']
            # tar_model.load_state_dict(checkpoint['model_state'])
        else:
            ckpt = checkpoint
            # tar_model.load_state_dict(checkpoint)
        # ckpt = {k.replace('model.', ''): v for k, v in ckpt.items()}
        tar_model.load_state_dict(ckpt)

        test_loader, test_size = construct_loader(args)

        path = '../../../data/drive_2/ssv2/labels/labels.json'
        with open(path) as json_file:
            data = json.load(json_file)

        labels = {int(v):k for k,v in data.items()}

        cor = 0
        tot = 0
        with tqdm(enumerate(test_loader), total=len(test_loader)) as prog_bar:
            for idx, image_label in prog_bar:
                img, label = image_label[0].to(device), image_label[1].to(device)

                if idx == 0:
                    print(img.shape, label.shape)
                # frame_idx = torch.randperm(img.shape[2])
                # img = img[:,:,frame_idx,:,:]

                out = tar_model(img)
                # print(out.shape)
                # return
                # # print(out[0][-1][0].shape, label.shape)
                out = out[0].argmax()
                # # break
                # if out == label:
                #     cor += 1
                # tot += 1

                # if idx % 100 == 0:
                #     print("Accuracy:", cor/tot * 100)
            
                if out == label:
                    tot+=1
                    
                    # save image as png
                    # print(img.permute(2,0,1,3,4).squeeze(1).shape)
                    # vutils.save_image(vutils.make_grid(img.permute(2,0,1,3,4).squeeze(1), normalize=True, scale_each=True), f'img_{labels[label.item()]}.png')
                    
                    # reverse image
                    frame_idx = list(range(img.shape[2]))
                    frame_idx.reverse()
                    # print(frame_idx)
                    img_rev = img[:,:,frame_idx,:,:]

                    out_rev = tar_model(img_rev)[0].argmax()
                    # vutils.save_image(vutils.make_grid(img_rev.permute(2,0,1,3,4).squeeze(1), normalize=True, scale_each=True), f'img_rev_{labels[out_rev.item()]}.png')
                    # if out_rev == label:
                    #     print("Reverse image is also correct")
                    # # print(out_rev, label)
                    # else:
                    #     print("Reverse image predicted", out_rev, "but label is", label)
                    if out_rev != label:
                        cor+=1
                        # print("Reverse image predicted", out_rev, "but label is", label)
                        # vutils.save_image(vutils.make_grid(img_rev.permute(2,0,1,3,4).squeeze(1), normalize=True, scale_each=True), f'img_rev_{labels[out_rev.item()]}.png')

                        # vutils.save_image(vutils.make_grid(img.permute(2,0,1,3,4).squeeze(1), normalize=True, scale_each=True), f'img_{labels[label.item()]}.png')
                        # tot += 1


                    # frame_idx = torch.randperm(img.shape[2])
                    # img_rand = img[:,:,frame_idx,:,:]
                    # vutils.save_image(vutils.make_grid(img_rand.permute(2,0,1,3,4).squeeze(1), normalize=True, scale_each=True), f'img_rand.png')

                    # out_rand = tar_model(img_rand)[0][-1][0].argmax()
                    # if out_rand == label:
                    #     print("Random image is also correct")

                    # if tot > 5:
                    #     break
                if idx % 100 == 1:
                    print(cor ,"out of", tot, "are different")


                    








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
