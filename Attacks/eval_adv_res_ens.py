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

from pathways.utils import get_model, get_data_loader, normalize, get_attn_maps, plot_joint_overlay
from pathways.attacks import Adv_Attack

from pathways.clip_models.clip_utils import imagenet_classes, imagenet_templates
from pathways.clip_models import clip

import pytorch_ssim
import lpips
import helper


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation of Vision Models')
    parser.add_argument('--test_dir', default='IN/val', help='ImageNet Validation Data Set')
    parser.add_argument('--image_list', default=None,
                        help='Image List from Validation Data stored as json file in data folder')
    parser.add_argument('--data_type', default='IN', help='ImageNet, CIFAR10/100')
    parser.add_argument('--src_model', type=str, default='deit_small_patch16_224', help='Source Model Name')
    parser.add_argument('--tar_model', type=str, default='deit_small_patch16_224', help='Source Model Name')
    parser.add_argument('--scale_size', type=int, default=256, help='')
    parser.add_argument('--img_size', type=int, default=224, help='')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch Size')

    # Transformer specific parameters
    parser.add_argument('--num_classes', type=int, default=1000, help='Number of output classes')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')
    parser.add_argument('--attn-drop-rate', type=float, default=0.0, metavar='PCT', help='Attention Drop rate ')

    # Attack Parameters
    parser.add_argument('--attack_type', type=str, default='dim',
                        help='Type of baseline attacks: fgsm, rfgsm, pgd, mifgsm, dim')
    parser.add_argument('--eps', type=int, default=16, help='Perturbation Budget')
    parser.add_argument('--iter', type=int, default=10, help='Attack iterations')
    parser.add_argument('--index', type=str, default='all', help='last(final classifier) or all(self-ensemble)')
    parser.add_argument('--target_label', type=int, default=-1, help='-1(untarget), 0,1,2...999')

    # Model at Resolution
    parser.add_argument('--res', type=str, default='224', help='Resolution Ensemble: 56, 96, 120, 224, 56_96, 56_96_120, 56_96_120_224')
    return parser.parse_args()


def main():
    # setup run
    args = parse_args()

    args.dir = f"results_adv/{args.attack_type}/{args.src_model}_{args.res}_{args.index}/{args.tar_model}"
    if not os.path.isdir(args.dir):
        os.makedirs(args.dir)
    json.dump(vars(args), open(f"{args.dir}/config.json", "w"), indent=4)

    # GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    forward_pass, src_mean, src_std = helper.get_resolution_ensemble(args)

    # res_list = args.res.split('_')

    # all_models = {}
    # for res in res_list:
    #     if res !='224':
    #         all_models[args.src_model+'_'+res] = get_model(args.src_model, args.num_classes, args)
    #         checkpoint = torch.load(f"pretrained_models/{args.src_model+'_'+res}/checkpoint.pth")
    #         all_models[args.src_model+'_'+res][0].load_state_dict(checkpoint['model'])
    #         all_models[args.src_model+'_'+res][0].to(device).eval()
    #     else:
    #         all_models[args.src_model + '_' + res] = get_model(args.src_model[0:-8], args.num_classes, args)
    #         all_models[args.src_model + '_' + res][0].to(device).eval()
    #
    # # All source models are trained with the same mean and std
    # src_mean = all_models[list(all_models.keys())[0]][1]
    # src_std = all_models[list(all_models.keys())[0]][2]
    #
    # def forward_pass(image):
    #     out = [0]
    #     for res in res_list:
    #         return out + all_models[args.src_model + '_' + res][0](image)

    # Load models
    # src_model_56, src_mean, src_std = get_model(args.src_model, args.num_classes, args)
    # checkpoint = torch.load("pretrained_models/deit_base_patch16_224_resPT_1_56/checkpoint.pth")
    # src_model_56.load_state_dict(checkpoint['model'])
    # src_model_56 = src_model_56.to(device).eval()
    #
    # src_model_96, _, _ = get_model(args.src_model, args.num_classes, args)
    # checkpoint = torch.load("pretrained_models/deit_base_patch16_224_resPT_1_96/checkpoint.pth")
    # src_model_96.load_state_dict(checkpoint['model'])
    # src_model_96 = src_model_96.to(device).eval()
    #
    # src_model_120, _, _ = get_model(args.src_model, args.num_classes, args)
    # checkpoint = torch.load("pretrained_models/deit_base_patch16_224_resPT_1_120/checkpoint.pth")
    # src_model_120.load_state_dict(checkpoint['model'])
    # src_model_120 = src_model_120.to(device).eval()
    #
    # src_model_224, _, _ = get_model("deit_base_patch16_224", args.num_classes, args)
    # src_model_224 = src_model_224.to(device).eval()
    #
    # if args.res == '56':
    #     def forward_pass(image):
    #         return src_model_56(image)
    # elif args.res == '56_96':
    #     def forward_pass(image):
    #         out_src_model_56 = src_model_56(image)
    #         out_src_model_96 = src_model_96(image)
    #         return [x + y  for x, y in zip(out_src_model_56, out_src_model_96)]
    # elif args.res == '56_96_120':
    #     def forward_pass(image):
    #         return src_model_56(image) + src_model_96(image) + src_model_120(image)
    # elif args.res == '56_96_120_224':
    #     def forward_pass(image):
    #         return src_model_56(image) + src_model_96(image) + src_model_120(image) + src_model_224(image)

    tar_model, tar_mean, tar_std = get_model(args.tar_model, args.num_classes, args)
    tar_model = tar_model.to(device).eval()
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

    distance = 0  # distance between clean and adversarial image
    ssim_d = 0  # structual similarity
    lpips_d = 0  # perceptual similarity

    with tqdm(enumerate(test_loader), total=len(test_loader)) as prog_bar:
        for idx, (img, label) in prog_bar:
            img, label = img.to(device), label.to(device)

            if 'clip' in args.tar_model:
                image_features = tar_model.encode_image(normalize(img.clone(), mean=src_mean, std=src_std))
                image_features /= image_features.norm(dim=-1, keepdim=True)
                clean_out = 100. * image_features @ zeroshot_weights
            else:
                clean_out = tar_model(normalize(img.clone(), mean=tar_mean, std=tar_std))

            # for robust models: they return tuple
            if isinstance(clean_out, tuple):
                clean_out = clean_out[0]
            if not isinstance(clean_out, list):
                clean_out = [clean_out]

            for block in range(num_blocks):
                acc[block] += torch.sum(clean_out[block].argmax(dim=-1) == label).item()

            if args.target_label == -1:
                target = None
            else:
                target = torch.LongTensor(img.size(0))
                target.fill_(args.target_label)
                target = target.to(device)

            adv = Adv_Attack(args.attack_type)(forward_pass, src_mean, src_std, img, label, target, args)

            if 'clip' in args.tar_model:
                image_features = tar_model.encode_image(normalize(adv.clone(), mean=src_mean, std=src_std))
                image_features /= image_features.norm(dim=-1, keepdim=True)
                adv_out = 100. * image_features @ zeroshot_weights
            else:
                adv_out = tar_model(normalize(adv.clone(), mean=tar_mean, std=tar_std))

            # for robust models: they return tuple
            if isinstance(adv_out, tuple):
                adv_out = adv_out[0]
            if not isinstance(adv_out, list):
                adv_out = [adv_out]
            for block in range(num_blocks):
                adv_acc[block] += torch.sum(adv_out[block].argmax(dim=-1) == label).item()

            for block in range(num_blocks):
                fool_rate[block] += torch.sum(adv_out[block].argmax(dim=-1) != clean_out[block].argmax(dim=-1)).item()

            if target is not None:
                for block in range(num_blocks):
                    target_acc[block] += torch.sum(adv_out[block].argmax(dim=-1) == target).item()

            distance += (img - adv).max().item() * 255
            ssim_d += pytorch_ssim.ssim(img, adv).item()
            lpips_d += loss_fn_alex((2 * img - 1), (2 * adv - 1)).view(-1, ).mean().item()

            if idx == 0:
                vutils.save_image(vutils.make_grid(adv, normalize=True, scale_each=True), f'{args.dir}/adv.png')
            del clean_out, adv_out

    distance = distance / (idx + 1)
    ssim_d = ssim_d / (idx + 1)
    lpips_d = lpips_d / (idx + 1)

    for block in range(num_blocks):
        acc[block] = round(acc[block] / test_size * 100, 3)
    acc['mean'] = round(np.array(list(acc.values())).mean(), 3)  # Average accuracy across blocks

    for block in range(num_blocks):
        adv_acc[block] = round(adv_acc[block] / test_size * 100, 3)
    adv_acc['mean'] = round(np.array(list(adv_acc.values())).mean(), 3)  # Average adversaral accuracy across blocks

    for block in range(num_blocks):
        fool_rate[block] = round(fool_rate[block] / test_size * 100, 3)
    fool_rate['mean'] = round(np.array(list(fool_rate.values())).mean(), 3)  # Average fool rate across blocks

    if target is not None:
        for block in range(num_blocks):
            target_acc[block] = round(target_acc[block] / test_size * 100, 3)
        target_acc['mean'] = round(np.array(list(target_acc.values())).mean(),
                                   3)  # Average target accuracy across blocks

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
    main()
