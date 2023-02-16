import numpy as np
import torch
import torch.nn.functional as F
from einops import reduce, rearrange
import torch.optim as optim


import random
from ..utils import normalize
from .attack_utils import *
import torchvision.utils as vutils


def get_gpu_memory(ind):
    stats = {}
    t = torch.cuda.get_device_properties(ind).total_memory
    r = torch.cuda.memory_reserved(ind)
    a = torch.cuda.memory_allocated(ind)
    f = r-a  # free inside reserved
    stats["total"], stats["reserved"], stats["allocated"], stats["free"] = t, r, a, f
    return stats

def base_adv(model, mean, std, img, label, target, args, video_data):

    criterion = torch.nn.CrossEntropyLoss()
    similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
    iterations = args.iter
    eps = args.eps/255
    attack_type = args.attack_type
    index = args.index
    size = img.shape[0]

    if video_data and args.src_frames == 1:
        out_img, features_img = model(normalize(img[:,:,args.num_frames // 2,:,:].unsqueeze(2).clone(), mean=mean, std=std), return_tokens=True)
    else:
        out_img, features_img = model(normalize(img.clone(), mean=mean, std=std), return_tokens=True)

    multiplier = 1 # Gradient ascent
    if target is not None:
        label = target
        multiplier = -1 # Gradient descent

    if attack_type == 'pifgsm':
        amp=10 # default attack parameters
        alpha_beta = (2/255) * amp
        gamma = alpha_beta
        amplification = 0.0

    if video_data and args.src_frames == 1:
        adv = img[:,:,args.num_frames // 2,:,:].unsqueeze(2).detach()
        adv_final = img.detach()
    else:
        adv = img.detach()
        adv_final = None

    if attack_type == 'rfgsm':
        alpha = 2 / 255
        adv = adv + alpha * torch.randn(img.shape).cuda().detach().sign()
        eps = eps - alpha

    adv.requires_grad = True

    adv_noise = 0
    for j in range(iterations):
        if attack_type in ['dim', 'pifgsm']:
            if video_data:
                adv_r = torch.Tensor().to(img.device)
                for num_frame in range(adv.shape[2]):
                    adv_tmp = input_diversity(adv[:, :,num_frame, :, :]).unsqueeze(2)
                    adv_r = torch.cat((adv_r,adv_tmp), dim=2)
            else:
                adv_r = input_diversity(adv)
        else:
            adv_r = adv

        # print(get_gpu_memory(0)["free"] / (1024 ** 3))
        out_adv, features_adv = model(normalize(adv_r.clone(), mean=mean, std=std), return_tokens=True)

        # loss = 0
        if isinstance(out_adv, list) and index == 'all':
            loss_sup = 0
            loss_unsup = 0
            for idx in range(len(out_adv)):
                loss_sup += criterion(out_adv[idx], label)
                loss_unsup = similarity(features_adv[idx].view(size, -1),
                                        features_img[idx].view(size, -1)).mean()
        elif isinstance(out_adv, list) and index == 'last':
            loss_sup = criterion(out_adv[-1], label)
            loss_unsup = similarity(features_adv[-1].view(size, -1),
                                     features_img[-1].view(size, -1)).mean()
        else:
            loss = criterion(out_adv, label)

        if args.no_sup_loss:
            loss_sup = 0
        if args.no_unsup_loss:
            loss_unsup = 0

        loss = loss_sup - loss_unsup
        loss.backward(retain_graph=True)
        if video_data:
            grads = torch.Tensor().to(img.device)
            for i_grad in range(adv.grad.shape[2]):
                # print(args.replicate_grad)
                if args.replicate_grad:
                    tmp = i_grad
                    i_grad = args.num_frames // 2
                grad = F.conv2d(adv.grad[:, :, i_grad, :, :], gaussian_kernel, bias=None, stride=1, padding=(2,2), groups=3)
                if args.replicate_grad:
                    i_grad = tmp
                grads = torch.cat((grads, grad.unsqueeze(2)), dim=2)
            # print(adv.data.shape, grads.shape)
            adv.grad = grads
            del grads
        else:
            adv.grad = F.conv2d(adv.grad, gaussian_kernel, bias=None, stride=1, padding=(2, 2), groups=3)

        # print(adv.requires_grad)
        if attack_type == 'mifgsm' or attack_type == 'dim':
            if not video_data:
                adv.grad = adv.grad / torch.mean(torch.abs(adv.grad), dim=(1, 2, 3), keepdim=True)
            else:
                adv.grad = adv.grad / torch.mean(torch.abs(adv.grad), dim=(1, 2, 3, 4), keepdim=True)
            adv_noise = adv_noise + adv.grad
        else:
            adv_noise = adv.grad



        if attack_type == 'pifgsm':
            amplification += alpha_beta * adv_noise.sign()
            cut_noise = torch.clamp(abs(amplification) - eps, 0, 10000.0) * torch.sign(amplification)
            projection_val = gamma * torch.sign(project_noise(cut_noise, stack_kern, padding_size))
            amplification += projection_val
            adv.data = adv.data + multiplier*alpha_beta * adv_noise.sign() + projection_val
            if video_data and args.src_frames == 1:
                adv_final = adv_final + multiplier*alpha_beta * adv_noise.sign() + projection_val
        else:
            adv.data = adv.data + multiplier * adv_noise.sign()
            if video_data and args.src_frames == 1:
                adv_final = adv_final + multiplier * adv_noise.sign()
        
        # vutils.save_image(vutils.make_grid(adv[:,:,0,:,:], normalize=True, scale_each=True), f'adv_before_project.png')
        if video_data and args.src_frames == 1:
            projection_operator(adv, img[:,:,args.num_frames // 2,:,:].unsqueeze(2), eps)
            projection_operator(adv_final, img, eps)
        else:
            projection_operator(adv, img, eps)
        
        # for iterr in range(8):
        #     print(iterr, adv.data[:,:,iterr,:,:].mean())

        adv.data.clamp_(img.min(), img.max())
        if video_data and args.src_frames == 1:
            adv_final.clamp_(img.min(), img.max())
        adv.grad.data.zero_()
    
    if not adv_final == None:
        assert torch.eq(adv_final[:,:,args.num_frames // 2,:,:], adv.squeeze(2)).all()
        adv = adv_final
    return adv.detach()


