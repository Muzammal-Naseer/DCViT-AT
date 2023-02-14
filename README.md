# Boosting Adversarial Transferability using Dynamic Cues (ICLR'23)

[Muzammal Naseer](https://scholar.google.ch/citations?user=tM9xKA8AAAAJ&hl=en),
[Ahmad Mahmood](https://www.linkedin.com/in/ahmad-mahmood-81339a212/),
[Salman Khan](https://scholar.google.com/citations?user=M59O9lkAAAAJ&hl=en), &
[Fahad Khan](https://scholar.google.ch/citations?user=zvaeYnUAAAAJ&hl=en&oi=ao)

[Paper](https://openreview.net/forum?id=SZynfVLGd5) ([arxiv](soon)), [Reviews & Response](https://openreview.net/forum?id=SZynfVLGd5), [Video Presentation](soon), [Poster](soon)
#

> **Abstract:** 
*The transferability of adversarial perturbations between image models has been extensively studied. In this case, an attack is generated from a known surrogate \eg, the ImageNet trained model, and transferred to change the decision of an unknown (black-box) model trained on an image dataset. However, attacks generated from image models do not capture the dynamic nature of a moving object or a changing scene due to a lack of temporal cues within image models. This leads to reduced transferability of adversarial attacks from representation-enriched \emph{image} models such as Supervised Vision Transformers (ViTs), Self-supervised ViTs (\eg, DINO), and Vision-language models (\eg, CLIP) to black-box \emph{video} models. In this work, we induce dynamic cues within the image models without sacrificing their original performance on images. To this end, we optimize \emph{temporal prompts} through frozen image models to capture motion dynamics. Our temporal prompts are the result of a learnable transformation that allows optimizing for temporal gradients during an adversarial attack to fool the motion dynamics. Specifically, we introduce spatial (image) and temporal (video) cues within the same source model through task-specific prompts. Attacking such prompts maximizes the adversarial transferability from image-to-video and image-to-image models using the attacks designed for image models. As an example, an iterative attack launched from image model Deit-B with temporal prompts reduces generalization (top1 \% accuracy) of a video model by 35\% on Kinetics-400. Our approach also improves adversarial transferability to image models by 9\% on ImageNet w.r.t the current state-of-the-art approach. Our attack results indicate that the attacker does not need specialized architectures, \eg, divided space-time attention, 3D convolutions, or multi-view convolution networks for different data modalities. Image models are effective surrogates to optimize an adversarial attack to fool black-box models in a changing environment over time. Code is available at \url{https://bit.ly/3Xd9gRQ}.* 
>

![demo](.github/demo.png)

#
<hr>

## Main Message & Highlights


1. The attacker does not need specialized architectures, e.g., divided space-time attention, 3D convolutions, or multi-view convolution networks for different data modalities. Image models are effective surrogates to optimize an adversarial attack to fool black-box models in a changing environment over time.

2. We introduce dynamic cues within frozen image models without losing the original image representation (e.g. generalization on ImageNet). Both image and video representations enhance adversarial transferability from our adapted image models. For example, a pre-trained ImageNet ViT with approximately 6 million parameters exhibits 44.6 and 72.2 top-1 (%) accuracy on Kinetics-400 and ImageNet validation sets using our approach.

3. Our approach simply augments the existing adversarial attacks developed for image models to fool video or multi-view models.

4. We analyze three types of training schemes (supervised, self-supervised, and text-supervised) and highlight new insights into the adversarial space of vision-language models.

5. We further anlayze the textual bias within vision-language model, CLIP, for the low adversarial transferability to vision models.

6. We observe highly transferable adversaial space of self-supervised vision transformer models such as DINO. 

#
<hr>

## Contents

1. [News](#News)
2. [Setup](#Setup)
2. [Model Zoo](#Model-Zoo)
   * [Pre-trained Image Models-Supervised](#Pre-trained-Image-Models-Supervised)
   * [Pre-trained Image Models-Self Supervised](#Pre-trained-Image-Models-Self-Supervised)
   * [Pre-trained Image-Language Models-Text Supervised](#Pre-trained-Image-Language-Models-Text-Supervised)
3. [Training for Dynamic Cues for Videos](#Training-for-Dynamic-Cues-for-Videos)
4. [Training for Dynamic Cues for Multi-Views-ModelNet40](#Training-for-Dynamic-Cues-for-Multi-Views-ModelNet40)
5. [Evaluation](#Evaluation)
6. [Attack Image2Videos](#Attack-Image2Videos)
7. [References](#References)

<hr>

## News

### (January 21, 2023)
* Our paper is accepted as a conference paper at [ICLR 2023](https://openreview.net/forum?id=SZynfVLGd5)


### (February 15, 2023)
* Pretrained weights released.
  * Kinetics-400
    * ```DeiT-tiny``` - 44.6
    * ```DeiT-small``` - 53.0
    * ```DeiT-base``` - 57.0
    * ```Dino``` - 57.4
    * ```Clip``` - 67.3
  * HMDB
    * ```DeiT-tiny``` - 36.2
    * ```DeiT-small``` - 44.6
    * ```DeiT-base``` - 47.7
    * ```Dino``` - 45.1
    * ```Clip``` - 54.6
  * UCF
    * ```DeiT-tiny``` - 70.0
    * ```DeiT-small``` - 77.2
    * ```DeiT-base``` - 81.4
    * ```Dino``` - 79.5
    * ```Clip``` - 86.0
  * SSV2
    * ```DeiT-tiny``` - 11.2
    * ```DeiT-small``` - 15.3
    * ```DeiT-base``` - 17.5
    * ```Dino``` - 17.4
    * ```Clip``` - 19.9
  * Shadow - ModelNet40
    * ```DeiT-tiny``` - 81.0
    * ```DeiT-small``` - 86.2
    * ```DeiT-base``` - 88.2
    * ```Dino``` - 89.8
    * ```Clip``` - 88.9
  * Depth - ModelNet40
    * ```DeiT-tiny``` - 86.0
    * ```DeiT-small``` - 86.6
    * ```DeiT-base``` - 90.1
    * ```Dino``` - 90.1
    * ```Clip``` - 89.5
<hr>

## Setup

<hr>

## Model Zoo

### Pre-trained Image Models-Supervised

| Dataset       | Input Size | Model | Pretrained Weights   |
|:--------------|:----------:|:-----:|:--------------------------------------------------------------------------------------------------------------------------:|
|    ImageNet & Kinetics-400    |   224 x 224 & 8x224x224     |  DeiT-tiny  | [Link]()  |
|    ImageNet & Hmdb-51    |   224 x 224 & 8x224x224     |  DeiT-tiny  | [Link]()  |
|    ImageNet & Ucf-101    |   224 x 224 & 8x224x224     |  DeiT-tiny  | [Link]()  |
|    ImageNet & Ssv2    |   224 x 224 & 8x224x224     |  DeiT-tiny  | [Link]()  |
|    ImageNet & Kinetics-400    |   224 x 224 & 8x224x224     |  DeiT-small  | [Link]()  |
|    ImageNet & Hmdb-51    |   224 x 224 & 8x224x224     |  DeiT-small  | [Link]()  |
|    ImageNet & Ucf-101    |   224 x 224 & 8x224x224     |  DeiT-small  | [Link]()  |
|    ImageNet & Ssv2    |   224 x 224 & 8x224x224     |  DeiT-small  | [Link]()  |
|    ImageNet & Kinetics-400    |   224 x 224 & 8x224x224     |  DeiT-base  | [Link]()  |
|    ImageNet & Hmdb-51    |   224 x 224 & 8x224x224     |  DeiT-base  | [Link]()  |
|    ImageNet & Ucf-101    |   224 x 224 & 8x224x224     |  DeiT-base  | [Link]()  |
|    ImageNet & Ssv2    |   224 x 224 & 8x224x224     |  DeiT-base  | [Link]()  |


### Pre-trained Image Models-Self Supervised

| Dataset       | Input Size | Model | Pretrained Weights   |
|:--------------|:----------:|:-----:|:--------------------------------------------------------------------------------------------------------------------------:|
|    ImageNet & Kinetics-400    |   224 x 224 & 8x224x224     |  Dino | [Link]()  |
|    ImageNet & Hmdb-51    |   224 x 224 & 8x224x224     |  Dino  | [Link]()  |
|    ImageNet & Ucf-101    |   224 x 224 & 8x224x224     |  Dino  | [Link]()  |
|    ImageNet & Ssv2    |   224 x 224 & 8x224x224     |  Dino  | [Link]()  |

### Pre-trained Image-Language Models-Text Supervised

| Dataset       | Input Size | Model | Pretrained Weights   |
|:--------------|:----------:|:-----:|:--------------------------------------------------------------------------------------------------------------------------:|
|    ImageNet & Kinetics-400    |   224 x 224 & 8x224x224     |  Clip | [Link]()  |
|    ImageNet & Hmdb-51    |   224 x 224 & 8x224x224     |  Clip  | [Link]()  |
|    ImageNet & Ucf-101    |   224 x 224 & 8x224x224     |  Clip  | [Link]()  |
|    ImageNet & Ssv2    |   224 x 224 & 8x224x224     |  Clip  | [Link]()  |

<hr>

## Training for Dynamic Cues for Videos

<hr>

## Training for Dynamic Cues for Multi-Views-ModelNet40

<hr>

## Evaluation

<hr>

## Attack Image2Videos

<hr>

## References

<hr>

## Setup the environment
### Build the codebase and environment

```
git clone https://github.com/Muzammal-Naseer/Adversarial-Transferability-using-Dynamic-Cues
cd Adversarial-Transferability-using-Dynamic-Cues
conda env create -n [name] --file environment.yml
conda activate [name]
python setup.py build develop
```
## Dataset Preparation
See details in md file

## Training
The folder [Image_Models_with_Temporal_Tokens](/Image_Models_with_Temporal_Tokens) contains all the code to train our models.

Change the arguments in [train_net.sh](/Image_Models_with_Temporal_Tokens/train_net.sh) file to train different variations.
