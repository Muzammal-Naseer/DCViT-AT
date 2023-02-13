# Boosting Adversarial Transferability using Dynamic Cues (ICLR'23)

[Muzammal Naseer](https://scholar.google.ch/citations?user=tM9xKA8AAAAJ&hl=en),
[Ahmad Mahmood](https://www.linkedin.com/in/ahmad-mahmood-81339a212/),
[Salman Khan](https://scholar.google.com/citations?user=M59O9lkAAAAJ&hl=en), &
[Fahad Khan](https://scholar.google.ch/citations?user=zvaeYnUAAAAJ&hl=en&oi=ao),

[Paper](https://openreview.net/forum?id=SZynfVLGd5) ([arxiv](soon)), [Reviews & Response](https://openreview.net/forum?id=SZynfVLGd5), [Video Presentation](soon), [Poster](soon)
#

> **Abstract:** 
*The transferability of adversarial perturbations between image models has been extensively studied. In this case, an attack is generated from a known surrogate \eg, the ImageNet trained model, and transferred to change the decision of an unknown (black-box) model trained on an image dataset. However, attacks generated from image models do not capture the dynamic nature of a moving object or a changing scene due to a lack of temporal cues within image models. This leads to reduced transferability of adversarial attacks from representation-enriched \emph{image} models such as Supervised Vision Transformers (ViTs), Self-supervised ViTs (\eg, DINO), and Vision-language models (\eg, CLIP) to black-box \emph{video} models. In this work, we induce dynamic cues within the image models without sacrificing their original performance on images. To this end, we optimize \emph{temporal prompts} through frozen image models to capture motion dynamics. Our temporal prompts are the result of a learnable transformation that allows optimizing for temporal gradients during an adversarial attack to fool the motion dynamics. Specifically, we introduce spatial (image) and temporal (video) cues within the same source model through task-specific prompts. Attacking such prompts maximizes the adversarial transferability from image-to-video and image-to-image models using the attacks designed for image models. As an example, an iterative attack launched from image model Deit-B with temporal prompts reduces generalization (top1 \% accuracy) of a video model by 35\% on Kinetics-400. Our approach also improves adversarial transferability to image models by 9\% on ImageNet w.r.t the current state-of-the-art approach. Our attack results indicate that the attacker does not need specialized architectures, \eg, divided space-time attention, 3D convolutions, or multi-view convolution networks for different data modalities. Image models are effective surrogates to optimize an adversarial attack to fool black-box models in a changing environment over time. Code is available at \url{https://bit.ly/3Xd9gRQ}.* 
>

![demo](.github/demo.png)

#
<hr>

## Main Message & Highlights


1. The attacker does not need specialized architectures, \eg, divided space-time attention, 3D convolutions, or multi-view convolution networks for different data modalities. Image models are effective surrogates to optimize an adversarial attack to fool black-box models in a changing environment over time

2. We introduce dynamic cues within frozen image models without losing the original image representation (e.g. generalization on ImageNet). Both image and video representations enhance adversarial transferability from our adapted image models.

3. Our approach simply augments the existing adversarial attacks developed for image models to fool video or multi-view models.

4. We analyze three types of training schemes (supervised, self-supervised, and text-supervised) and highlight new insights into the adversarial space of vision-language models.

5. We further anlayze the textual bias within vision-language model, CLIP, for the low adversarial transferability to vision models.

6. We observe highly transferable adversaial space of self-supervised vision transformer models such as DINO. 

#
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
