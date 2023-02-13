# Boosting Adversarial Transferability using Dynamic Cues (ICLR'23)

[Muzammal Naseer](https://scholar.google.ch/citations?user=tM9xKA8AAAAJ&hl=en),
[Ahmad Mahmood](https://www.linkedin.com/in/ahmad-mahmood-81339a212/),
[Salman Khan](https://scholar.google.com/citations?user=M59O9lkAAAAJ&hl=en), &
[Fahad Khan](https://scholar.google.ch/citations?user=zvaeYnUAAAAJ&hl=en&oi=ao),

[Paper](https://openreview.net/forum?id=SZynfVLGd5) ([arxiv](soon)), [Reviews & Response](https://openreview.net/forum?id=SZynfVLGd5), [Video Presentation](soon), [Poster](soon)






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
