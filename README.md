# ADVERSARIAL TRANSFERABILITY USING DYNAMIC CUES
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
