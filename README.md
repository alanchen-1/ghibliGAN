# ghibliGAN

CycleGAN implementation trained to "Ghibli"-ize real images

## Requirements
I've only tested this thoroughly on a WSL system running Ubuntu 20.04. I loosened the package requirements in the `environment.yml` file, so it should work on every other platform.

Other requirements:
- conda
- CPU/GPU

## Installation
To get the implementation, you can simply run the following commands:
```bash
git clone https://github.com/alanchen-1/ghibliGAN.git
cd ghibliGAN
conda env create -f environment.yml
conda activate ghibli
```

Other forms of installation (`pip`, Docker) are on the longlist of things to add.

## Quick Inference
<UNDER CONSTRUCTION>
Pretrained checkpoints and dataset are coming soon.

## Training
If you would like to retrain the model either on the dataset I used or a different dataset, `train.py` is provided for this. Run
```bash
python train.py -h
```
to see all the flags. 

It expects a `--dataroot` folder with corresponding subfolders laid out like this:
```
.
├── ...
├── data                    # Data root
│   ├── trainX              # Real images for training
│   ├── trainY              # Fake images/styled images for training
│   └── testX               # Real images for testing
│   └── testY               # Fake images for testing
└── ...
```
To help with splitting, `data/split_data.py` is provided, which runs on a single image folder and splits it into train and test with the desired naming scheme.

It also expects a `.yaml` file with the configuration options for the model. See the included `config/main-config.yaml` for an example - if you would like to change some numbers, I would recommend just copy pasting and editing the parameters from there.

## Data Notes
The data was taken from Howl's Moving Castle, Spirited Away, and Totoro. I chose these movies relatively arbitrarily, but it was mainly what I could get my hands on. The scene detection is done through `scenedetect`, a super nice PyPi package.

## Misc
This is probably my biggest Python project to date (~2k lines). I made a significant effort to use good software principles like OOP, docs, testing (even unit testing training process for model, not usually done), CI/CD, etc. 

CycleGAN is a pretty outdated architecture, as there exist models these days that can learn a style from a single image (StyleGAN2 is really powerful), but I don't have the GPU power to train models of that caliber right now (my poor laptop getting abused again).

