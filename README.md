# ghibliGAN

CycleGAN implementation trained to "Ghibli"-ize real landscapes

## Requirements
I've only tested this thoroughly on a WSL system running Ubuntu 20.04. I loosened the package requirements in the `environment.yml` file, so it should work on every other platform.

Requirements:
- `conda`
- CPU/NVIDIA GPU (you probably want a GPU if you want to try the training pipeline)

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
to see all the flags. Check [Data](https://github.com/alanchen-1/ghibliGAN#data) for information on data formatting.

Training and testing expect a `.yaml` file with the configuration options for the model. See the included `config/main-config.yaml` for an example. If you would like to change some numbers, I would recommend just copy pasting and editing the parameters from there.

## Data
The data is expected to be organized as a `--dataroot` folder with corresponding subfolders laid out as below:
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
To help with splitting into train and test, `data/split_data.py` is provided, which runs on a single image folder and splits it into train and test with the desired naming scheme. There are some other random utilities scattered around.

The `real2ghilbli` dataset (~900 images) was taken from 9 Ghibli movies. These movies were chosen relatively arbitrarily, but it was mainly what I could get my hands on. The scene detection is done through `scenedetect`, a super nice PyPi package.

The `real2shinkai` dataset (~300 images) was taken from 3 Makoto Shinkai films - Your Name, Weathering with You, and Garden of Words. This was to test if there was any big performance drop when using a much smaller dataset.

The real data (`trainX` and `testX`) are ~6000 landscapes downloaded from Flickr stolen from the original CycleGAN repository.

## Misc
All code adheres to PEP8 style guidelines (as checked by `flake8`). Most of the non-model-architecture code is unittested in `test/` - a lot of AI just architecture constructors so its harder to test those. 

CycleGAN is a pretty outdated architecture, as there exist models these days that can learn a style from a single image (StyleGAN2 is really powerful), but I don't have the GPU power to train models of that caliber right now - my poor laptop :sob:. Additionally, most of the values for the hyperparameters I used were just taken from Section 4 and the Appendix of the original paper.

This is probably my biggest solo Python project to date (~2500 lines). I made a significant effort to use good software principles like OOP, docs, testing (even unit testing training process for model, not usually done), CI/CD, etc. Was it worth it though...
