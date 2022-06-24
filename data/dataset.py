
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import random

IMAGE_EXTENSIONS = ['.png', '.jpg']
def image_walk(root_dir):
    images = []
    for root, _, files in sorted(os.walk(root_dir)):
        for filename in files:
            if (os.path.splitext(filename)[1] in IMAGE_EXTENSIONS):
                images.append(os.path.join(root, filename))
    return images


class CycleDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.dataroot = opt.dataroot
        assert os.path.isdir(self.dataroot), f"{self.dataroot} is not a recognized directory"

        mode = 'train' if opt.to_train else 'test'
        self.Xdir = os.path.join(self.dataroot, f"{mode}X")
        self.Ydir = os.path.join(self.dataroot, f"{mode}Y")

        assert os.path.isdir(self.Xdir), f"{mode}X subdirectory not found in {self.dataroot}"
        assert os.path.isdir(self.Ydir), f"{mode}Y subdirectory not found in {self.dataroot}"

        self.create_dataset()
        self.Xsize = len(self.X_images)
        self.Ysize = len(self.Y_images)

        self.out_size = [opt.scale_size, opt.scale_size]
        self.transform_X = self.get_transforms(opt.in_channels == 1)
        self.transform_Y = self.get_transforms(opt.out_channels == 1)
    
    def get_transforms(self, grayscale=False):
        core_transforms = [
            transforms.Resize(self.out_size, Image.BICUBIC),
            transforms.RandomCrop(self.opt.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        if grayscale:
            transform = [transforms.Grayscale(1)] + core_transforms +  [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform = core_transforms + [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        return transform
    
    def __getitem__(self, index : int):
        X_img_path = self.X_images[index % self.Xsize]
        if self.opt.in_order:
            Y_img_path = self.Y_images[index % self.Ysize]
        else:
            Y_img_path = self.Y_images[random.randint(0, self.Ysize)]
        
        X_img = Image.open(X_img_path).convert('RGB')
        Y_img = Image.open(Y_img_path).convert('RGB')

        return {
            'X' : self.transform_X(X_img), 
            'Y' : self.transform_Y(Y_img),
            'X_paths' : X_img_path,
            'Y_paths' : Y_img_path
        }

    def __len__(self):
        return max(self.Xsize, self.Ysize)

    def both_len(self):
        return self.Xsize, self.Ysize
    
    def create_dataset(self):
        self.X_images = sorted(image_walk(self.Xdir))
        self.Y_images = sorted(image_walk(self.Ydir))

