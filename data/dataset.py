from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import random

IMAGE_EXTENSIONS = ['.png', '.jpg']
def image_walk(root_dir : str):
    """
    Gets all images recursively in the root directory using os.walk().
        Parameters: 
            root_dir (str) : path to explore
        Returns:
            images (list[str]) : list of image paths
    """
    images = []
    for root, _, files in sorted(os.walk(root_dir)):
        for filename in files:
            if (os.path.splitext(filename)[1] in IMAGE_EXTENSIONS):
                images.append(os.path.join(root, filename))
    return images

class CycleDataset(Dataset):
    def __init__(self, to_train : bool, dataroot : str, scale_size : int, in_channels : int, out_channels : int, crop_size : int, in_order : bool, **kwargs):
        """
        Constructor for CycleDataset.
            Parameters:
                to_train (bool) : is this dataset for training
                dataroot (str) : root folder, should house trainX, trainY, testX, testY folders
                scale_size (int) : scaling size 
                in_channels (int) : number of channels in the input domain X
                out_channels (int) : number of channels in the output domain Y
                crop_size (int) : size to crop images to
                in_order (bool) : should the images be loaded in lexicographical order
                **kwargs : used to filter out other unneeded parameters so that passing around config file is easy
        """
        self.dataroot = dataroot
        assert os.path.isdir(self.dataroot), f"{self.dataroot} is not a recognized directory"

        mode = 'train' if to_train else 'test'
        self.Xdir = os.path.join(self.dataroot, f"{mode}X")
        self.Ydir = os.path.join(self.dataroot, f"{mode}Y")

        assert os.path.isdir(self.Xdir), f"{mode}X subdirectory not found in {self.dataroot}"
        assert os.path.isdir(self.Ydir), f"{mode}Y subdirectory not found in {self.dataroot}"

        self.create_dataset()
        self.Xsize = len(self.X_images)
        self.Ysize = len(self.Y_images)
        self.in_order = in_order

        self.out_size = [scale_size, scale_size]
        self.crop_size = crop_size
        self.transform_X = self.get_transforms(in_channels == 1)
        self.transform_Y = self.get_transforms(out_channels == 1)
    
    def get_transforms(self, grayscale=False):
        """
        Gets transforms based on the crop size and the channels of the domains.
            Parameters:
                grayscale (bool) : is the domain this is for grayscale
            Returns:
                transforms (list(transforms)) : list of transforms to apply
        """
        core_transforms = [
            transforms.Resize(self.out_size, transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(self.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        if grayscale:
            transform = [transforms.Grayscale(1)] + core_transforms +  [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform = core_transforms + [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        return transforms.Compose(transform)
    
    def __getitem__(self, index : int):
        """
        Gets image item at index <index>.
        Required by inheritance of Dataset.
            Parameters:
                index (int) : index
            Returns:
                (dict) : dictionary with keys X, Y, X_paths, Y_paths
        """
        X_img_path = self.X_images[index % self.Xsize]
        if self.in_order:
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
        """
        Returns the bigger of the two dataset.
        Required by inheritance of Dataset.
        """
        return max(self.Xsize, self.Ysize)

    def both_len(self):
        """
        Gets the size of both directories.
        """
        return self.Xsize, self.Ysize
    
    def create_dataset(self):
        """
        Initializes the X_images and Y_images.
        """
        self.X_images = sorted(image_walk(self.Xdir))
        self.Y_images = sorted(image_walk(self.Ydir))

