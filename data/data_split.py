# script to format and split a folder of images
import argparse
import os
import random
import math
from PIL import Image

IMAGE_EXTENSIONS = ['.jpg', '.png']


def split_images(root_dir, result_dir, ratio, label):
    img_label = 'X' if label == 'real' else 'Y'
    train_dir = os.path.join(result_dir, f"train{img_label}")
    test_dir = os.path.join(result_dir, f"test{img_label}")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    image_list = sorted([
        os.path.join(root_dir, file)
        for file in os.listdir(root_dir)
        if os.path.splitext(file)[1] in IMAGE_EXTENSIONS
    ])
    random.shuffle(image_list)
    end_index = math.ceil(ratio * len(image_list))

    for i, img_file in enumerate(image_list):
        base_name = os.path.basename(img_file)
        if i < end_index:
            # save in train
            Image.open(img_file).save(os.path.join(train_dir, base_name))
        else:
            # save in test
            Image.open(img_file).save(os.path.join(test_dir, base_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help="root directory")
    parser.add_argument('--ratio', type=float, default=0.75,
                        help="proportion to put into train")
    parser.add_argument('--result', type=str, required=True,
                        help="where to store results")
    parser.add_argument('--label', type=str, required=True,
                        choices=['real', 'fake'],
                        help="is this dataset of real or fake images")
    args = parser.parse_args()
    split_images(args.root, args.result, args.ratio, args.label)
