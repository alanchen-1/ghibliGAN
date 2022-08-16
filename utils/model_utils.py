# defines useful utilities for other files to use
from collections import OrderedDict
import torch.nn as nn
import os
from PIL import Image
import numpy as np


def print_network(net: nn.Module, verbose: bool = True):
    """
    Prints network if verbose and the number of parameters.
        Parameters:
            net (nn.Module) : network to print
            verbose (bool) : print network architecture or not
    """
    if verbose:
        print(net)

    num_params = 0
    for parameter in net.parameters():
        num_params += parameter.numel()

    num_params /= 1e6
    print(f'Total params:  {num_params:.3f} million')


def print_losses(
    losses: OrderedDict,
    epoch: int,
    total_epochs: int,
    iters: int,
    total_iters: int,
    subset: list = None
):
    """
    Prints losses at end of every dataloader iteration.
        Parameters:
            losses (OrderedDict) : ordered dictionary of losses
            epoch (int) : current epoch
            total_epochs (int) : total number of epochs
            iters (int) : number of iterations
            total_iters (int) : total number of iterations
            subset (list) : subset of keys, if desired. default is None
    """
    if subset is None:
        subset = losses.keys()

    to_print = f"Losses [{epoch} / {total_epochs}] [{iters} / {total_iters}]:"
    for key in subset:
        to_print += f" [{key} : {losses[key]:.7f}] "

    print(to_print)


def get_latest_num(checkpoints_dir: str):
    """
    Calculates the epoch corresponding to the "latest" saved checkpoint.
    Assumes there exists such a checkpoint, its saved as a .pth file,
    and that the files have been named such the epoch they were saved
    at is a prefix before the first underscore
    (ex. "{epoch}_{rest of name}.pth").
        Parameters:
            checkpoints_dir (str) : directory of checkpoints
        Returns:
            mx (int) : epoch corresponding to the "latest" checkpoint
    """
    files = list(filter(
        lambda name: os.path.splitext(name)[1] == '.pth',
        os.listdir(checkpoints_dir)))
    mx = -1
    for filename in files:
        prefix = filename.split('_')[0]
        if (prefix != 'latest'):
            try:
                if int(prefix) > mx:
                    mx = int(prefix)
            except ValueError:
                pass
    return mx


def save_outs(
    outs: OrderedDict,
    out_dir: str,
    save_separate: bool = False,
    extension: str = 'jpg',
    save: bool = True
):
    """
    Concatenates images in order in outs and saves them in a combined graphic.
    Assumes images have not been touched at all since being returned by the
    model. For instance, assumes NCHW format and that the values are
    torch.Tensor.
        Parameters:
            outs (OrderedDict) : collected ordered image outputs
            out_dir (str) : where to save the images
            save_separate (bool) : whether to save all images separately
            extension (str) : extension to save images with
            save (bool) : turn saving on or off. useful for testing
        Returns:
            cat_img (np.array) : concatenated image, mainly used for testing
    """
    os.makedirs(out_dir, exist_ok=True)
    cat_img = None
    transformed_imgs = []
    for (_, v) in outs.items():
        v_transform = np.transpose(
            v.cpu().float().detach().numpy(),
            (0, 2, 3, 1))

        if v_transform.shape[-1] == 1:  # make grayscale into color
            v_transform = np.tile(v_transform, (1, 1, 1, 3))

        # scale properly
        v_transform = ((v_transform + 1) / 2.0 * 255.0).astype(np.uint8)

        cat_img = v_transform if cat_img is None \
            else np.concatenate([cat_img, v_transform], 2)
        if save_separate:
            transformed_imgs.append(v_transform)

    # save the combined image
    if save:
        Image.fromarray(cat_img[0]).save(
            os.path.join(out_dir, f"combined.{extension}"))

        for name, img in zip(outs.keys(), transformed_imgs):
            Image.fromarray(img[0]).save(
                os.path.join(out_dir, f"{name}.{extension}"))

    return cat_img  # useful for testing, but not normally used
