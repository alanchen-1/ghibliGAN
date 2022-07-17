import sys
sys.path.append('..')
from utils.model_utils import get_latest_num, save_outs
from collections import OrderedDict
import numpy as np
from predicates import within_bounds
import torch

def test_get_latest_num():
    assert get_latest_num('./test_dir/') == 9

def test_get_latest_num_empty():
    assert get_latest_num('./test_dir2') == -1

def test_save_outs():
    img_1 = torch.rand(1, 3, 256, 256)
    img_2 = torch.rand(1, 3, 256, 256) + 10
    img_3 = torch.rand(1, 3, 256, 256) + 100
    outs = OrderedDict({
            'img_1' : img_1,
            'img_2' : img_2,
            'img_3' : img_3
            })
    cat_img = save_outs(outs, '.', save=False)
    # check size
    assert np.shape(cat_img) == (1, 256, 256 * 3, 3)
    # check order
    assert within_bounds(cat_img[:, :, :256, :], 0, 1, hi_inclusive=False)
    assert within_bounds(cat_img[:, :, 256:512, :], 10, 11, hi_inclusive=False)
    assert within_bounds(cat_img[:, :, 512:, :], 100, 101, hi_inclusive=False)
