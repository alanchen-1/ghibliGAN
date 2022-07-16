import sys
sys.path.append('..')
from utils.model_utils import get_latest_num

def test_get_latest_num():
    assert get_latest_num('./test_dir/') == 9

def test_get_latest_num_empty():
    assert get_latest_num('./test_dir2') == -1


