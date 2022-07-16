import sys
sys.path.append('..')
import yaml
import torch
from options.train_options import CycleTrainOptions
from models.cyclegan import CycleGAN
from utils.model_utils import print_losses, get_latest_num

class DummyClass():
    def __init__(self):
        pass

def test_train():
    """
    Tests if the model parameters actually change by optimizing.
    """
    opt = DummyClass()
    opt.to_train = True
    opt.load_epoch = 'latest'
    opt.verbose = False
    opt.phase = 'train'
    opt.continue_train = False
    opt.checkpoints_dir = '.'
    opt.model_name = ''
    opt.use_gpu = False

    # config params
    with open('../config/main_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    config['train']['start_epoch'] = 1

    # create model + dataset
    model = CycleGAN(opt, config)
    model.general_setup()

    params = [named_param for named_param in model.named_parameters() if named_param[1].requires_grad]
    initial_params = [(name, p.clone()) for (name, p) in params]
    # train iteration
    dummy = ((0.5)**2) * torch.randn(1, 3, 256, 256) + 0.5
    data = {
            'X' : dummy,
            'Y' : dummy,
            'X_paths' : '',
            'Y_paths' : ''
            }
    model.setup_input(data)
    model.optimize()

    for (_, p0), (name, p1) in zip(initial_params, params):
        # assert that params have changed
        assert not torch.equal(p0.to('cpu'), p1.to('cpu'))
