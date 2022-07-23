import yaml
import torch
import sys
sys.path.append('..')
from models.cyclegan import CycleGAN


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
    config['train']['warmup_epochs'] = 0
    config['train']['decay_epochs'] = 1

    # create model + dataset
    model = CycleGAN(opt, config)
    model.general_setup()

    params = [
        named_param for named_param in model.named_parameters()
        if named_param[1].requires_grad
    ]
    initial_params = [(name, p.clone()) for (name, p) in params]
    # train iteration
    dummy = ((0.5)**2) * torch.randn(1, 3, 256, 256) + 0.5
    data = {
        'X': dummy,
        'Y': dummy,
        'X_paths': '',
        'Y_paths': ''
    }
    schedulers = model.schedulers
    init_lr_vals = [scheduler.get_last_lr() for scheduler in schedulers]
    model.setup_input(data)
    model.optimize()
    model.update_schedulers()
    lr_vals = [scheduler.get_last_lr() for scheduler in schedulers]

    for (_, p0), (name, p1) in zip(initial_params, params):
        # assert that params have changed
        assert not torch.equal(p0.to('cpu'), p1.to('cpu'))

    for prev, new in zip(init_lr_vals, lr_vals):
        assert not prev == new
