"""
Rough outline:
1) create dataset
2) create model, setup
"""
import yaml
import torch
from options.train_options import CycleTrainOptions
from models.cyclegan import CycleGAN
from data.dataset import CycleDataset
from utils.model_utils import print_losses, get_latest_num

if __name__ == '__main__':
    # parse options
    parser = CycleTrainOptions()
    opt = parser.parse()
    opt.phase = 'train'
    parser.export_options(opt)

    # config params
    with open(opt.config, 'r') as file:
        config = yaml.safe_load(file)
    warmup_epochs = config['train']['warmup_epochs']
    decay_epochs = config['train']['decay_epochs']
    save_epoch_freq = config['train']['save_epoch_freq']
    total_epochs = warmup_epochs + decay_epochs
    if not(opt.continue_train):
        start_epoch = 1
    else:
        # do it based on load epoch
        if (opt.load_epoch == 'latest'):
            start_epoch = get_latest_num(opt.checkpoints_dir)
            pass
        else:
            start_epoch = int(opt.load_epoch)
    config['train']['start_epoch'] = start_epoch

    # create model + dataset
    model = CycleGAN(opt, config)
    model.general_setup()
    dataset = CycleDataset(opt.to_train, dataroot=opt.dataroot, **config['dataset'])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = config['dataset']['batch_size'],
        shuffle = not(config['dataset']['in_order']),
        num_workers = config['dataset']['num_workers']
    )
    max_size = len(dataloader)
    X_size, Y_size = dataset.both_len()
    print(f'Number of X images: {X_size}, Number of Y images: {Y_size}')
    
    print ("Starting training loop...")
    print("Losses printed as [epoch / total epochs] [batch / total batches]")

    # main loop
    print("Total epochs: ", total_epochs)
    for epoch in range(start_epoch, total_epochs + 1):
        for i, data in enumerate(dataloader):
            print(data['X'].shape)
            model.setup_input(data)
            model.optimize()
            model.update_schedulers()
            losses = model.get_losses() # ordered dict
            print_losses(losses, epoch, total_epochs, i + 1, max_size)
        
        if (epoch % save_epoch_freq == 0) or (epoch == total_epochs):
            # save version with latest and also with epoch num
            print(f"Saving models at end of epoch {epoch}")
            model.save_networks()
            model.save_networks(str(epoch))
