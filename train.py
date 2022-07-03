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
from utils.model_utils import print_losses

if __name__ == '__main__':
    opt = CycleTrainOptions().parse()

    with open(opt.config, 'r') as file:
        config = yaml.safe_load(file)
    # create data
    model = CycleGAN(opt)
    model.general_setup(opt)
    dataset = CycleDataset((opt.phase == 'train'), dataroot=opt.dataroot, **config['dataset'])
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
    total_epochs = opt.warmup_epochs + opt.decay_epochs
    print("Total epochs: ", total_epochs)
    for epoch in range(opt.start_epoch, total_epochs + 1):
        for i, data in enumerate(dataloader):
            model.setup_input(data)
            model.optimize()
            model.update_schedulers()
            losses = model.get_losses() # ordered dict
            print_losses(losses, epoch, total_epochs, i + 1, max_size)
        
        if (epoch % opt.save_epoch_freq == 0) or (epoch == total_epochs):
            # save version with latest and also with epoch num
            print(f"Saving models at end of epoch {epoch}")
            model.save_networks()
            model.save_networks(str(epoch))
