"""
Rough outline:
1) create dataset
2) create model, setup
"""

from options.train_options import CycleTrainOptions
from models.cyclegan import CycleGAN
from data.dataset import CycleDataset
from utils.model_utils import print_losses

if __name__ == '__main__':
    opt = CycleTrainOptions().parse()
    # create data
    model = CycleGAN(opt)
    model.general_setup(opt)
    dataset = CycleDataset(opt)
    max_size = len(dataset)
    X_size, Y_size = dataset.both_len()
    print(f'Number of X images: {X_size}, Number of Y images: {Y_size}')

    
    print ("Starting training loop...")
    # main loop
    total_epochs = opt.warmup_epochs + opt.decay_epochs
    print("total epochs: ", total_epochs)
    for epoch in range(opt.start_epoch, total_epochs + 1):
        for i, data in enumerate(dataset):
            model.setup_input(data)
            model.optimize()
            model.update_schedulers()
        if (epoch % opt.save_epoch_freq == 0) or (epoch == total_epochs):
            # save version with latest and also with epoch num
            model.save_networks()
            model.save_networks(str(epoch))

        losses = model.get_losses() # ordered dict
        print_losses(losses, epoch, total_epochs, i, max_size)