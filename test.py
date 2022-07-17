import yaml
import torch
from options.test_options import CycleTestOptions
from models.cyclegan import CycleGAN
from data.dataset import CycleDataset
from utils.model_utils import print_losses, get_latest_num, save_outs

if __name__ == '__main__':
    # parse options
    parser = CycleTestOptions()
    opt = parser.parse()
    opt.phase = 'test'
    parser.export_options(opt)

    result_dir = os.path.join(opt.result_dir, f"{opt.model_name}_{opt.phase}"),
    os.makedirs(result_dir, exist_ok=True)
    print(f"Saving images in {result_dir}")

    # config params
    with open(opt.config, 'r') as file:
        config = yaml.safe_load(file)
    config['dataset']['scale_size'] = config['dataset']['crop_size']

    # create model + dataset
    model = CycleGAN(opt, config)
    model.general_setup()

    dataset = CycleDataset(opt.to_train, dataroot=opt.dataroot, **config['dataset'])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = config['dataset']['num_workers']
    )
    for i, data in enumerate(dataloader):
        if i + 1 >= opt.num_tests:
            break
        model.setup_input(data)
        out = model.test() # default is real image and style transferred image

        # save in results dir

        file_name = os.path.splitext(os.path.basename(model.image_paths[0]))[0]
        save_outs(out, os.path.join(result_dir, file_name), opt.save_separate)
