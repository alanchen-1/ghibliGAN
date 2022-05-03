import argparse
import os

class Options():
    """
    Class defining Options for command line argument parsing.
    """
    def __init__(self):
        """
        'Initialize the parser'.
        """
        self.initialized = False
    
    def initialize(self, parser : argparse.ArgumentParser):
        """
        Initializes the parser with arguments.
            Parameters:
                parser (argparse.ArgumentParser) : uninitialized parser
            Returns:
                parser (argparse.ArgumentParser) : initialized parser
        """
        # general params
        parser.add_argument('--dataroot', type=str, help='path to images w/ subfolders for respective domains')
        parser.add_argument('--model_name', type=str, default='model_results', help='name of model for directories')
        parser.add_arguemnt('--use_gpu', type=bool, default=True, help ='whether to use gpu or not (will use gpu index 0)')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='directory to save model checkpoints in')
        # default values here taken from original CycleGAN paper
        parser.add_argument('--scale_size', type=int, defualt=286, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=256, help='crop scaled images to this size')

        # load params
        parser.add_argument('--load_epoch', type=str, default='latest', help='epoch to load')
        parser.add_argument('--verbose',  type=bool, default=True, help="use verbose or not")

        # model params
        parser.add_argument('--in_channels', type=int, default=3, help='number of input channels, default is 3 (rgb), 1 is grayscale')
        parser.add_argument('--out_channels', type=int, default=3, help='number of output channels, default is 3 (rgb), 1 is grayscale')
        parser.add_argument('--num_g_f', type=int, default=64, help='number of generator filters in the last convolutional layer')
        parser.add_argument('--num_d_f', type=int, default=64, help='number of discriminator filters in the first convolutional layer')
        parser.add_argument('--num_g_blocks', type=int, default=9, help='number of residual generator blocks')
        parser.add_argument('--num_d_layers', type=int, default=3, help='number of convolutional blocks in discriminator')
        parser.add_argument('--norm', type=str, default='instance', help='type of normalization to use')
        parser.add_argument('--init_type', type=str, default='normal', help='type of initialization to use')
        parser.add_argument('--init_scale', type=float, default=0.02, help='initialization scale to use')
        parser.add_argument('--use_dropout', action='store_true', help='include if you want to use dropout layers')
    
        # data options

        self.initialized = True
        return parser
    
    def get_options(self):
        """
        Initializes parser and gets the options by parsing.
        Calls additional parser initialization if needed.
        """
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        
        opt, _ = parser.parse_known_args()

        self.parser = parser
        return parser.parse_args()
    
    def export_options(self, opt):
        """
        Exports parser options to a file in <checkpoints_dir>.
            Parameters:
                opt : options to export
        """
        string = 'using options: \n'
        for option, value in sorted(vars(opt).items()):
            string += f'{option} : {value}\n'
        string += '\n'
        print(string)
        export_dir = os.path.join(opt.checkpoints_dir, opt.model_name)
        if not os.path.isdir(export_dir):
            os.makedirs(export_dir)
        filename = os.path.join(export_dir, f'{opt.phase}_options.txt')
        with open(filename, 'wt') as open_file:
           open_file.write(string) 
    
    def parse(self):
        """
        Parse, update to_train, and export options.
        """
        opt = self.get_options()
        opt.to_train = self.to_train
        self.opt = opt
        self.export_options(opt)
        return self.opt

