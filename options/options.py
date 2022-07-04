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
        parser.add_argument('--dataroot', type=str, required=True, help='[R] path to images w/ subfolders for respective domains')
        parser.add_argument('--model_name', type=str, default='model_results', help='name of model for directories')
        parser.add_argument('--use_gpu', action="store_true", help ='whether to use gpu or not (will use gpu index 0)')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='directory to save model checkpoints in')
        parser.add_argument('--config', type=str, required=True, help="path to config file")

        # load params
        parser.add_argument('--load_epoch', type=str, default='latest', help='Epoch to load. Only used if --continue_train is included or you are running test.py.')
        parser.add_argument('--verbose', action="store_true", help="use verbose or not")

        # data options
        self.initialized = True
        return parser
    
    def get_options(self):
        """
        Initializes parser and gets the options by parsing.
        Calls additional parser initialization if needed.
        """
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Options for ghibliGAN, [R] marks required")
            parser = self.initialize(parser)
        
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
        return self.opt

