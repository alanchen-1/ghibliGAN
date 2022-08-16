from .options import Options
import argparse


class CycleTestOptions(Options):
    """
    Class defining extra options for testing the CycleGAN model.
    Inherits from Options defined in options.py.
    """
    def initialize(self, parser: argparse.ArgumentParser):
        """
        Initialize method with arguments.
        Adds various arguments to the specified parser.
        Also updates to_train to be False.
            Parameters:
                parser (argparse.ArgumentParser) : parser to update
            Returns:
                parser (argparse.ArgumentParser) : updated parser
        """
        parser = Options.initialize(self, parser)
        parser.add_argument('--num_tests', type=int, default=float('inf'),
                            help="max number of images to run on")
        parser.add_argument('--result_dir', default='./results',
                            help='result directory')
        parser.add_argument('--save_separate', action='store_true',
                            help="save all separate graphics")
        self.to_train = False
        return parser
