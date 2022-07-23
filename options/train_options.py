from .options import Options
import argparse


class CycleTrainOptions(Options):
    """
    Class defining extra options for training the CycleGAN model.
    Inherits from Options defined in options.py.
    """
    def initialize(self, parser: argparse.ArgumentParser):
        """
        Initialize method with arguments.
        Adds various arguments to the specified parser.
        Also updates to_train to be True.
            Parameters:
                parser (argparse.ArgumentParser) : parser to update
            Returns:
                parser (argparse.ArgumentParser) : updated parser
        """
        parser = Options.initialize(self, parser)
        parser.add_argument('--continue_train', action='store_true',
                            help='continue training: load the latest model')
        self.to_train = True
        return parser
