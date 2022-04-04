import argparse
import os

class SceneOptions():
    """
    Class for scene options.
    Used for create_scene.py in the data package.
    """
    def __init__(self):
        """
        Constructor for SceneOptions.
        """
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser = self.add_arguments(parser)
    
    def add_arguments(self, parser : argparse.ArgumentParser):
        """
        Adds arguments to the parser.
            Parameters:
                parser (argparse.ArgumentParser) : argument parser to initialize
            Returns:
                parser (argparse.ArgumentParser) : initialized argument parser
        """
        parser.add_argument('--video_path', type=str, required=True, help='path to video file')
        parser.add_argument('--video_name', type=str, required=True, help='video name (name of subdirectory)')
        parser.add_argument('--output_dir', type=str, default='./images/', help='output directory. will create video_name subdirectory in here.')
        parser.add_argument('--threshold', type=float, default=30.0, help='video cutting threshold (read more in scenedetect docs)')
        parser.add_argument('--num_images', type=int, default=1, help='number of images per scene')
        return parser

    def export_options(self, opt):
        """
        Exports the options to a file in <output_dir>.
            Parameters:
                opt : parsed options
        """
        string = 'using options: \n'
        for option, value in sorted(vars(opt).items()):
            string += f'{option} : {value}\n'
        string += '\n'
        print(string)
        
        # export options to file in output_dir
        output_dir = os.path.join(opt.output_dir, opt.video_name)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        filename = os.path.join(output_dir, f'{opt.video_name}_options.txt')
        with open(filename, 'wt') as open_file:
           open_file.write(string) 

    def parse(self):
        """
        Parses, exports, and returns the options.
        """
        # get options
        opt = self.parser.parse_args()
        self.opt = opt
        # export
        self.export_options(opt)
        # return parsed options
        return opt
