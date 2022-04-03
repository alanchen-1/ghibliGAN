import argparse
import os

class SceneOptions():
    def __init__(self):
        self.initialized = False
    
    def initialize(self, parser : argparse.ArgumentParser):
        parser.add_argument('--video_path', type=str, required=True, help='path to video file')
        parser.add_argument('--video_name', type=str, required=True, help='video name (name of subdirectory)')
        parser.add_argument('--output_dir', type=str, default='./images/', help='output directory. will create video_name subdirectory in here.')
        parser.add_argument('--threshold', type=float, default=30.0, help='video cutting threshold (read more in scenedetect docs)')
        parser.add_argument('--num_images', type=int, default=1, help='number of images per scene')
        self.initialized = True
        return parser

    def get_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        self.parser = parser
        return parser.parse_args()

    def export_options(self, opt):
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
        # get options, then export, then 
        opt = self.get_options()
        self.opt = opt
        self.export_options(opt)
        return opt
