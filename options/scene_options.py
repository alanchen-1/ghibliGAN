import argparse

class SceneOptions():
    def __init__(self):
        self.initialized = False
    
    def initialize(self, parser : argparse.ArgumentParser):
        parser.add_argument('--video_path', type=str, required=True, help='path to video file')
        parser.add_argument('--output_dir', type=str, default = None, help='output directory. will default to creating a custom one based on video file name.')
        parser.add_argument('--threshold', type=float, default=30.0, help='video cutting threshold (read more in scenedetect docs)')
        parser.add_argument('--num_images', type=int, default=2, help='number of images per scene')
        self.initialized = True

    def get_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        self.parser = parser
        return parser.parse_args()

    def export_options(self, opt):
        # export options to file in output_dir
        return None

    def parse(self):
        # get options, then export, then 
        return None
