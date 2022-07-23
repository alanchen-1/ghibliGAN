import sys
import os
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..'))
from scenes import Scenes
from options.scene_options import SceneOptions

# to run, use
#   `python create_scenes.py --video_path <video_path> \
#       --video_name <video_name>
# and of course, include any other extra arguments

parser = SceneOptions()
opt = parser.parse()

scenes_manager = Scenes(opt.video_path, opt.video_name, opt.threshold)
scenes_manager.save_scenes(opt.num_images, opt.output_dir)
