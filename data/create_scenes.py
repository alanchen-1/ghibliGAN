import sys
sys.path.append('..')
from scenes import Scenes
from options.scene_options import SceneOptions

# to run, use python create_scenes.py --video_path <video_path> --video_name <video_name> 
# and of course, include any other extra arguments

parser = SceneOptions()
opt = parser.parse()

scenes_manager = Scenes(opt.video_path, opt.video_name, opt.threshold)
scenes_manager.save_scenes(opt.num_images, opt.output_dir)

