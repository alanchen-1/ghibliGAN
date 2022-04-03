from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager, save_images
from scenedetect.detectors.content_detector import ContentDetector
import ntpath
import os

def get_video_alias(video_path : str) -> str:
    return ntpath.basename(video_path).split('.')[0]

class Scenes():
    def __init__(self, video_path : str, threshold : float = 30.0):
        self.filename = get_video_alias(video_path)
        self.video_manager = VideoManager([video_path])
        self.scene_manager = SceneManager()
        self.scene_manager.add_detector(ContentDetector(threshold=threshold))

        self.video_manager.set_downscale_factor()

        self.video_manager.start()
        self.scene_manager.detect_scenes(frame_source=self.video_manager)

        self.scene_list = self.scene_manager.get_scene_list()

    def save_scenes(self, num_images : int = 1, output_dir : str = './images/'):
        output_dir = os.path.join(output_dir, self.filename)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        save_images(self.scene_list, self.video_manager, num_images=num_images,
        output_dir=output_dir)

