from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager, save_images
from scenedetect.detectors.content_detector import ContentDetector
import os

class Scenes():
    """
    Scene detector class of a video.
    """
    def __init__(self, video_path : str, video_name : str, threshold : float = 30.0):
        """
        Constructor for scene detector class.
            Parameters:
                video_path (str) : path to video
                video_name (str) : video name
                threshold (float) : threshold for scene detection
        """
        self.filename = video_name
        self.video_manager = VideoManager([video_path])
        self.scene_manager = SceneManager()
        self.scene_manager.add_detector(ContentDetector(threshold=threshold))

        self.video_manager.set_downscale_factor()

        self.video_manager.start()
        self.scene_manager.detect_scenes(frame_source=self.video_manager)

        self.scene_list = self.scene_manager.get_scene_list()

    def save_scenes(self, num_images : int = 1, output_dir : str = './images/'):
        """
        Saves <num_images> scenes of a video.
            Parameters: 
                num_images (int) : number of images
                output_dir (str) : output directory
        """
        output_dir = os.path.join(output_dir, self.filename)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        save_images(self.scene_list, self.video_manager, num_images=num_images,
        output_dir=output_dir)

