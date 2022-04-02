import sys
sys.path.append('..')
from data.create_data import get_video_alias
import unittest

class TestCreateData(unittest.TestCase):
    def test_get_video_alias(self):
        self.assertEqual(get_video_alias('../data/videos/ponyo.mp4'), 'ponyo')
        self.assertEqual(get_video_alias('deeznuts/data/'), '')

if __name__ == '__main__':
    unittest.main()