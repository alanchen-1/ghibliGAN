import argparse
import cv2
import os
from tqdm import tqdm


def detect(root_dir: str, cascade: cv2.CascadeClassifier) -> None:
    """
    Removes files with detected faces.
        Parameters:
            root_dir (str) : directory to look in
            cascade (cv2.CascadeClassifier) : loaded cascade file
        Returns:
            (bool) : if faces were outputted
    """

    removed_count = 0
    for filename in tqdm(os.listdir(root_dir)):
        try:
            filepath = os.path.join(root_dir, filename)
            image = cv2.imread(filepath, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(24, 24)
            )
            if len(faces) > 0:
                # remove image
                os.remove(filepath)
                removed_count += 1
        except Exception:
            pass
    print(f"Removed {removed_count} images")


# apply cascade as demonstrated in example
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', required=True,
                        help="image dir to filter")
    parser.add_argument('--cascade_file', required=True,
                        help="cascade file to use")
    args = parser.parse_args()
    if not os.path.isfile(args.cascade_file):
        raise RuntimeError("%s: not found " % args.cascade_file)

    cascade = cv2.CascadeClassifier(args.cascade_file)

    detect(args.image_dir, cascade)
