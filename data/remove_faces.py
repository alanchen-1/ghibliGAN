import argparse
import cv2
import os
from tqdm import tqdm

def detect(root_dir, cascade):
    """
    Detects faces in the specified file.
        Parameters:
            filename (str) : filename to detect faces in
            page, site (str) : used in naming
            out_dir (str) : output directory
            cascade_file (str) : cascade xml file to use with OpenCV
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

            faces = cascade.detectMultiScale(gray,
                                            # detector options
                                            scaleFactor = 1.1,
                                            minNeighbors = 5,
                                            minSize = (24, 24))
            if len(faces) > 0:
                # remove image
                os.remove(filepath)
                removed_count += 1
                #print("removed ", filename)
        except:
            pass
            #print("problem in loading file: ", filepath)
    print(f"Removed {removed_count} images")


# apply cascade as demonstrated in example
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', required=True, help="image dir to filter")
    parser.add_argument('--cascade_file', required=True, help="cascade file to use")
    args = parser.parse_args()
    if not os.path.isfile(args.cascade_file):
        raise RuntimeError("%s: not found " % cascade_file)

    cascade = cv2.CascadeClassifier(args.cascade_file)

    detect(args.image_dir, cascade)

