import os
import numpy as np
import cv2

def saveResult(img_file, img, boxes, dirname='result'):
    """ Save cropped text detection results
    Args:
        img_file (str): image file name
        img (array): raw image context
        boxes (array): array of result file
    Return:
        None
    """
    result_dir = os.path.join('CRAFT-Pytorch', dirname)
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    img = np.array(img)
    filename, _ = os.path.splitext(os.path.basename(img_file))

    for i, box in enumerate(boxes):
        poly = np.array(box).astype(np.int32).reshape((-1, 2))
        
        x_coords = [point[0] for point in poly]
        y_coords = [point[1] for point in poly]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        cropped_image = img[y_min:y_max, x_min:x_max]

        crop_filename = os.path.join(result_dir, f"{filename}_crop_{i}.jpg")
        cv2.imwrite(crop_filename, cropped_image)

def get_files(img_dir):
    """
    Get image files from directory.
    Args:
        img_dir (str): Path to image directory.
    Returns:
        imgs (list): List of image file paths.
    """
    imgs = []
    for dirpath, _, filenames in os.walk(img_dir):
        for file in filenames:
            ext = os.path.splitext(file)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png', '.gif', '.pgm']:
                imgs.append(os.path.join(dirpath, file))
    return imgs

def process_single_image(image_path):
    """
    Process a single image for text detection and save cropped images.
    Args:
        image_path (str): Path to the input image.
    """
    result_folder = 'result'

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    boxes = [
        [[100, 100], [200, 100], [200, 150], [100, 150]],
        [[300, 200], [400, 200], [400, 250], [300, 250]],
    ]

    saveResult(image_path, image, boxes, dirname=result_folder)

# if __name__ == "__main__":
#     image_path = "test_img/0700094.jpg"
#     process_single_image(image_path)
#     print("Results saved to: CRAFT-Pytorch/result")
