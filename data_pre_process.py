import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import imageio
from albumentations import HorizontalFlip, VerticalFlip, Rotate


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_data(path):
    train_x = sorted(glob(os.path.join(path, "Train", "image", "*.png")))
    train_y = sorted(glob(os.path.join(path, "Train", "mask", "*.png")))

    test_x = sorted(glob(os.path.join(path, "Test", "image", "*.png")))
    test_y = sorted(glob(os.path.join(path, "Test", "mask", "*.png")))

    return (train_x, train_y), (test_x, test_y)


def augumentation(images, masks, save_path, augment=True):
    size = (512, 512)

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        """Extracting the name of the file"""
        name = x.split("/")[-1].split(".")[0]

        """Reading image and mask"""
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = imageio.imread(y)

        if augment == True:

            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented["image"]
            y2 = augmented["mask"]

            aug = Rotate(limit=45, p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented["image"]
            y3 = augmented["mask"]

            X = [x1, x2, x3]
            Y = [y1, y2, y3]

        else:
            X = [x]
            Y = [y]

        index = 0
        for i, m in zip(X, Y):
            i = cv2.resize(i, size)
            m = cv2.resize(m, size)
            tmp_image_name = f"{name}_{index}.png"
            tmp_mask_name = f"{name}_{index}.png"

            image_path = os.path.join(save_path, "image", tmp_image_name)
            mask_path = os.path.join(save_path, "mask", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            index += 1


if __name__ == "__main__":
    """Seeding"""
    np.random.seed(42)

    """Load the data"""
    data_path = "/home/surya/projects/Breast-Cancer-Segmentation-using-Deep-Learning/Datasets/"
    (train_x, train_y), (test_x, test_y) = load_data(data_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")

    """Create Directories to sava augumented data"""
    create_dir("/home/surya/projects/Breast-Cancer-Segmentation-using-Deep-Learning/Datasets//new_dataset/train/image")
    create_dir("/home/surya/projects/Breast-Cancer-Segmentation-using-Deep-Learning/Datasets//new_dataset/train/mask/")
    create_dir("/home/surya/projects/Breast-Cancer-Segmentation-using-Deep-Learning/Datasets//new_dataset/test/image/")
    create_dir("/home/surya/projects/Breast-Cancer-Segmentation-using-Deep-Learning/Datasets//new_dataset/test/mask/")

    """Data Augumetation"""
    augumentation(train_x, train_y, "/home/surya/projects/Breast-Cancer-Segmentation-using-Deep-Learning/Datasets//new_dataset/train/", augment=True)
    augumentation(test_x, test_y, "/home/surya/projects/Breast-Cancer-Segmentation-using-Deep-Learning/Datasets//new_dataset/test/", augment=False)
