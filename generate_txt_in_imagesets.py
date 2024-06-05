import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm


def generate_txt_in_imagesets(vocdevkit_path='VOCdevkit', trainval_percent=1, train_percent=0.9):
    random.seed(0)
    print("Generating txt files in ImageSets.")
    segfilepath = os.path.join(vocdevkit_path, 'VOC2007', 'SegmentationClass')
    saveBasePath = os.path.join(vocdevkit_path, 'VOC2007', 'ImageSets', 'Segmentation')

    temp_seg = os.listdir(segfilepath)
    total_seg = [seg for seg in temp_seg if seg.endswith(".png")]

    num = len(total_seg)
    indices = list(range(num))
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(indices, tv)
    train = random.sample(trainval, tr)

    print(f"train and val size: {tv}")
    print(f"train size: {tr}")

    os.makedirs(saveBasePath, exist_ok=True)
    with open(os.path.join(saveBasePath, 'trainval.txt'), 'w') as ftrainval, \
            open(os.path.join(saveBasePath, 'test.txt'), 'w') as ftest, \
            open(os.path.join(saveBasePath, 'train.txt'), 'w') as ftrain, \
            open(os.path.join(saveBasePath, 'val.txt'), 'w') as fval:

        for i in indices:
            name = total_seg[i][:-4] + '\n'
            if i in trainval:
                ftrainval.write(name)
                if i in train:
                    ftrain.write(name)
                else:
                    fval.write(name)
            else:
                ftest.write(name)

    print("Generated txt files in ImageSets.")
    print("Checking dataset format, this may take a while.")

    classes_nums = np.zeros([256])
    for i in tqdm(indices):
        name = total_seg[i]
        png_file_name = os.path.join(segfilepath, name)
        if not os.path.exists(png_file_name):
            raise ValueError(
                f"Label image {png_file_name} not found. Please check if the file exists and the extension is .png.")

        png = np.array(Image.open(png_file_name), np.uint8)
        if len(np.shape(png)) > 2:
            print(
                f"The shape of label image {name} is {np.shape(png)}, which is not a grayscale or 8-bit color image. Please check the dataset format.")
            print(
                f"The label image should be a grayscale or 8-bit color image, and each pixel value should represent the category of that pixel.")

        classes_nums += np.bincount(np.reshape(png, [-1]), minlength=256)

    print("Pixel value and count:")
    print('-' * 37)
    print("| %15s | %15s |" % ("Key", "Value"))
    print('-' * 37)
    for i in range(256):
        if classes_nums[i] > 0:
            print("| %15s | %15s |" % (str(i), str(classes_nums[i])))
            print('-' * 37)

    if classes_nums[255] > 0 and classes_nums[0] > 0 and np.sum(classes_nums[1:255]) == 0:
        print("Detected that the pixel values in the label only contain 0 and 255, which is incorrect.")
        print(
            "For binary classification, the pixel value of the background should be 0 and the pixel value of the target should be 1.")
    elif classes_nums[0] > 0 and np.sum(classes_nums[1:]) == 0:
        print(
            "Detected that the label only contains background pixel values, which is incorrect. Please check the dataset format.")

if __name__ == "__main__":
    generate_txt_in_imagesets()
