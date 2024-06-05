import random
import numpy as np
import torch
from PIL import Image
import os
from torch.hub import load_state_dict_from_url


# Convert image to RGB format to prevent grayscale image errors during prediction
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        return image.convert('RGB')


# Resize the input image
def resize_image(image, size):
    iw, ih = image.size
    w, h = size

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    return new_image, nw, nh


# Get learning rate from optimizer
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# Set random seed for reproducibility
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Set seed for DataLoader workers
def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


# Preprocess input image
def preprocess_input(image):
    image /= 255.0
    return image


# Display configuration parameters
def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)


# Download pretrained weights for backbone networks
def download_weights(backbone, model_dir="./model_data"):
    download_urls = {
        'vgg': 'https://download.pytorch.org/models/vgg16-397923af.pth',
        'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth'
    }
    url = download_urls[backbone]

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)
