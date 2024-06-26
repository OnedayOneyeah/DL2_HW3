# data augmentation
# import modules
import torch
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from tqdm.auto import tqdm
import random
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image
from torch.optim.lr_scheduler import LambdaLR


transform_mean = (0.485, 0.456, 0.406)
transform_std = (0.229, 0.224, 0.225)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=transform_mean, std=transform_std)])


PARAMETER_MAX = 10
IMG_SIZE = 224

def AutoContrast(img, **kwarg):
    return PIL.ImageOps.autocontrast(img)
def Brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Brightness(img).enhance(v)
def Color(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Color(img).enhance(v)
def Contrast(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v)
def Cutout(img, v, max_v, bias=0):
    if v == 0:
        return img
    v = _float_parameter(v, max_v) + bias
    v = int(v * min(img.size))
    return CutoutAbs(img, v)
def CutoutAbs(img, v, **kwarg):
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    color = (127, 127, 127) # gray
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img
def Equalize(img, **kwarg):
    return PIL.ImageOps.equalize(img)
def Identity(img, **kwarg):
    return img
def Invert(img, **kwarg):
    return PIL.ImageOps.invert(img)
def Posterize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.posterize(img, v)
def Rotate(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.rotate(v)
def Sharpness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v)
def ShearX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))
def ShearY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))
def Solarize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.solarize(img, 256 - v)
def SolarizeAdd(img, v, max_v, bias=0, threshold=128):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    img_np = np.array(img).astype(np.int)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)
def TranslateX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))
def TranslateY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))
def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX
def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)

def fixmatch_augment_pool(): # FixMatch paper
    augs = [(AutoContrast, None, None), (Brightness, 0.9, 0.05), (Color, 0.9, 0.05), (Contrast, 0.9, 0.05),
            (Equalize, None, None), (Identity, None, None), (Posterize, 4, 4), (Rotate, 30, 0), (Sharpness, 0.9, 0.05),
            (ShearX, 0.3, 0), (ShearY, 0.3, 0), (Solarize, 256, 0), (TranslateX, 0.3, 0), (TranslateY, 0.3, 0)]
    return augs

class RandAugmentMC(object):
    def __init__(self, n, m, size):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.size = size,
        self.augment_pool = fixmatch_augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)
        img = CutoutAbs(img, int(self.size[0]*0.5))
        return img

# weak / strong augs
def get_weak_augment(dataset):
    normalize = transforms.Normalize(mean=transform_mean, std=transform_std)

    train_transform = transforms.Compose([
        # transforms.Resize((IMG_SIZE,IMG_SIZE)), # added
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=IMG_SIZE, padding=int(IMG_SIZE*0.125), padding_mode='reflect'),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform(dataset)

def get_strong_augment(dataset):

    normalize = transforms.Normalize(mean=transform_mean, std=transform_std)
    train_transform = transforms.Compose([
        # transforms.Resize((IMG_SIZE,IMG_SIZE)), # added
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=IMG_SIZE, padding=int(IMG_SIZE*0.125), padding_mode='reflect'),
        RandAugmentMC(n=2, m=10, size = IMG_SIZE),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform(dataset)

# collate func
def collate_labeled_list(batch):
    return [(get_weak_augment(image), target) for image, target in batch]

def collate_unlabeled_list(batch):
    return [(get_weak_augment(image), get_strong_augment(image), target) for image, target in batch]

def collate_labeled(batch):
    images, targets = [], []
    for image, target in batch:
        images.append(get_weak_augment(image))
        targets.append(target)
    return torch.stack(images), torch.LongTensor(targets)

def collate_unlabeled(batch):
    images_weak, images_strong, targets = [], [], []
    for image, target in batch:
        images_weak.append(get_weak_augment(image))
        images_strong.append(get_strong_augment(image))
        targets.append(target)
    return torch.stack(images_weak), torch.stack(images_strong), torch.LongTensor(targets)

def collate_val(batch):
    return [(transform(image), target) for image, target in batch]

def collate_fn(batch): # regularization
    images, targets = [], []
    for image, target in batch:
        if random.random() < 0.5:
            images.append(get_strong_augment(image))
        else:
            images.append(get_weak_augment(image))
            
        targets.append(target)
        
    return torch.stack(images), torch.LongTensor(targets)