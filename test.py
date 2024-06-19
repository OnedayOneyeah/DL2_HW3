# import modules
import time
import numpy as np
import random

# build model and optimizer
# from resnet50 import ResNet50
# from resnet50_ import ResNet50
import torch.optim as optim
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn

# utils
import argparse
from tqdm import tqdm
from collections import OrderedDict
import pickle as pkl
import os
import torch.nn.functional as F
import warnings
from torchvision import datasets, models
warnings.filterwarnings('ignore')
from utils import CUB_Dataset
from PIL import Image
from aug import collate_fn

# global vars
BATCH_SIZE = 256
IMG_SIZE = 224
SAVE_PATH = './202335068_HANYEWON_HW3'

class TestDataset(Dataset):
    def __init__(self, img_file, transform=None):
        self.img =np.load(img_file)
        self.transform = transform

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        image = self.img[idx]
        image = Image.fromarray(np.uint8(image*255))
        
        if self.transform is not None:
            image = self.transform(image)

        return image

test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


def load_model():
    return model

def test(model, test_loader, DEVICE):
  model.eval()
  test_predictions = []

  with torch.inference_mode():
      for i, data in enumerate(tqdm(test_loader)):
          data = data.float().to(DEVICE)
          output = model(data)
          test_predictions.append(output.cpu())

  return torch.cat(test_predictions, dim=0)




if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        default = 'resnet18',
                        type = str,
                        help = 'model architecture')
    parser.add_argument('--ckpt_dir',
                        default = './checkpoints',
                        type = str,
                        help = 'checkpoint directory of the pretrained model')
    parser.add_argument('--ckpt',
                        default = 'resnet18_FIXMATCH_None_100.pt',
                        type = str,
                        help = 'checkpoint of the pretrained model')
    cfg = parser.parse_args()
    
    # load dataset
    test_dataset = TestDataset(img_file="./data/CUB_test_images.npy",transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)
    
    examples = next(iter(test_loader))
    print("Test Data is loaded...\n===================")
    print(f'BS: {examples[0].shape[0]}, IMG SHAPE: {examples[0].shape[1:]}, Dataset')
    
    # load model
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Device: %s" % DEVICE)
    
    if cfg.model == 'resnet18':
        model = models.resnet18(pretrained = False)
    elif cfg.model == 'resnet34':
        model = models.resnet34(pretrained = False)
    elif cfg.model == 'resnet50':
        model = models.resnet50(pretrained = False)
    else:
        raise NotImplementedError
    
    # change the classifier
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 200)
    
    ckpt = os.path.join(cfg.ckpt_dir, cfg.ckpt)
    print(f"Loading pretrained {cfg.model} : {ckpt}...")
    
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    print(f"Pretrained {cfg.model} is loaded...\tbest_val_acc: {checkpoint['test_acc']:.2f}%")
    
    
    # test
    # Save test output npy file
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    predictions = test(model, test_loader, DEVICE)
    
    np.save(SAVE_PATH+'/predictions.npy', predictions.numpy())