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
from aug import *

# global vars
BATCH_SIZE = 256
# BATCH_SIZE = 128
NUM_CLASSES = 100
EPOCHS = 300 # 100
PATH = '../data'
transform_mean = (0.507, 0.487, 0.441)
transform_std = (0.267, 0.256, 0.276)

PREPROCESSINGS = {
    'Normalize':
        transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = transform_mean, std = transform_std)
        ]),
 
    'Augmix':
            transforms.Compose([
            transforms.Resize((224,224)),
            transforms.AugMix(all_ops = True), # imagenet-c overlapping corruptions are excluded
            transforms.ToTensor(),
            transforms.Normalize(mean = transform_mean, std = transform_std)
        ]),
    'Augmix_alphs':
        transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.AugMix(all_ops = True), # imagenet-c overlapping corruptions are excluded
            transforms.ToTensor(),
            transforms.Normalize(mean = transform_mean, std = transform_std)
        ]),
    'HERBS_TRAIN':
        transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.RandomCrop((224, 224)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
                        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
                        # customize
                        transforms.RandomVerticalFlip(p=0.1),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = transform_mean, std = transform_std)
                ]),
    'HERBS_TEST'  :
        transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.CenterCrop((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = transform_mean, std = transform_std)])
                    
}


# # input image dimensions
# img_rows, img_cols = 64, 64
# # The images are RGB
# img_channels = 3
def mixup(input, target, alpha = 1.0):
    lmd = np.random.beta(alpha, alpha)
    bs = input.shape[0]
    idx = torch.randperm(bs)
    mixed_input = lmd * input + (1-lmd) * input[idx, :]
    labels_a, labels_b = target, target[idx]
    
    return mixed_input, labels_a, labels_b

def MixUpLoss(criterion, pred, labels_a, labels_b, lmd = 1):
    return lmd * criterion(pred, labels_a) + (1 - lmd) * criterion(pred, labels_b)
    
def get_dataloader(split:str = 'train', cifar100_transform = PREPROCESSINGS['Normalize']):
    if split == 'train':
        dataset = datasets.CIFAR100(root="../data/", train=True, download=True, transform=cifar100_transform)
        print('get cifar100 train data')
        
    elif split == 'val':
        dataset = datasets.CIFAR100(root="../data/", train=False, download=True, transform=cifar100_transform)    
        print('get cifar100 val data')
        
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

    return dataloader
  
def build_model_and_optimizer(device:None, 
                              model_arc: str = '',
                              pretrained: bool = False,
                              resume = False):
    
    if model_arc == 'resnet34':
        model = models.resnet34(pretrained = pretrained)
    elif model_arc == 'resnet50':
        model = models.resnet50(pretrained = pretrained)
    elif model_arc == 'resnet18':
        model = models.resnet18(pretrained = pretrained)
    else:
        raise NotImplementedError
    
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr = 0.01, weight_decay = 0.001, momentum = 0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = EPOCHS)
    start_epoch = 1
    
    if resume:
        ckpt_dir = f'./checkpoints/{model_arc}_HERBS_Tmax{EPOCHS}_ep{EPOCHS}.pt'
        # ckpt_dir = './checkpoints/resnet32_HERBS_Tmax50_ep300.pt'
        ckpt = torch.load(ckpt_dir)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch']

    return model, optimizer, scheduler, start_epoch

def train(model, 
          train_loader,
          criterion,
          scheduler,
          cfg,
          ):
    
    # setting    
    model.train()
    train_loss = 0.
    correct = 0
    tqdm_bar = tqdm(train_loader)
    
    for image, label in tqdm_bar:
        # print(len(image)) # 3
        # print(image[0].shape) # (batch_size, c, h, w)
        optimizer.zero_grad()
                    
        image, label = image.to(DEVICE), label.to(DEVICE)
        
        if cfg.mixup:
            mixed_input, labels_a, labels_b = mixup(image, label)
            preds = model(mixed_input)
            loss = MixUpLoss(criterion, preds, labels_a, labels_b)
        else:
            preds = model(image)
            loss = criterion(preds, label)
            
        prediction = preds.max(1, keepdim = True)[1]
    
        correct += prediction.eq(label.view_as(prediction)).sum().item()    
        loss.backward()
        train_loss += float(loss.item())
        
        optimizer.step()
        tqdm_bar.set_description("Epoch {} - train loss {:.6f}".format(epoch, loss.item()))
    
    scheduler.step()
    # make dataloader
    
    train_loss /= len(train_loader.dataset)
    train_acc = 100. * correct / len(train_loader.dataset)
    return train_loss, train_acc

def evaluate(model,
             test_loader,
             criterion,
            ):
    # setting
    model.eval()
    test_loss = 0.
    correct = 0
    
    with torch.no_grad():
        for image, label in tqdm(test_loader):
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            preds = model(image)
            test_loss += criterion(preds, label).item()
            prediction = preds.max(1, keepdim = True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()
            
    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    
    return test_loss, test_acc

def get_argparser():
    # parse arg
    parser = argparse.ArgumentParser()
    parser.add_argument("--initialization",
                        default = 'he_normal', # random_uniform, random_normal, he_normal
                        type = str,
                        help = "model initialization method")
    parser.add_argument("--augmix",
                        default = False,
                        action='store_true',
                        help = "apply augmix")
    parser.add_argument("--gpu_id", type=str, default='0', help="GPU ID")
    parser.add_argument("--random_seed", type=int, default=1, help="random seed (default: 1)")        
    parser.add_argument("--num_workers", type=int, default=2,
                        help="number of workers on dataloaders")
    parser.add_argument("--model",
                        default = 'resnet34', # resnet34, wrn50-2, wrn-28-2
                        type = str,
                        help = "choose the model architecture")
    parser.add_argument("--check",
                        default = False,
                        type = bool,
                        help = '5 epochs to check if the program works')
    parser.add_argument("--aug",
                        default = 'Normalize',
                        type = str,
                        help = 'choose augmentation methods')
    parser.add_argument("--mixup",
                        default = False,
                        type = bool,
                        help = 'apply mixup')
    parser.add_argument("--resume",
                        default = False,
                        action = 'store_true',
                        help = 'resume training from the last best checkpoint')
    
    return parser

if __name__ == "__main__":
    
    cfg = get_argparser().parse_args()
    print("Collecting data...")
    print("===============")
    
    train_loader = get_dataloader(split = 'train', cifar100_transform=PREPROCESSINGS[cfg.aug if cfg.aug != 'HERBS' else 'HERBS_TRAIN'])
    val_loader = get_dataloader(split = 'val', cifar100_transform=PREPROCESSINGS['Normalize' if cfg.aug != 'HERBS' else 'HERBS_TEST'])
    
    examples = next(iter(train_loader))
    
    print("Dataset: %s, Train set: %d, Val set: %d" %
          ("Cifar100", len(train_loader), len(val_loader)))
    
    print(f'BS: {examples[0].shape[0]}, IMG SHAPE: {examples[0].shape[1:]}, Dataset')
    
    print("===============")
    print("Data Collected")
    
    # build model, optimizer, and scheduler
    print(f"Building {cfg.model}...")
    print("=====================")
    
    # os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_id
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Device: %s" % DEVICE)
    model, optimizer, scheduler, start_epoch = build_model_and_optimizer(DEVICE, 
                                                                         model_arc = cfg.model,
                                                                         resume = cfg.resume)
    model = nn.DataParallel(model).to(DEVICE)
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()
    # Setup random seed
    torch.manual_seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    random.seed(cfg.random_seed)
    print("=====================")
    print("Done.")
    
    # train
    print("Train starts~!!!")

    best_acc = 0.
    for epoch in range(start_epoch, EPOCHS + 1):

        # model = nn.DataParallel(model)
        train_loss, train_accuracy = train(model, train_loader, criterion, scheduler, cfg)
        test_loss, test_accuracy = evaluate(model, val_loader, criterion)
        print("\n[EPOCH: {}], \tLR: {:.4f}, \tTrain Loss: {:.4f}, \tTrain Accuracy: {:.2f} %, \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} % \n".format(
        epoch, scheduler.get_last_lr()[-1], train_loss, train_accuracy, test_loss, test_accuracy))

        MODEL_PATH = f'./checkpoints'
        
        if not os.path.exists(MODEL_PATH):
            os.mkdir(MODEL_PATH)
        
        if test_accuracy >= best_acc:
            print("New checkpoint! Test Acc.", test_accuracy)
            torch.save({
                'epoch' : epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict' : scheduler.state_dict(),
                'loss': test_loss,
                'test_acc': test_accuracy
            },
                MODEL_PATH + f'/{cfg.model}_{cfg.aug}_Tmax{EPOCHS}_ep{EPOCHS}.pt'
                    )
            # torch.save(model.module.state_dict(), MODEL_PATH + '/best-model.pt')
            best_acc = test_accuracy