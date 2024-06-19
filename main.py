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
from utils import cosine_decay, adjust_lr, get_lr, configure_model

# global vars
BATCH_SIZE = 8
NUM_CLASSES = 200
EPOCHS = 80
PATH = './data'
transform_mean = (0.485, 0.456, 0.406)
transform_std = (0.229, 0.224, 0.225)

PREPROCESSINGS = {
    'Normalize':
        transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = transform_mean, std = transform_std)
        ]),
 
    'Augmix':
            transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.AugMix(all_ops = True), # imagenet-c overlapping corruptions are excluded
            transforms.ToTensor(),
            transforms.Normalize(mean = transform_mean, std = transform_std)
        ]),
            
    'HERBS_TRAIN':
        transforms.Compose([
                        transforms.Resize((600, 600), Image.BILINEAR),
                        transforms.RandomCrop((600, 600)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
                        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = transform_mean, std = transform_std)
                ]),
    'HERBS_TEST'  :
        transforms.Compose([
                        transforms.Resize((600, 600), Image.BILINEAR),
                        transforms.CenterCrop((600, 600)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = transform_mean, std = transform_std)])
}


def mixup(input, target, alpha = 1.0):
    lmd = np.random.beta(alpha, alpha)
    bs = input.shape[0]
    idx = torch.randperm(bs)
    mixed_input = lmd * input + (1-lmd) * input[idx, :]
    labels_a, labels_b = target, target[idx]
    
    return mixed_input, labels_a, labels_b

def MixUpLoss(criterion, pred, labels_a, labels_b, lmd = 1):
    return lmd * criterion(pred, labels_a) + (1 - lmd) * criterion(pred, labels_b)    
    
def get_dataloader(split:str = 'train', 
                   cub200_transform = PREPROCESSINGS['Normalize']):
    if split == 'train':
        dataset = CUB_Dataset(img_file="./data/CUB_train_images.npy",
                                        label_file="./data/CUB_train_labels.npy",
                                        transform=cub200_transform)
    else:
        dataset = CUB_Dataset(img_file="./data/CUB_val_images.npy",
                                        label_file="./data/CUB_val_labels.npy",
                                        transform=cub200_transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
    
    return dataloader


def configure_model(model, options = None): # fc / all
    
    """Configure model"""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    # configure norm for tent updates: enable grad + force batch statisics
    if options == 'fc':
        print("Freeze model but the classifier...")
        model.requires_grad_(False)
        for m in model.modules():
            if isinstance(m, nn.Linear):
                m.requires_grad_(True)
        
        # check if the model is freezed but the classifier
        for n,p in model.named_parameters():
            if p.requires_grad == True:
                assert 'fc' in n
                
    else:
        print("Train all modules...")
        model.requires_grad_(True)
        
        # check if the model is on train mode for all params
        for n,p in model.named_parameters():
            assert p.requires_grad == True
    
    return model


def load_model(model: torch.nn.Module, 
               ckpt_dir = './cifar100_pretraining/checkpoints',
               ckpt = None,
               pretrained = True,
               options = None,
               device = None):
    
    # load checkpoints (% acc.)
    if pretrained:
        print("Load the pretrained checkpoint...")
        ckpt_dir = os.path.join(ckpt_dir, ckpt)
        checkpoint = torch.load(ckpt_dir, map_location = device)
    
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # modify the last classifier
    # print(model);exit()
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 200)
    
    # print(model);exit()
        
    model = configure_model(model, options = options)
    
    print("The model is properly loaded!:)))")
    
    return model

def build_model_and_optimizer(model: str = 'resnet34',
                              ckpt = None,
                              device = None,
                              pretrained = True,
                              cfg = None):
    # build model
    if model == 'resnet34':
        model = models.resnet34(pretrained = False)
    
    elif model == 'resnet50':
        model = models.resnet50(pretrained = False)
    else:
        pass
    
    # load the pretrained weights
    if pretrained:
        print("Load the pretrained checkpoint...")
        ckpt_dir = os.path.join(ckpt_dir, ckpt)
        checkpoint = torch.load(ckpt_dir, map_location = device)
    
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # modify the last classifier
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 200)
    
    # check if the model is properly freezed
    model = configure_model(model, options = cfg.options)
    print("The model is properly loaded!:)))")
        
    model = model.to(device)
    
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    if cfg.aug == 'HERBS':
        cfg.max_epochs = EPOCHS
        cfg.max_lr = 0.01
        cfg.wdecay = 0.0005
        cfg.warmup_batchs = 100
        optimizer = torch.optim.SGD(parameters, lr=cfg.max_lr, nesterov=True, momentum=0.9, weight_decay=cfg.wdecay)
        scheduler = cosine_decay(cfg, len(train_loader))

    else:
        optimizer = optim.SGD(parameters, lr = 0.01, weight_decay = 0.001, momentum = 0.9)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = EPOCHS//10)

    return model, optimizer, scheduler

def train(model, 
          train_loader,
          criterion,
          optimizer, 
          schedule,
          cfg,
          epoch):
    
    # setting    
    model.train()
    train_loss = 0.
    correct = 0
    tqdm_bar = tqdm(train_loader)
    
    for i, data in enumerate(tqdm_bar):
        # update scheduler
        iterations = epoch * len(train_loader) + i
        
        adjust_lr(iterations, optimizer, schedule)
        if i == 100:
            exit()
        
        image, label = data
        # print(len(image)) # 3
        # print(image[0].shape) # (batch_size, c, h, w)
                    
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
        
        # update model
        if (i+1)%4 == 0 or (i+1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()

        tqdm_bar.set_description("Epoch {} - train loss {:.6f}".format(epoch, loss.item()))
        
    # scheduler.step()
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
    parser.add_argument("--augmix",
                        default = False,
                        action='store_true',
                        help = "apply augmix")
    parser.add_argument("--random_seed", type=int, default=1, help="random seed (default: 1)")        
    parser.add_argument("--num_workers", type=int, default=2,
                        help="number of workers on dataloaders")
    parser.add_argument("--model",
                        default = 'resnet34', # resnet34, wrn50-2, wrn-28-2
                        type = str,
                        help = "choose the model architecture")
    parser.add_argument("--check",
                        default = False,
                        # type = bool,
                        action = 'store_true',
                        help = 'check if the model is properly load')
    parser.add_argument("--aug",
                        default = 'Normalize',
                        type = str,
                        help = 'choose augmentation methods')
    parser.add_argument("--mixup",
                        action = 'store_true',
                        default = False,
                        help = 'apply mixup augmentation')
    parser.add_argument("--pretrained",
                        action = 'store_true',
                        default = False,
                        help = 'use pretrained weights')
    parser.add_argument("--options",
                        default = None,
                        type = str,
                        help = "modules to be trained")
    return parser

if __name__ == "__main__":
    cfg = get_argparser().parse_args()
    
    # additional args
    if cfg.pretrained:
        cfg.ckpt_dir = './cifar100_pretraining/checkpoints'
        cfg.ckpt = f'{cfg.model}_{cfg.aug}.pt'

    print(f"""
          |CONFIG|================
          augmix: {cfg.augmix}
          model: {cfg.model}
          augmentation: {cfg.aug}
          mixup: {cfg.mixup}
          pretrained: {cfg.pretrained} 
          =========================
          """)
    
    if cfg.check:
        # check dataset loading
        sample_loader = get_dataloader(split = 'val', cub200_transform=PREPROCESSINGS['Augmix'])
        examples = next(iter(sample_loader))
        print(f'BS: {examples[0].shape[0]}, IMG SHAPE: {examples[0].shape[1:]}, Dataset')
        
        # check model loading
        # build_model_and_optimizer();exit()

    print("Collecting data...")
    print("Choosen augmentation: ", cfg.aug)
    print("===============")
    
    train_loader = get_dataloader(split = 'train', cub200_transform=PREPROCESSINGS[cfg.aug if cfg.aug != 'HERBS' else cfg.aug + '_TRAIN'])
    val_loader = get_dataloader(split = 'val', cub200_transform=PREPROCESSINGS['Normalize' if cfg.aug != 'HERBS' else cfg.aug + '_TEST'])
    
    examples = next(iter(train_loader))
    print("Dataset: %s, Train set: %d, Val set: %d" %
        ("Cub200", len(train_loader)*BATCH_SIZE, len(val_loader)*BATCH_SIZE))

    print(f'BS: {examples[0].shape[0]}, IMG SHAPE: {examples[0].shape[1:]}, Dataset')
    
    print("===============")
    print("Data Collected")
    
    # build model, optimizer, and scheduler
    # cfg.pretrained = False
    print(f"Building {cfg.model}..., pretrained?: {cfg.pretrained}")
    print("=====================")
    
    # os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_id
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Device: %s" % DEVICE)
    model, optimizer, scheduler = build_model_and_optimizer(model = cfg.model, 
                                                            ckpt = f'{cfg.model}_Augmix.pt' if cfg.pretrained else None,
                                                            device = DEVICE,
                                                            pretrained = cfg.pretrained,
                                                            cfg = cfg)
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
    print("TOTAL EPOCHS: ", EPOCHS)
    print("Train starts~!!!")

    best_acc = 0.
    for epoch in range(1, EPOCHS + 1):

        # model = nn.DataParallel(model) #def train(model, 
        train_loss, train_accuracy = train(model, train_loader, criterion,  optimizer, scheduler, cfg, epoch)
        test_loss, test_accuracy = evaluate(model, val_loader, criterion)
        
        print("\n[EPOCH: {}], \tLR: {:.4f}, \tTrain Loss: {:.4f}, \tTrain Accuracy: {:.2f} %, \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} % \n".format(
        epoch, get_lr(optimizer), train_loss, train_accuracy, test_loss, test_accuracy))

        MODEL_PATH = f'./checkpoints'
        
        if not os.path.exists(MODEL_PATH):
            os.mkdir(MODEL_PATH)
        
        if test_accuracy >= best_acc:
            print("New checkpoint! Test Acc.", test_accuracy)
            torch.save({
                'epoch' : epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_loss,
                'test_acc': test_accuracy
            },
                MODEL_PATH + f'/{cfg.model}_{cfg.aug}.pt'
                    )
            best_acc = test_accuracy
    
    
    
    
    