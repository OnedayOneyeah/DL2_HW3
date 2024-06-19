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
from aug import collate_fn, collate_fn_strong, collate_fn_weak

# global vars
BATCH_SIZE = 256
NUM_CLASSES = 200
EPOCHS = 100
PATH = './data'
transform_mean = (0.485, 0.456, 0.406)
transform_std = (0.229, 0.224, 0.225)

PREPROCESSINGS = {
    'Normalize':
        transforms.Compose([
            transforms.Resize((224,224),Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean = transform_mean, std = transform_std)
        ]),
 
    'Augmix':
            transforms.Compose([
            transforms.Resize((224,224),Image.BILINEAR),
            transforms.AugMix(all_ops = True), # imagenet-c overlapping corruptions are excluded
            transforms.ToTensor(),
            transforms.Normalize(mean = transform_mean, std = transform_std)
        ]),
            
    'HERBS_TRAIN':
        transforms.Compose([
                        transforms.Resize((224, 224), Image.BILINEAR),
                        transforms.RandomCrop((224, 224)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
                        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = transform_mean, std = transform_std)
                ]),
    'HERBS_TEST' :
        transforms.Compose([
                        transforms.Resize((224, 224), Image.BILINEAR),
                        transforms.CenterCrop((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = transform_mean, std = transform_std)]),
        
    'FIXMATCH_TEST' : transforms.Compose([
            transforms.Resize((224,224),Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean = transform_mean, std = transform_std)
        ])
        
}
  
def get_dataloader(split:str = 'train', 
                   cub200_transform = PREPROCESSINGS['Normalize'],
                   collate_fn = None,
                   fixmatch = False,
                   ):
    if split == 'train':
        if fixmatch:
            dataset = CUB_Dataset(img_file="./data/CUB_train_images.npy",
                                        label_file="./data/CUB_train_labels.npy",
                                        )
        else:
            dataset = CUB_Dataset(img_file="./data/CUB_train_images.npy",
                                        label_file="./data/CUB_train_labels.npy",
                                        transform = cub200_transform)
    else:
        dataset = CUB_Dataset(img_file="./data/CUB_val_images.npy",
                                        label_file="./data/CUB_val_labels.npy",
                                        transform=cub200_transform)
    
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=BATCH_SIZE, 
                                             shuffle=True if split == 'train' else False,
                                             collate_fn = collate_fn)
        
    return dataloader


def load_model(model: torch.nn.Module, 
               ckpt_dir = './cifar100_pretraining/checkpoints',
               ckpt = None,
               pretrained = True,
               options = None,
               device = None):
    
    # load checkpoints (% acc.)
    if pretrained:
        print("Load the pretrained checkpoint...")
        print(ckpt_dir)
        ckpt_dir = os.path.join(ckpt_dir, ckpt)
        checkpoint = torch.load(ckpt_dir, map_location = device)
    
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # modify the last classifier
    # print(model);exit()
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 200)

    # xavier initialization
    nn.init.xavier_normal_(model.fc.weight)
    nn.init.constant_(model.fc.bias, 0.0)
    
    
    # print(model);exit()
        
    model = configure_model(model, options = options)
    
    print("The model is properly loaded!:)))")
    
    return model

def build_model_and_optimizer(model: str = 'resnet34',
                              ckpt_dir = None,
                              ckpt = None,
                              device = None,
                              pretrained = True,
                              cfg = None):
    # build model
    if model == 'resnet18':
        model = models.resnet18(pretrained = False)
        
    elif model == 'resnet34':
        model = models.resnet34(pretrained = False)
    
    elif model == 'resnet50':
        model = models.resnet50(pretrained = False)
    else:
        raise NotImplementedError
    
    # load the pretrained weights
    if pretrained:
        print("Load the pretrained checkpoint...")
        
        # ckpt_dir = os.path.join(ckpt_dir, ckpt)
        ckpt_dir = './cifar100_pretraining/checkpoints/resnet18_HERBS_Tmax300_ep300.pt'
        # ckpt_dir = './cifar100_pretraining/checkpoints/resnet34_HERBS_Tmax300_ep300.pt'
        checkpoint = torch.load(ckpt_dir, map_location = device)
        print(ckpt_dir, f'\tbest val acc: {checkpoint['test_acc']}')
    
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # modify the last classifier
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 200)
    
    # check if the model is properly freezed
    model = configure_model(model, options = cfg.options)
    
    print("The model is properly loaded!:)))")
        
    model = model.to(device)
    
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optim.SGD(parameters, lr = cfg.lr, weight_decay = cfg.weight_decay, momentum = cfg.momentum)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = EPOCHS)

    return model, optimizer, scheduler

def train(model, 
          train_loader,
          criterion,
          scheduler):
    
    # setting    
    model.train()
    train_loss = 0.
    correct = 0
    tqdm_bar = tqdm(train_loader)
    
    for image, label in tqdm_bar:
        optimizer.zero_grad()
        # update scheduler
        image, label = image.to(DEVICE), label.to(DEVICE)
        
        preds = model(image)
        loss = criterion(preds, label)
            
        prediction = preds.max(1, keepdim = True)[1]
    
        correct += prediction.eq(label.view_as(prediction)).sum().item()    
        
        loss.backward()
        train_loss += float(loss.item())
        
        # update model
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
    parser.add_argument("--lr",
                        default = 0.01,
                        type = float)
    parser.add_argument("--momentum",
                        default = 0.9,
                        type = float)
    parser.add_argument("--weight_decay",
                        default = 0.0001,
                        type = float)
    parser.add_argument("--aug_sev",
                        default = 'both',
                        type = str,
                        help = 'both/strong/weak')
    return parser

if __name__ == "__main__":
    cfg = get_argparser().parse_args()
    arr = []
    
    # additional args
    if cfg.pretrained:
        cfg.ckpt_dir = './cifar100_pretraining/checkpoints'
        # cfg.ckpt = f'{cfg.model}_Augmix.pt'
        cfg.ckpt = f'{cfg.model}_HERBS_Tmax300_ep300.pt'
        
    

    print(f"""
          |CONFIG|================
          augmix: {cfg.augmix}
          model: {cfg.model}
          augmentation: {cfg.aug},
          collate_fn: {cfg.aug_sev},
          mixup: {cfg.mixup}
          pretrained: {cfg.pretrained}
          options: {cfg.options}
          lr: {cfg.lr}
          weight_decay: {cfg.weight_decay}
          momentum: {cfg.momentum}
          =========================
          """)
    
    if cfg.check:
        # check dataset loading
        sample_loader = get_dataloader(split = 'val', 
                                       cub200_transform=PREPROCESSINGS[f'{cfg.aug}'])
        examples = next(iter(sample_loader))
        print(f'BS: {examples[0].shape[0]}, IMG SHAPE: {examples[0].shape[1:]}, Dataset')
        
        # check model loading
        # build_model_and_optimizer();exit()

    print("Collecting data...")
    print("Choosen augmentation: ", cfg.aug)
    print("===============")
    
    if cfg.aug == 'FIXMATCH':
        if cfg.aug_sev == 'both':
            collate_function = collate_fn
        elif cfg.aug_sev == 'strong':
            collate_function = collate_fn_strong
        elif cfg.aug_sev == 'weak':
            collate_function = collate_fn_weak
        else:
            raise ValueError
        
        train_loader = get_dataloader(split = 'train', 
                                      collate_fn = collate_function,
                                      fixmatch = True)
        val_loader = get_dataloader(split = 'val', 
                                    cub200_transform=PREPROCESSINGS['FIXMATCH_TEST'])
    
    else:
        train_loader = get_dataloader(split = 'train', 
                                      cub200_transform=PREPROCESSINGS[cfg.aug if cfg.aug != 'HERBS' else cfg.aug + '_TRAIN'])
        val_loader = get_dataloader(split = 'val', 
                                    cub200_transform=PREPROCESSINGS['Normalize' if cfg.aug != 'HERBS' else cfg.aug + '_TEST'])
        # print("data loder ehrehere")
    
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
                                                            # ckpt = f'{cfg.model}_Augmix.pt' if cfg.pretrained else None,
                                                            ckpt_dir = cfg.ckpt_dir,
                                                            ckpt = cfg.ckpt,
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
            
        train_loss, train_accuracy = train(model, train_loader, criterion,  scheduler)
        test_loss, test_accuracy = evaluate(model, val_loader, criterion)
        arr.append(test_accuracy)
        
        print("\n[EPOCH: {}], \tLR: {:.4f}, \tTrain Loss: {:.4f}, \tTrain Accuracy: {:.2f} %, \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} % \n".format(
        epoch, scheduler.get_last_lr()[-1], train_loss, train_accuracy, test_loss, test_accuracy))

        MODEL_PATH = f'./checkpoints'
        
        if not os.path.exists(MODEL_PATH):
            os.mkdir(MODEL_PATH)
        
        if test_accuracy >= best_acc:
            print("New checkpoint! Test Acc.", test_accuracy)
            torch.save({
                'EPOCHS' : EPOCHS,
                'epoch' : epoch,
                'aug' : cfg.aug,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': test_loss,
                'test_acc': test_accuracy
            },
                MODEL_PATH + f'/{cfg.model}_{cfg.aug}_{cfg.options}_{EPOCHS}.pt'
                    )
            best_acc = test_accuracy
    
    import pickle as pkl
    with open(f'./{cfg.aug_sev}.pkl', 'wb') as f:
        pkl.dump(arr, f)
    
    
    
    