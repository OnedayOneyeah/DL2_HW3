# import modules
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import math 
import cv2
import PIL.Image as Image
import random
PATH = './checkpoints'

class CUB_Dataset(Dataset):
    def __init__(self,img_file, label_file, transform=None):
        self.img = np.load(img_file)
        self.labels = np.load(label_file)
        self.transform = transform

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        image = self.img[idx]
        image = Image.fromarray(np.uint8(image*255))
                                
        if self.transform:
            # print(image[0]*255)
            # print(type(image))
            image = self.transform(image)
            
        label = self.labels[idx]

        return image,label
    
# augmentation
class CUB(Dataset):
    def __init__(self, img_file, label_file,
                 transform=False):   
        
        self.img = np.load(img_file)
        self.labels = np.load(label_file)
        self.transform = transform    
      
        
    def __len__(self):
        return len(self.img)
    
    def __getitem__(self, idx):
        
        image = self.img[idx]
        label = self.labels[idx]

        if self.transform:
            image = apply_transforms(image)
        else:
            image = cv2.resize(image, (224,224))
    
        image = transforms.ToTensor()(normalize(image))
        
        return image, label
    


# utils
def get_optimizer(model, lr = 0.01, wd = 0.0):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optim = torch.optim.Adam(parameters, lr=lr, weight_decay=wd)
    return optim

def save_model(m, p): torch.save(m.state_dict(), p)
    
def load_model(m, p): m.load_state_dict(torch.load(p))

def val_metrics(model, valid_dl):
    model.eval()
    total = 0
    sum_loss = 0
    correct = 0 
    for x, y in valid_dl:
        batch = y.shape[0]
        x = x.cuda().float()
        y = y.cuda().long()
        out = model(x)
        _, pred = torch.max(out, 1)
        correct += pred.eq(y.data).sum().item()
        y = y.long()
        criterion = nn.CrossEntropyLoss()
        loss = criterion(out, y)
        sum_loss += batch*(loss.item())
        total += batch
        
    print("val loss and accuracy", sum_loss/total, correct/total)
    
def LR_range_finder(model, train_dl, lr_low=1e-5, lr_high=1, epochs=1, beta=0.9):
  losses = []
  # Model save path
  p = PATH+"/mode_tmp.pth"
  save_model(model, str(p))
  num = len(train_dl)-1
  mult = (lr_high / lr_low) ** (1.0/num)
  lr = lr_low
  avg_loss = 0.
  best_loss = 0.
  batch_num = 0
  log_lrs = []

  model.train()
  for i in range(epochs):
    for x,y in train_dl:
      batch_num +=1
      optim = get_optimizer(model, lr=lr)
      x = x.cuda().float()
      y = y.cuda().long()   
      out = model(x)
      criterion = nn.CrossEntropyLoss()
      loss = criterion(out, y)

      #Compute the smoothed loss
      avg_loss = beta * avg_loss + (1-beta) *loss.item()
      smoothed_loss = avg_loss / (1 - beta**batch_num)

      #Stop if the loss is exploding
      if batch_num > 1 and smoothed_loss > 4 * best_loss:
        return log_lrs, losses

      #Record the best loss
      if smoothed_loss < best_loss or batch_num==1:
        best_loss = smoothed_loss
      #Store the values
      losses.append(smoothed_loss)
      log_lrs.append(math.log10(lr))

      optim.zero_grad()
      loss.backward()
      optim.step()
      #Update the lr for the next step
      lr *= mult
  load_model(model, str(p))
  
  return log_lrs, losses

def get_triangular_lr(lr_low, lr_high, iterations):
    iter1 = int(0.35*iterations)
    iter2 = int(0.85*iter1)
    iter3 = iterations - iter1 - iter2
    delta1 = (lr_high - lr_low)/iter1
    delta2 = (lr_high - lr_low)/(iter1 -1)
    lrs1 = [lr_low + i*delta1 for i in range(iter1)]
    lrs2 = [lr_high - i*(delta1) for i in range(0, iter2)]
    delta2 = (lrs2[-1] - lr_low)/(iter3)
    lrs3 = [lrs2[-1] - i*(delta2) for i in range(1, iter3+1)]
    return lrs1+lrs2+lrs3

def train_triangular_policy(model, train_dl, valid_dl, lr_low=1e-5, 
                            lr_high=0.01, epochs = 4):
    idx = 0
    iterations = epochs*len(train_dl)
    lrs = get_triangular_lr(lr_low, lr_high, iterations)
    for i in range(epochs):
        model.train()
        total = 0
        sum_loss = 0
        for i, (x, y) in enumerate(train_dl):
            optim = get_optimizer(model, lr = lrs[idx], wd =0)
            batch = y.shape[0]
            x = x.cuda().float()
            y = y.cuda().long()
            out = model(x)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(out, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            idx += 1
            total += batch
            sum_loss += batch*(loss.item())
        print("train loss", sum_loss/total)
        val_metrics(model, valid_dl)
    return sum_loss/total

from datetime import datetime

def training_loop(model, train_dl, valid_dl, steps=3, lr_low=1e-6, lr_high=0.01, epochs = 4):
    for i in range(steps):
        start = datetime.now() 
        loss = train_triangular_policy(model, train_dl, valid_dl, lr_low, lr_high, epochs)
        end = datetime.now()
        t = 'Time elapsed {}'.format(end - start)
        print("----End of step", t)


import math
import numpy as np

def cosine_decay(args, batchs: int, decay_type: int = 1):
    total_batchs = args.max_epochs * batchs
    iters = np.arange(total_batchs - args.warmup_batchs)

    if decay_type == 1:
        schedule = np.array([1e-12 + 0.5 * (args.max_lr - 1e-12) * (1 + \
                             math.cos(math.pi * t / total_batchs)) for t in iters])
    elif decay_type == 2:
        schedule = args.max_lr * np.array([math.cos(7*math.pi*t / (16*total_batchs)) for t in iters])
    else:
        raise ValueError("Not support this deccay type")
    
    if args.warmup_batchs > 0:
        warmup_lr_schedule = np.linspace(1e-9, args.max_lr, args.warmup_batchs)
        schedule = np.concatenate((warmup_lr_schedule, schedule))

    return schedule

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        if param_group["lr"] is not None:
            return param_group["lr"]

def adjust_lr(iteration, optimizer, schedule):
    for param_group in optimizer.param_groups:
        param_group["lr"] = schedule[iteration]

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
                
    elif options == 'last_n_fc': # freeze the last and fc layer
        for name, child in model.named_children():
            if name not in ['layer4','fc']:
                for p in child.parameters():
                    p.requires_grad = False
                    
        for n, p in model.named_parameters():
            if p.requires_grad == True:
                assert 'fc' in n or 'layer4' in n
        
    else:
        print("Train all modules...")
        model.requires_grad_(True)
        
        # check if the model is on train mode for all params
        for n,p in model.named_parameters():
            assert p.requires_grad == True
    
    return model