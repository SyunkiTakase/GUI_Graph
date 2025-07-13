import os
import numpy as np
from PIL import Image
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage

def train(device, train_loader, model, criterion, optimizer, scaler, epoch):
    model.train()
    sum_loss = 0.0
    count = 0

    for idx, (img, label) in enumerate(tqdm(train_loader)):
        img = img.to(device, non_blocking=True).float()
        label = label.to(device, non_blocking=True).long()
        
        with torch.autocast(device_type="cuda", dtype=torch.float16):
                logit = model(img)
                loss = criterion(logit, label)
            
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        sum_loss += loss.item()
        count += torch.sum(logit.argmax(dim=1) == label).item()
        
    return sum_loss, count

def validation(device, val_loader, model, criterion):
    model.eval()
    sum_loss = 0.0
    count = 0

    with torch.no_grad():
        for idx, (img, label) in enumerate(tqdm(val_loader)):
            img = img.to(device, non_blocking=True).float()
            label = label.to(device, non_blocking=True).long()
            
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logit = model(img)
                loss = criterion(logit, label)

            sum_loss += loss.item()
            count += torch.sum(logit.argmax(dim=1) == label).item()

    return sum_loss, count