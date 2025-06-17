import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

def get_optimizer(model, choice, learing_rate, weight_decay=1e-5):
    if choice == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=learing_rate,
            weight_decay=weight_decay
        )
    elif choice == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learing_rate,
            weight_decay=weight_decay
        )
    elif choice == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            momentum=0.9,
            lr=learing_rate,
            weight_decay=weight_decay
        )
    else:
        raise NotImplementedError(f"Optimizer {choice} not implemented")
        
    return optimizer

def get_scheduler(optimizer, choice, learing_rate, steps_per_epoch, n_epochs=5):
    if choice == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=n_epochs,
            eta_min=1e-6
        )
    elif choice == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=False
        )
    elif choice == 'StepLR':
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learing_rate,
            steps_per_epoch=steps_per_epoch,
            epochs=n_epochs,
            pct_start=0.1
        )
    elif choice == 'OneCycleLR':
        scheduler = None  
    else:
        scheduler = None
        
    return scheduler

def get_criterion(choice):
 
    if choice == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
    elif choice == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss(reduction='mean')
    else:
        raise NotImplementedError(f"Criterion {choice} not implemented")
        
    return criterion