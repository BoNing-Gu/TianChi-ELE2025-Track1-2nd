import timm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics import Accuracy
import pytorch_lightning as pl
from pytorch_lightning.loggers.logger import Logger, rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only

import os
import re
from datetime import datetime
import csv
import json

# -------------------------
# Logger & Callback Module
# -------------------------
class CustomLogger(Logger):
    def __init__(self, dirpath, filename):
        super().__init__()
        self._dirpath = dirpath
        self._filename = filename
        self.trainer = None
        
        # Initialize file paths
        self._train_file = os.path.join(self._dirpath, f'{self._filename}-train_metrics.csv')
        self._val_file = os.path.join(self._dirpath, f'{self._filename}-val_metrics.csv')
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self._train_file), exist_ok=True)
        os.makedirs(os.path.dirname(self._val_file), exist_ok=True)
        
        # Initialize CSV files with headers if they don't exist
        self._init_csv_files()

    @property
    def save_dir(self):
        return self._dirpath
    
    @property
    def version(self):
        match = re.search(r'exp_(\w+)', self._dirpath)
        return match.group(1) if match else "unknown_version"
    
    @property
    def name(self):
        return self._filename
    
    def set_trainer(self, trainer):
        self.trainer = trainer

    @rank_zero_only
    def _init_csv_files(self):
        # Initialize training metrics file
        if not os.path.exists(self._train_file):
            with open(self._train_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["step", "epoch", "train_loss", "train_acc", "timestamp"])
        else:
            open(self._train_file, 'a', newline='').close()
        
        # Initialize validation metrics file
        if not os.path.exists(self._val_file):
            with open(self._val_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "val_loss", "val_acc", "timestamp"])
        else:
            open(self._val_file, 'a', newline='').close()

    @rank_zero_only
    def log_hyperparams(self, params, ignore=None):
        try:
            # Save hyperparameters to a separate JSON file
            ignore = ignore or []
            hp_file = os.path.join(self.save_dir, f"{self.name}-hparams.json")
            os.makedirs(os.path.dirname(hp_file), exist_ok=True)
            # [Debug] 打印原始参数信息
            if isinstance(params, dict):
                param_dict = params
            else:
                param_dict = vars(params)
            # print(f"[Debug] Total params: {len(param_dict)}")
            # print(f"[Debug] Param keys: {list(param_dict.keys())}")
            # for key, value in param_dict.items():
            #     print(f"[Debug] {key}: type={type(value)}")
            filtered_params = {
                key: value
                for key, value in param_dict.items()
                if key not in ignore
            }
            with open(hp_file, 'w') as f:
                json.dump(filtered_params, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"[Debug] Failed to log hyperparameters: {e}")
            print(f'[Debug] {params}')

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # Determine if it's training or validation metrics
        if "train_loss" in metrics and "train_acc" in metrics:
            # Training metrics (step-level)
            with open(self._train_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    step,
                    metrics.get("epoch", "N/A"),
                    metrics.get("train_loss", "N/A"),
                    metrics.get("train_acc", "N/A"),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "(step)"
                ])
        elif "val_loss" in metrics and "val_acc" in metrics:
            # Validation metrics (epoch-level)
            with open(self._val_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    metrics.get("epoch", "N/A"),
                    metrics.get("val_loss", "N/A"),
                    metrics.get("val_acc", "N/A"),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ])
            
    @rank_zero_only
    def save(self):
        # Nothing special to save here, as we write to CSV immediately
        pass

    @rank_zero_only
    def finalize(self, status):
        # Nothing special to finalize
        pass

class NetModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # 获取模型路径
        self.model_path = os.path.join(args['model_dir'], args['model_id'], 'pytorch_model.bin')
        self.backbone = timm.create_model(
            args['model_id'],
            pretrained=args['pretrained'],
            in_chans=args['in_channels'],
            drop_rate=0.2,
            drop_path_rate=0.2,
        )
        state_dict = torch.load(self.model_path)
        conv_stem_weight = state_dict['conv_stem.weight']
        state_dict['conv_stem.weight'] = conv_stem_weight.mean(dim=1, keepdim=True)  # [32, 3, 3, 3] -> [32, 1, 3, 3]
        self.backbone.load_state_dict(state_dict, strict=True)
        if 'efficientnet' in args['model_id']:
            backbone_out = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif 'resnet' in args['model_id']:
            backbone_out = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            backbone_out = self.backbone.get_classifier().in_features
            self.backbone.reset_classifier(0, '')
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = backbone_out
        self.classifier = nn.Linear(backbone_out, args['n_classes'])
        
        self.mixup_enabled = 'mixup_alpha' in args and args['mixup_alpha'] > 0
        if self.mixup_enabled:
            self.mixup_alpha = args['mixup_alpha']
            
    def forward(self, x, targets=None):
    
        if self.training and self.mixup_enabled and targets is not None:
            mixed_x, targets_a, targets_b, lam = self.mixup_data(x, targets)
            x = mixed_x
        else:
            targets_a, targets_b, lam = None, None, None
        
        features = self.backbone(x)
        
        if isinstance(features, dict):
            features = features['features']
            
        if len(features.shape) == 4:
            features = self.pooling(features)
            features = features.view(features.size(0), -1)
        
        logits = self.classifier(features)
        
        if self.training and self.mixup_enabled and targets is not None:
            loss = self.mixup_criterion(F.binary_cross_entropy_with_logits, 
                                       logits, targets_a, targets_b, lam)
            return logits, loss
            
        return logits
    
    def mixup_data(self, x, targets):
        """Applies mixup to the data batch"""
        batch_size = x.size(0)

        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

        indices = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[indices]
        
        return mixed_x, targets, targets[indices], lam
    
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """Applies mixup to the loss function"""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
class NetClassifier(pl.LightningModule):
    def __init__(self, model, criterion, optimizer=None, scheduler=None, n_classes=2):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self._optimizer = optimizer
        self._scheduler = scheduler
        # Metrics
        self.train_accuracy = Accuracy(task='binary', num_classes=n_classes)
        self.val_accuracy = Accuracy(task='binary', num_classes=n_classes)
        self.save_hyperparameters(ignore=['model', 'criterion'])

    def forward(self, x, targets=None):
        return self.model(x, targets) 
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")
    
    def _shared_step(self, batch, stage):
        device = self.device
        criterion = self.criterion
        
        if isinstance(batch['melspec'], list):
            batch_outputs = []
            batch_losses = []
            
            for i in range(len(batch['melspec'])):
                inputs = batch['melspec'][i].unsqueeze(0).to(device)
                target = batch['target'][i].unsqueeze(0).to(device)
                
                output = self(inputs)
                loss = criterion(output, target)
                
                batch_outputs.append(output.detach().cpu())
                batch_losses.append(loss.item())
            
            outputs = torch.cat(batch_outputs, dim=0)
            loss = np.mean(batch_losses)
            targets = batch['target']
            
        else:
            inputs = batch['melspec'].to(device)
            targets = batch['target'].to(device)
            
            outputs = self(inputs)
            
            if isinstance(outputs, tuple):
                outputs, loss = outputs  
            else:
                loss = criterion(outputs, targets)
                
            outputs = outputs.detach().cpu()
            targets = targets.detach().cpu()
        
        # # Calculate AUC
        # auc = self._calculate_auc(targets, outputs)

        # Calculate accuracy
        preds = torch.sigmoid(outputs)  # 二分类需要 sigmoid
        if stage == "train":
            self.train_accuracy(preds, targets)
            self.log(f"{stage}_acc", self.train_accuracy, prog_bar=True)
        else:
            self.val_accuracy(preds, targets)
            self.log(f"{stage}_acc", self.val_accuracy, prog_bar=True)
        
        # Log metrics
        self.log(f"{stage}_loss", loss, prog_bar=True)
        # self.log(f"{stage}_auc", auc, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        if self._optimizer is None:
            # Default optimizer if none provided
            self._optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        
        if self._scheduler is None:
            return self._optimizer
        else:
            return {
                "optimizer": self._optimizer,
                "lr_scheduler": {
                    "scheduler": self._scheduler,
                    "interval": "step" if isinstance(self._scheduler, OneCycleLR) else "epoch"
                }
            }
    
    def _calculate_auc(self, targets, outputs):
        num_classes = targets.shape[1]
        aucs = []
        
        probs = 1 / (1 + np.exp(-outputs))
        
        for i in range(num_classes):
            if np.sum(targets[:, i]) > 0:
                class_auc = roc_auc_score(targets[:, i], probs[:, i])
                aucs.append(class_auc)
        
        return np.mean(aucs) if aucs else 0.0