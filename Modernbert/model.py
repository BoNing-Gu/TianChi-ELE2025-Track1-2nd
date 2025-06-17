import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy
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

class BertClassifier(LightningModule):
    def __init__(self, model, criterion, optimizer=None, scheduler=None, n_classes=2):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self._optimizer = optimizer
        self._scheduler = scheduler
        # Metrics
        self.train_accuracy = Accuracy(task='binary', num_classes=n_classes)
        self.val_accuracy = Accuracy(task='binary', num_classes=n_classes)
        
        self.save_hyperparameters(ignore=["model"])

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits
    
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

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)
        
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, labels)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_accuracy, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy(preds, labels)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_accuracy, prog_bar=True)

    # def save_model(self, path):
    #     # Save the model checkpoint
    #     self.trainer.save_checkpoint(f"{path}.ckpt")
    #     # Save the BERT model configuration
    #     self.model.config.save_pretrained(path)