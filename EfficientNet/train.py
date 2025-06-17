import os
import warnings
from pathlib import Path
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from EfficientNet.data_module import AudioDatasetFromNPY, LightDataModule
from EfficientNet.model import NetModel, NetClassifier, CustomLogger
from EfficientNet.utils import get_optimizer, get_scheduler, get_criterion

warnings.filterwarnings("ignore")



def train(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    df = pd.read_csv(args['train_file'], sep='\t', header=0)
    args['doms'] = df['dom'].unique().tolist()

    spectrograms = None
    print("Loading pre-computed mel spectrograms from NPY file...")
    spectrograms = np.load(args['processed_audio_dir'], allow_pickle=True).item()
    print(f"Loaded {len(spectrograms)} pre-computed mel spectrograms")
        
    kfold = KFold(args['n_split'], shuffle = True, random_state = args['SEED'])
    
    best_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tqdm(kfold.split(df))):
        print(f'\n{"="*30} Fold {fold} {"="*30}')
        
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        
        print(f'Training set: {len(train_df)} samples')
        print(f'Validation set: {len(val_df)} samples')
        
        # Create DataModule
        train_dataset = AudioDatasetFromNPY(train_df, args, spectrograms=spectrograms, mode='train')
        val_dataset = AudioDatasetFromNPY(val_df, args, spectrograms=spectrograms, mode='valid')
        dm = LightDataModule(train_dataset, val_dataset, args['batch_size'], args['eval_batch_size'])
        train_dataloader = dm.train_dataloader()
        valid_dataloader = dm.valid_dataloader()
        
        # Load Model
        model = NetModel(args)
        optimizer = get_optimizer(model, choice=args['optimizer'], learing_rate=args['LR'])
        criterion = get_criterion(choice=args['criterion'])
        scheduler = get_scheduler(optimizer, choice=args['scheduler'], learing_rate=args['LR'], steps_per_epoch=len(train_dataloader), n_epochs=args['n_epochs'])
        
        # Define LightningModule
        lightning_model = NetClassifier(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            n_classes=args['n_classes']
        )

        # Define callbacks
        csv_logger = CustomLogger(
            dirpath=os.path.join(args["logger_dir"], f'{args["name"]}', f'exp_{args["version"]}'), 
            filename=f'{fold}'
        )
        checkpoint_callback = ModelCheckpoint(
            monitor='val_acc',
            mode='max',
            save_top_k=1,
            dirpath=os.path.join(args["ckpt_dir"], f'{args["name"]}', f'exp_{args["version"]}'),
            filename=f'{fold}' + '-{epoch:02d}-{val_acc:.2f}'
        )
        early_stopping = EarlyStopping(
            monitor='val_acc',
            patience=5,
            mode='max'
        )

        # Initialize trainer
        trainer = Trainer(
            max_epochs=args['n_epochs'],
            accelerator='auto',
            devices=1,
            enable_progress_bar=True,
            enable_model_summary=True,
            logger=csv_logger,
            log_every_n_steps=10,
            enable_checkpointing=True,
            callbacks=[checkpoint_callback, early_stopping],
        )

        # Train the model
        trainer.fit(
            lightning_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=valid_dataloader
        )

        best_acc = checkpoint_callback.best_model_score.item()
        best_scores.append(best_acc)
        print(f'Fold {fold} - Best val_acc: {best_acc:.4f}')
    
    print("\n" + "="*60)
    print("Cross-Validation Results:")
    for fold, score in enumerate(best_scores):
        print(f"Fold {fold}: {score:.4f}")
    print(f"Mean ACC: {np.mean(best_scores):.4f}")
    print("="*60)