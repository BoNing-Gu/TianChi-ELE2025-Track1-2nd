from typing import Optional
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class TextClassificationDataModule(LightningDataModule):
    def __init__(self, train_df=None, valid_df=None, tokenizer=None, 
                 batch_size=16, eval_batch_size=32, max_length=128):
        super().__init__()
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.train_dataset = TextDataset(
            texts=train_df['text'].values,
            labels=train_df['label'].values,
            tokenizer=tokenizer,
            max_length=max_length
        )
        self.valid_dataset = TextDataset(
            texts=valid_df['text'].values,
            labels=valid_df['label'].values,
            tokenizer=tokenizer,
            max_length=max_length
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=True
        )

    def valid_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=False, 
        )