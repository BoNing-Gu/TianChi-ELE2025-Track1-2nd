import os
import re
import csv
import json
import re
import glob
from datetime import datetime
from pathlib import Path
import pandas as pd
import editdistance

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.strategies import DDPStrategy, FSDPStrategy

from FireRedASR.data.asr_feat import ASRFeatExtractor
from FireRedASR.models.fireredasr import FireRedAsr, load_fireredasr_aed_model
from FireRedASR.tokenizer.aed_tokenizer import ChineseCharEnglishSpmTokenizer
from sklearn.model_selection import train_test_split


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
                writer.writerow(["step", "epoch", "train_loss", "lr", "timestamp"])
        else:
            open(self._train_file, 'a', newline='').close()
        
        # Initialize validation metrics file
        if not os.path.exists(self._val_file):
            with open(self._val_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "val_asr", "timestamp"])
        else:
            open(self._val_file, 'a', newline='').close()

    @rank_zero_only
    def log_hyperparams(self, params, ignore=None):
        try:
            # Save hyperparameters to a separate JSON file
            ignore = ignore or []
            hp_file = os.path.join(self.save_dir, f"{self.name}-hparams.json")
            os.makedirs(os.path.dirname(hp_file), exist_ok=True)
            
            if isinstance(params, dict):
                param_dict = params
            else:
                param_dict = vars(params)
                
            filtered_params = {
                key: value
                for key, value in param_dict.items()
                if key not in ignore
            }
            with open(hp_file, 'w') as f:
                json.dump(filtered_params, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to log hyperparameters: {e}")

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # Determine if it's training or validation metrics
        if "train_loss" in metrics:
            # Training metrics (step-level)
            with open(self._train_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    step,
                    metrics.get("epoch", "N/A"),
                    metrics.get("train_loss", "N/A"),
                    metrics.get("lr", "N/A"),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ])
        elif "val_asr" in metrics:
            # Validation metrics (epoch-level)
            with open(self._val_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    metrics.get("epoch", "N/A"),
                    metrics.get("val_asr", "N/A"),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ])
            
    @rank_zero_only
    def save(self):
        pass

    @rank_zero_only
    def finalize(self, status):
        pass

class ASRDataset(Dataset):
    def __init__(self, df, audio_dir, feat_extractor, tokenizer):
        self.df = df
        self.audio_dir = audio_dir
        self.feat_extractor = feat_extractor
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_path = os.path.join(self.audio_dir, row['uuid'] + '.wav')
        
        # 提取音频特征
        feats, lengths, _ = self.feat_extractor([wav_path])
        feats = feats.squeeze(0)  # 去掉batch维度
        lengths = lengths.squeeze(0)
        
        # 处理文本标签
        text = row['text']
        _, token_ids = self.tokenizer.tokenize(text)
        sos_id = self.tokenizer.dict.word2id["<sos>"]
        eos_id = self.tokenizer.dict.word2id["<eos>"]   
        target = torch.tensor([sos_id] + token_ids + [eos_id], dtype=torch.long)  # 加特殊符号   
        target_length = torch.tensor(len(target), dtype=torch.long)  # 更新长度
        # target = torch.tensor(token_ids, dtype=torch.long)
        # target_length = torch.tensor(len(token_ids), dtype=torch.long)
        if (target < 0).any():
            raise ValueError(f"Found negative token id in sample {idx}, text: '{text}', token_ids: {token_ids.tolist()}")
        if (target >= len(self.tokenizer.dict.word2id)).any():
            raise ValueError(f"Found token id >= vocab size in sample {idx}, text: '{text}', token_ids: {token_ids.tolist()}")
        
        return feats, lengths, target, target_length

class ASRDataModule(pl.LightningDataModule):
    def __init__(self, args, train_file, audio_dir, feat_extractor, tokenizer, batch_size=16, num_workers=4):
        super().__init__()
        self.args = args
        self.train_file = train_file
        self.audio_dir = audio_dir
        self.feat_extractor = feat_extractor
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def setup(self, stage=None):
        # 加载数据
        train_df = pd.read_csv(self.train_file, sep='\t', encoding='utf-8')
        train_df = train_df[(train_df['audio_len'] <= 17) & (train_df['text_len'] <= 45)]
        # # 小批量测试
        # train_df = train_df.sample(n=1000, random_state=42)
        # cross valid 8:2
        train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
        
        self.train_dataset = ASRDataset(train_df, self.audio_dir, self.feat_extractor, self.tokenizer)
        self.val_dataset = ASRDataset(val_df, self.audio_dir, self.feat_extractor, self.tokenizer)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True
        )
    
    def collate_fn(self, batch):
        # 获取pad_id（<pad>=2）
        pad_id = self.tokenizer.dict.word2id["<pad>"]
        
        # 对变长序列进行padding
        feats, feat_lengths, targets, target_lengths = zip(*batch)
        
        # 特征padding（不需要pad_id，用零填充即可）
        max_feat_len = max([f.size(0) for f in feats])
        feat_dim = feats[0].size(1)
        padded_feats = torch.zeros(len(batch), max_feat_len, feat_dim)
        for i, feat in enumerate(feats):
            padded_feats[i, :feat.size(0)] = feat
        
        # 目标文本padding（使用pad_id填充）
        max_target_len = max([t.size(0) for t in targets])
        padded_targets = torch.full((len(batch), max_target_len), fill_value=pad_id).long()  # 用pad_id填充
        for i, target in enumerate(targets):
            padded_targets[i, :target.size(0)] = target
        
        return (
            padded_feats,
            torch.stack(feat_lengths),
            padded_targets,
            torch.stack(target_lengths)
        )

class FireRedASRLightning(pl.LightningModule):
    def __init__(self, model, feat_extractor, tokenizer, lr=1e-4):
        super().__init__()
        self.model = model
        self.feat_extractor = feat_extractor
        self.tokenizer = tokenizer
        self.pad_id = self.tokenizer.dict.word2id["<pad>"]
        self.sos_id = self.tokenizer.dict.word2id["<sos>"]
        self.eos_id = self.tokenizer.dict.word2id["<eos>"]
        self.lr = lr
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.pad_id)  # 假设0是padding索引
        self.validation_step_outputs = []

        self.save_hyperparameters(ignore=['model', 'feat_extractor', 'tokenizer', 'pad_id', 'criterion', 'validation_step_outputs'])
    
    def forward(self, x, x_lengths, y, y_lengths):
        return self.model(x, x_lengths, y, y_lengths)
    
    def training_step(self, batch, batch_idx):
        feats, feat_lengths, targets, target_lengths = batch              # targets: (batch_size, seq_len)
        logits, _ = self(feats, feat_lengths, targets, target_lengths)    # logits: (batch_size, seq_len, vocab_size)

        # logits_for_loss = logits.contiguous().view(-1, logits.size(-1))
        # targets_for_loss = targets.contiguous().view(-1)
        logits_for_loss = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))  # 去掉最后一个预测
        targets_for_loss = targets[:, 1:].contiguous().view(-1)  # 去掉 <sos>
        
        loss = self.criterion(logits_for_loss, targets_for_loss)
        
        lr = self.optimizers().param_groups[0]['lr']
        self.log('train_loss', loss, prog_bar=True)
        self.log('lr', lr, prog_bar=True)  # 记录学习率
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        feats, feat_lengths, targets, target_lengths = batch
        logits, _ = self(feats, feat_lengths, targets, target_lengths)
        
        logits_for_loss = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))  # 去掉最后一个预测
        targets_for_loss = targets[:, 1:].contiguous().view(-1)  # 去掉 <sos>
        
        loss = self.criterion(logits_for_loss, targets_for_loss)
        self.log('val_loss', loss, prog_bar=True)

        # ASR指标
        preds = logits.argmax(dim=-1)  # (batch_size, seq_len)
        
        total_cer = 0
        total_len = 0
        batch_size = preds.size(0)

        for i in range(batch_size):
            pred_seq = preds[i].tolist()
            target_seq = targets[i].tolist()

            # 去掉 padding
            pred_seq = [p for p in pred_seq if p != self.pad_id and p != self.sos_id and p != self.eos_id]
            target_seq = [t for t in target_seq if t != self.pad_id and t != self.sos_id and t != self.eos_id]

            if len(target_seq) == 0:
                continue  # 避免分母为0
            
            dist = editdistance.eval(pred_seq, target_seq)
            cer = dist / len(target_seq)

            total_cer += cer
            total_len += 1

        combined_output = torch.tensor([[total_cer, total_len]], device=self.device)
        return combined_output
    
    def on_validation_batch_end(self, validation_step_outputs, batch, batch_idx):
        validation_step_outputs = self.trainer.strategy.all_gather(validation_step_outputs)
        # print(f"[Debug] Rank {self.trainer.global_rank} collected {len(validation_step_outputs)} batches")
        if self.trainer.is_global_zero:
            # print(f"\n[Debug] Collected {len(validation_step_outputs)} test batches (from all GPUs)")
            # for i, batch_output in enumerate(validation_step_outputs):
            #     print(f"\n[Debug] --- Batch {i} ---")
                
            #     print(f"[Debug] batch output shape: {batch_output.shape} | dtype: {batch_output.dtype}")
            self.validation_step_outputs.append(validation_step_outputs)
        return validation_step_outputs
    
    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero:
            validation_step_outputs = [
                batch.reshape(-1, batch.shape[-1]) 
                for batch in self.validation_step_outputs
            ]
            # print(f"[Debug] after gather, len = {len(validation_step_outputs)}")
            
            validation_step_outputs = torch.cat(validation_step_outputs, dim=0)  # (total_batches, 2)
            # print(f"[Debug] after gather, shape = {validation_step_outputs.shape}")
        
            total_cer_sum = validation_step_outputs[:, 0].sum()
            total_len_sum = validation_step_outputs[:, 1].sum()

            if total_len_sum > 0:
                avg_cer = total_cer_sum / total_len_sum
                val_asr = 1.0 - avg_cer
            else:
                val_asr = torch.tensor(0.0, device=self.device)

            self.log("val_asr", val_asr, prog_bar=True)
            self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

def train(args):
    # 1. 加载预训练模型
    model_path = os.path.join(args['model_dir'], args['model_id'])
    model, model_args = load_fireredasr_aed_model(os.path.join(model_path, "model.pth.tar"))
    # 加载特征提取器和tokenizer (从预训练模型获取)
    cmvn_path = os.path.join(model_path, "cmvn.ark")
    feat_extractor = ASRFeatExtractor(cmvn_path)
    
    dict_path = os.path.join(model_path, "dict.txt")
    spm_model = os.path.join(model_path, "train_bpe1000.model")
    tokenizer = ChineseCharEnglishSpmTokenizer(dict_path, spm_model)

    # 2. 准备数据
    datamodule = ASRDataModule(
        args=args,
        train_file=args['train_file'],
        audio_dir=args['audio_dir'],
        feat_extractor=feat_extractor,
        tokenizer=tokenizer,
        batch_size=20,
        num_workers=4
    )
    
    # 3. 创建Lightning模块
    lightning_model = FireRedASRLightning(model, feat_extractor, tokenizer, lr=1e-4)
    
    # 4. 设置callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args['ckpt_dir'],
        filename='{epoch}-{val_asr:.2f}',
        monitor='val_asr',
        mode='max',
        save_top_k=1
    )
    if args['resume']:
        ckpt_dir = checkpoint_callback.dirpath
        list_of_files = glob.glob(os.path.join(ckpt_dir, '*.ckpt'))  
        if list_of_files:
            ckpt_path = max(list_of_files, key=os.path.getctime)
            print(f"Resuming from checkpoint: {ckpt_path}")
        else:
            ckpt_path = None
    else:
        ckpt_path = None
    early_stop_callback = EarlyStopping(
        monitor='val_asr',
        patience=5,
        mode='max'
    )
    
    # 5. 设置自定义logger
    custom_logger = CustomLogger(
        dirpath=args['logger_dir'],
        filename=args['name']
    )
    
    # 6. 训练器配置
    # strategy = FSDPStrategy(state_dict_type="sharded")
    # strategy = DDPStrategy(find_unused_parameters=False)
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='gpu',
        devices=[0],
        precision="32",
        logger=custom_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        enable_checkpointing=True,
    )
    
    # 7. 开始训练
    trainer.fit(lightning_model, datamodule=datamodule, ckpt_path=ckpt_path)
    
    # 8. 训练完成后处理最佳checkpoint
    best_model_path = checkpoint_callback.best_model_path
    # best_model_path = '/hy-tmp/TianChi-ELE2025-Track1-exp/checkpoints/FireRedASR/epoch=0-val_asr=0.61.ckpt'
    if best_model_path:
        print(f"Loading best model from {best_model_path}")

        stem = Path(best_model_path).stem
        filename = Path(best_model_path).name
        print(f"Best model filename: {filename}")
        
        # 生成最终文件名
        final_filename = f"{filename}.pth.tar"
        
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        checkpoint = torch.load(best_model_path, map_location="cpu")
        model_state_dict = {
            k.replace("model.", ""): v 
            for k, v in checkpoint['state_dict'].items()
        }
        package = {
            'model_state_dict': model_state_dict,
            'args': model_args
        }
        save_path = os.path.join(args['ckpt_dir'], final_filename)
        torch.save(package, save_path)
        print(f"Final model saved to {save_path}")
    else:
        print("No best model found to save")
