from EfficientNet.process import process_audio
from EfficientNet.train import train
from EfficientNet.inference import NetModel, AudioClassifier
import pytorch_lightning as pl
import pandas as pd
from tqdm.auto import tqdm
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EfficientNet Audio Classifier")
    parser.add_argument("--step", "-s", 
                        type=str, 
                        choices=["Process", "Train", "Inference"], 
                        help="step to run")
    parser.add_argument("--dataset", "-d", 
                        type=str, 
                        default="B", 
                        choices=['T', 'A', 'B'],
                        help="dataset")
    update_args = parser.parse_args()
    args = {
        # Train args
        "model_id": "efficientnet_b0",
        "model_dir": "checkpoints",
        "pretrained": False,   
        "in_channels": 1,   # 灰度图
        "train_file": "data/智慧养老_label_A/train.txt",
        "test_file": "data/智慧养老_label_{}/{}.txt",
        "audio_dir": "data/train_audio",  
        "processed_audio_dir": "data/processed_train_audio/all_data_256,256.npy",
        "test_audio_dir": "data/{}_audio",
        "result_dir": "results_{}/EfficientNet",
        "logger_dir": "logs/EfficientNet",
        "ckpt_dir": "checkpoints/EfficientNet",
        "name": "5fold",
        "version": "0",
        'n_split': 5,
        "augment_probability": 0.5,
        "mixup_alpha": 0.5,
        "batch_size": 128,
        "eval_batch_size": 128,
        "n_epochs": 5,
        "n_classes": 2,
        "optimizer": "AdamW",
        "criterion": "BCEWithLogitsLoss",
        "scheduler": "CosineAnnealingLR",
        "LR": 5e-4,
        # Data args
        "FS": 16000,                 # 采样率（sampling rate）
        "TARGET_DURATION": 7.0,      # 目标时长（秒）
        "TARGET_SHAPE": (256, 256),  # 目标图像尺寸
        "N_FFT": 512,          # FFT窗口大小（影响频率分辨率）
        "HOP_LENGTH": 160,     # 帧移（影响时间分辨率）
        "N_MELS": 128,         # Mel滤波器数量（决定输出维度）
        "FMIN": 50,            # 最小频率
        "FMAX": 7500,          # 最大频率
        # Inference args
        "use_tta": False,
        "tta_count": 0,
        # Control args
        "SEED": 42,
        "Debug": False
    }
    args['test_file'] = args['test_file'].format(update_args.dataset, update_args.dataset)
    args['test_audio_dir'] = args['test_audio_dir'].format(update_args.dataset)
    args['result_dir'] = args['result_dir'].format(update_args.dataset)
    pl.seed_everything(args['SEED'])

    if update_args.step == "Process":
        process_audio(args)
    elif update_args.step == "Train":
        # Train the model
        train(args)
    elif update_args.step == "Inference":
        result_df = []
        classifier = AudioClassifier(args)
        test_df = pd.read_csv(args['test_file'], sep='\t', header=0)
        test_df['filepath'] = args['test_audio_dir'] + '/' + test_df.uuid + '.wav'
        total_samples = len(test_df)
        for i, row in tqdm(test_df.iterrows(), total=total_samples):
            audio_path = row['filepath']
            prediction = classifier.predict_on_spectrogram(str(audio_path))
            result_df.append([row['uuid'], prediction])

        result_df = pd.DataFrame(result_df, columns=['uuid', 'dom'])
        os.makedirs(args['result_dir'], exist_ok=True)
        result_df.to_csv(os.path.join(args['result_dir'], 'dom.csv'), index=False, encoding='utf-8')