import os
import cv2
import math
import time
import librosa
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import torch
import warnings
warnings.filterwarnings("ignore")

def audio2melspec(audio_data, args):
    if np.isnan(audio_data).any():
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal)

    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=args['FS'],
        n_fft=args['N_FFT'],
        hop_length=args['HOP_LENGTH'],
        n_mels=args['N_MELS'],
        fmin=args['FMIN'],
        fmax=args['FMAX'],
        power=2.0   # 输出功率谱
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    
    return mel_spec_norm

def process_audio_segment(audio_data, args):
    """Process audio segment to get mel spectrogram"""
    if len(audio_data) < args['FS'] * args['TARGET_DURATION']:
        audio_data = np.pad(audio_data, 
                          (0, int(args['FS'] * args['TARGET_DURATION'] - len(audio_data))), 
                          mode='constant')
    
    mel_spec = audio2melspec(audio_data, args)
    
    # Resize if needed
    if mel_spec.shape != args['TARGET_SHAPE']:
        mel_spec = cv2.resize(mel_spec, args['TARGET_SHAPE'], interpolation=cv2.INTER_LINEAR)
        
    return mel_spec.astype(np.float32)

def process_audio(args):
        
    print(f"Debug mode: {'ON' if {args['debug_mode']} else 'OFF'}")
    print(f"Process audio files: {args['audio_dir']}")

    print("Loading training metadata...")
    train_df = pd.read_csv(args['train_file'], sep='\t', header=0)

    label_list = sorted(train_df['dom'].unique())
    label_id_list = list(range(len(label_list)))
    label2id = dict(zip(label_list, label_id_list))
    id2label = dict(zip(label_id_list, label_list))

    print(f'Found {len(label_list)} unique classes')
    working_df = train_df[['uuid', 'dom']].copy()
    working_df['target'] = working_df.dom.map(label2id)
    working_df['filepath'] = args['audio_dir'] + '/' + working_df.uuid + '.wav'

    if args['debug_mode']:
        # 列出音频目录中的所有文件
        audio_files = [f for f in os.listdir(args['audio_dir']) if os.path.isfile(os.path.join(args['audio_dir'], f))]
        # 筛选出 working_df 中 filename 列在 audio_files 列表中的数据
        working_df = working_df[working_df['uuid'].apply(lambda x: f"{x}.wav" in audio_files)]
        
    total_samples = len(working_df)
    print(f'Total samples to process: {total_samples} ')
    print(f'Samples by class:')
    print(working_df['dom'].value_counts())

    print("Starting audio processing...")
    start_time = time.time()

    all_data = {}
    errors = []

    for i, row in tqdm(working_df.iterrows(), total=total_samples):
        try:
            audio_data, _ = librosa.load(row.filepath, sr=args['FS'])

            target_samples = int(args['TARGET_DURATION'] * args['FS'])

            if len(audio_data) < target_samples:
                n_copy = math.ceil(target_samples / len(audio_data))
                if n_copy > 1:
                    audio_data = np.concatenate([audio_data] * n_copy)

            start_idx = max(0, int(len(audio_data) / 2 - target_samples / 2))
            end_idx = min(len(audio_data), start_idx + target_samples)
            center_audio = audio_data[start_idx:end_idx]

            if len(center_audio) < target_samples:
                center_audio = np.pad(center_audio, 
                                    (0, target_samples - len(center_audio)), 
                                    mode='constant')

            mel_spec = audio2melspec(center_audio, args)

            if mel_spec.shape != args['TARGET_SHAPE']:
                mel_spec = cv2.resize(mel_spec, args['TARGET_SHAPE'], interpolation=cv2.INTER_LINEAR)

            all_data[row.uuid] = mel_spec.astype(np.float32)
            
        except Exception as e:
            print(f"Error processing {row.filepath}: {e}")
            errors.append((row.filepath, str(e)))

    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds")
    print(f"Successfully processed {len(all_data)} files out of {total_samples} total")
    print(f"Failed to process {len(errors)} files")

    os.makedirs(args['processed_audio_dir'], exist_ok=True)
    shape_str = f"{args['TARGET_SHAPE'][0]},{args['TARGET_SHAPE'][1]}"  # '256,256'
    output_filepath = os.path.join(args['processed_audio_dir'], f'all_data_{shape_str}.npy')
    np.save(output_filepath, all_data)
    print(f"✅ All data saved to {output_filepath}")

if __name__ == '__main__':
    args = {
        "debug_mode": False,
        "train_file": "data/智慧养老_label/train.txt",
        "audio_dir": "data/train_audio",
        "processed_audio_dir": "data/processed_train_audio",
        "FS": 16000,                 # 采样率（sampling rate）
        "TARGET_DURATION": 7.0,      # 目标时长（秒）
        "TARGET_SHAPE": (256, 256),  # 目标图像尺寸
        "N_FFT": 512,          # FFT窗口大小（影响频率分辨率）
        "HOP_LENGTH": 160,     # 帧移（影响时间分辨率）
        "N_MELS": 128,         # Mel滤波器数量（决定输出维度）
        "FMIN": 50,            # 最小频率
        "FMAX": 7500,          # 最大频率
    }
    process_audio(args)