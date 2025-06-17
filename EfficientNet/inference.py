import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from EfficientNet.model import NetModel, NetClassifier
from EfficientNet.process import process_audio_segment
from EfficientNet.utils import get_optimizer, get_scheduler, get_criterion
import librosa


class AudioClassifier:
    def __init__(self, args):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.models = self.load_models(self.args)

    def find_model_files(self, args):
        """
        Find all .pth model files in the specified model directory
        """
        model_files = []
        dirpath = os.path.join(args["ckpt_dir"], f'{args["name"]}', f'exp_{args["version"]}')
        dirpath = Path(dirpath)
        for path in dirpath.glob('*.ckpt'):
            model_files.append(str(path))
        return model_files

    def load_models(self, args):
        """
        Load all found model files and prepare them for ensemble
        """
        models = []
        
        model_files = self.find_model_files(args)
        if not model_files:
            print(f"Warning: No model files found under {args['ckpt_dir']}!")
            return models
        print(f"Found a total of {len(model_files)} model files.")
        
        # if cfg.use_specific_folds:
        #     filtered_files = []
        #     for fold in cfg.folds:
        #         fold_files = [f for f in model_files if f"fold{fold}" in f]
        #         filtered_files.extend(fold_files)
        #     model_files = filtered_files
        #     print(f"Using {len(model_files)} model files for the specified folds ({cfg.folds}).")
        
        for model_path in model_files:
            try:
                print(f"Loading model: {model_path}")
                model = NetClassifier.load_from_checkpoint(
                    model_path, 
                    model=NetModel(args), 
                    criterion=get_criterion(args['criterion']), 
                    n_classes=args['n_classes']
                )
                model = model.to(self.device)
                model.eval()
                
                models.append(model)
            except Exception as e:
                print(f"Error loading model {model_path}: {e}")
        
        return models

    def predict_on_spectrogram(self, audio_path):
        """Process a single audio file and predict species presence for each 5-second segment"""
        uuid = Path(audio_path).stem
    
        # print(f"Processing {uuid}")
        audio_data, _ = librosa.load(audio_path, sr=self.args['FS'])
        
        if len(audio_data) < self.args['FS'] * self.args['TARGET_DURATION']:
            total_segments = 1
        else:
            total_segments = int(len(audio_data) / (self.args['FS'] * self.args['TARGET_DURATION']))

        # 分段结果聚合
        prediction_form_segment = []
        
        for segment_idx in range(total_segments):
            start_sample = int(segment_idx * self.args['FS'] * self.args['TARGET_DURATION'])
            end_sample = int(start_sample + self.args['FS'] * self.args['TARGET_DURATION'])
            segment_audio = audio_data[start_sample:end_sample]
            
            # end_time_sec = (segment_idx + 1) * self.args['TARGET_DURATION']
            # row_id = f"{uuid}_{end_time_sec}"
            # row_ids.append(row_id)

            if self.args['use_tta']:
                all_preds = []
                
                for tta_idx in range(self.args['tta_count']):
                    mel_spec = process_audio_segment(segment_audio, self.args)
                    mel_spec = apply_tta(mel_spec, tta_idx)

                    mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    mel_spec = mel_spec.to(self.device)

                    if len(self.models) == 1:
                        with torch.no_grad():
                            outputs = self.models[0](mel_spec)
                            probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
                            all_preds.append(probs)
                    else:
                        segment_preds = []
                        for model in self.models:
                            with torch.no_grad():
                                outputs = model(mel_spec)
                                probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
                                segment_preds.append(probs)
                        
                        avg_preds = np.mean(segment_preds, axis=0)
                        all_preds.append(avg_preds)

                final_probs = np.mean(all_preds, axis=0)
            else:
                mel_spec = process_audio_segment(segment_audio, self.args)  # (256,256)
                mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)   #(1,1,256,256)
                mel_spec = mel_spec.to(self.device)
                
                if len(self.models) == 1:
                    with torch.no_grad():
                        outputs = self.models[0](mel_spec)
                        final_probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
                else:
                    segment_preds = []
                    for model in self.models:
                        with torch.no_grad():
                            outputs = model(mel_spec)
                            probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
                            segment_preds.append(probs)

                    final_probs = np.mean(segment_preds, axis=0)

            # # 片段prob>threshold则记录1
            # threshold = 0.5
            # final_class = 1 if final_probs[1] > threshold else 0
            final_class = np.argmax(final_probs)
            prediction_form_segment.append(final_class)
        # 有片段为1则记录1
        if 1 in prediction_form_segment:
            prediction = 1
        else:
            prediction = 0
        
        return prediction

def apply_tta(spec, tta_idx):
    """Apply test-time augmentation"""
    if tta_idx == 0:
        # Original spectrogram
        return spec
    elif tta_idx == 1:
        # Time shift (horizontal flip)
        return np.flip(spec, axis=1)
    elif tta_idx == 2:
        # Frequency shift (vertical flip)
        return np.flip(spec, axis=0)
    else:
        return spec
