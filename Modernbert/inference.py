import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from Modernbert.model import BertClassifier
from Modernbert.utils import get_optimizer, get_scheduler, get_criterion
import os

class TextClassifier:
    def __init__(self, args):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.model_path = os.path.join(args["model_dir"], args["model_id"])
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False, trust_remote_code=True)
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
        filtered_files = []
        # for fold in cfg.folds:
        for fold in ["0", "1", "2"]:
            fold_files = [f for f in model_files if f.split("/")[-1].startswith(f"{fold}")]
            filtered_files.extend(fold_files)
        model_files = filtered_files
        print(f"Using {len(model_files)} model files for the specified folds ([0, 1, 2]).")
        
        for model_path in model_files:
            try:
                print(f"Loading model: {model_path}")
                model = BertClassifier.load_from_checkpoint(
                    model_path, 
                    model=AutoModelForSequenceClassification.from_pretrained(self.model_path, num_labels=args["n_classes"]), 
                    criterion=get_criterion(args['criterion']), 
                    n_classes=args['n_classes']
                )
                model = model.to(self.device)
                model.eval()
                
                models.append(model)
            except Exception as e:
                print(f"Error loading model {model_path}: {e}")
        
        return models

    def preprocess_text(self, text, max_length=64):
        # Tokenize the input text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device)
        }

    @torch.no_grad()
    def predict(self, text):
        # Preprocess the input text
        inputs = self.preprocess_text(text)
        
        # Get model predictions
        preds = []
        for model in self.models:
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs, dim=1).cpu().numpy().squeeze()
                preds.append(probs)
        final_probs = np.mean(preds, axis=0)
        final_class = np.argmax(final_probs)
        
        return {
            'prediction': final_class,
            'probabilities': final_probs
        }

# Example usage
if __name__ == '__main__':
    # Initialize the classifier
    classifier = TextClassifier()
    
    # Example text for prediction
    text = "我想吃香骨鸡"
    
    # Get prediction
    result = classifier.predict(text)
    print(f"Prediction: {result['prediction']}")
    print(f"Probabilities: {result['probabilities']}")