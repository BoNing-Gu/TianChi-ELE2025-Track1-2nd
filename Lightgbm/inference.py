import os
import joblib
import numpy as np
from pathlib import Path
from typing import List, Dict, Union
from text2vec import SentenceModel, cos_sim, semantic_search, BM25

class TextClassifier:
    def __init__(self, args: Dict):
        self.args = args
        self.embedder = SentenceModel(os.path.join(args['embmodel_dir'], args['embmodel_id']))
        self.models = self.load_models(args)
        
    def find_model_files(self, args: Dict) -> List[str]:
        """
        Find all model files (.pkl or .txt) in the specified model directory
        """
        model_files = []
        dirpath = Path(args["ckpt_dir"]) / args["name"] / f'exp_{args["version"]}'
        
        # Support both .pkl (sklearn) and .txt (native lightgbm) formats
        for ext in ['*.pkl', '*.txt']:
            model_files.extend([str(p) for p in dirpath.glob(ext)])
        
        return sorted(model_files)  # Sort for reproducibility

    def load_models(self, args: Dict) -> List:
        """
        Load all found model files and prepare them for ensemble
        """
        models = []
        model_files = self.find_model_files(args)
        
        if not model_files:
            raise FileNotFoundError(f"No model files found under!")
        print(f"Found {len(model_files)} model files for ensemble.")
        
        for model_path in model_files:
            try:
                print(f"Loading model: {os.path.basename(model_path)}")
                
                # Handle both sklearn (.pkl) and native lightgbm (.txt) formats
                if model_path.endswith('.pkl'):
                    model = joblib.load(model_path)
                else:  # .txt format
                    import lightgbm as lgb
                    model = lgb.Booster(model_file=model_path)
                
                models.append(model)
            except Exception as e:
                print(f"Error loading {model_path}: {str(e)}")
                continue
                
        if not models:
            raise ValueError("No models were successfully loaded!")
        return models
    
    def text_to_embedding(self, text: str) -> np.ndarray:
        """Convert text to embedding vector"""
        return self.embedder.encode(text).reshape(1, -1)
    
    def predict_proba(self, text: Union[str, np.ndarray]) -> np.ndarray:
        """
        Get probability predictions from all models (soft voting)
        
        Args:
            text: Input text or pre-computed embedding
            
        Returns:
            Averaged probabilities across all models [shape: (n_classes,)]
        """
        if isinstance(text, str):
            embedding = self.text_to_embedding(text)
        else:
            embedding = text
            
        all_probs = []
        for model in self.models:
            try:
                # Handle both sklearn and native lightgbm interfaces
                if hasattr(model, 'predict_proba'):  # sklearn-style
                    probs = model.predict_proba(embedding)[0]
                else:  # native lightgbm
                    probs = model.predict(embedding)[0]
                    if probs.ndim == 1:  # binary classification
                        probs = np.array([1-probs, probs])
                
                all_probs.append(probs)
            except Exception as e:
                print(f"Prediction error with model {model}: {str(e)}")
                continue
                
        if not all_probs:
            raise RuntimeError("All models failed during prediction!")
            
        return np.mean(all_probs, axis=0)  # Average probabilities
    
    def predict(self, text: Union[str, np.ndarray]) -> int:
        """
        Get final class prediction (hard vote from soft probabilities)
        
        Returns:
            Predicted class label (0 or 1 for binary classification)
        """
        probs = self.predict_proba(text)
        return np.argmax(probs)
    
    def predict_on_text(self, text: str) -> Dict:
        """Predict on a single text input with confidence"""
        probs = self.predict_proba(text)
        return {
            'prediction': int(np.argmax(probs)),
            'confidence': float(np.max(probs)),
            'probabilities': probs.tolist()
        }