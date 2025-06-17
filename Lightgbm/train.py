import os
import warnings
from pathlib import Path
from tqdm.auto import tqdm
tqdm.pandas()
import joblib

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from lightgbm import LGBMClassifier
from text2vec import SentenceModel, cos_sim, semantic_search, BM25

warnings.filterwarnings("ignore")


def train(args):
    embedder = SentenceModel(os.path.join(args['model_dir'], args['embmodel_id']))

    df = pd.read_csv(args['train_file'], encoding='utf-8', sep='\t', header=0)
    args['doms'] = df['dom'].unique().tolist()
    df.rename(columns={'dom': 'label'}, inplace=True)
    df = df[['text', 'label']]
    print(f"Loaded {len(df)} samples from {args['train_file']}")
    if args['Process']:
        df['embedding'] = df['text'].progress_apply(lambda x: embedder.encode(x))
        os.makedirs(args['ckpt_dir'], exist_ok=True)
        df.to_pickle(os.path.join(args['ckpt_dir'], 'train.pkl'))
    else:
        df = pd.read_pickle(os.path.join(args['ckpt_dir'], 'train.pkl'))

    kfold = KFold(args['n_split'], shuffle=True, random_state=args['SEED'])
    
    best_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tqdm(kfold.split(df))):
        print(f'\n{"="*30} Fold {fold} {"="*30}')
        
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        
        print(f'Training set: {len(train_df)} samples')
        print(f'Validation set: {len(val_df)} samples')

        # Prepare data
        X_train = np.stack(train_df['embedding'].values)
        y_train = train_df['label'].values
        X_val = np.stack(val_df['embedding'].values)
        y_val = val_df['label'].values

        # Initialize LGBMClassifier with early stopping
        model = LGBMClassifier(
            objective='binary',
            n_estimators=args['n_estimators'],  
            learning_rate=args['LR'],
            num_leaves=31,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            bagging_freq=5,
            random_state=args['SEED'],
            n_jobs=-1  # 使用所有可用的CPU核心
        )

        # Fit model with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='binary_error',  # 使用错误率(1-accuracy)
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(10)],
        )

        # Make predictions
        y_pred = model.predict(X_val)
        best_acc = accuracy_score(y_val, y_pred)
        n_estimators = model.best_iteration_

        # Save best model
        model_save_dir = os.path.join(args['ckpt_dir'], args['name'], args['version'])
        os.makedirs(model_save_dir, exist_ok=True)
        model_save_path = os.path.join(model_save_dir, f"{fold}-n_estimators{n_estimators:02d}-val_acc{best_acc:.2f}.pkl")
        joblib.dump(model, model_save_path)
        print(f"Saved model for fold {fold} to {model_save_path}")

        best_scores.append(best_acc)
        print(f'Fold {fold} - Best val_acc: {best_acc:.4f}')
        print(f'Best iteration: {n_estimators}')
    
    print("\n" + "="*60)
    print("Cross-Validation Results:")
    for fold, score in enumerate(best_scores):
        print(f"Fold {fold}: {score:.4f}")
    print(f"Mean ACC: {np.mean(best_scores):.4f}")
    print("="*60)