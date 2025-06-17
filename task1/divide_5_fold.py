from sklearn.model_selection import KFold
import pandas as pd
from tqdm import tqdm
import os

# 参数配置
n_split = 5  # 折数
SEED = 42  # 随机种子
train_txt_path = "./data/智慧养老_label/train_with_source_len.txt"  # 原始数据集路径
output_folder = "./folds/"  # 输出文件夹路径

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 1. 读取数据集，跳过第一行但保留标题
data = pd.read_csv(train_txt_path, sep="\t", header=0)  # 读取数据，默认第一行为标题

# 2. 初始化 K-Fold
kfold = KFold(n_splits=n_split, shuffle=True, random_state=SEED)

# 3. 遍历每折数据并保存为 TXT 文件
for fold, (train_idx, val_idx) in enumerate(tqdm(kfold.split(data), desc="Processing folds")):
    # 获取当前折的训练集和验证集
    train_df = data.iloc[train_idx]  # 按索引提取训练集
    val_df = data.iloc[val_idx]  # 按索引提取验证集

    # 保存训练集和验证集为 TXT 文件
    train_txt_path = os.path.join(output_folder, f"train_fold_{fold}.txt")
    val_txt_path = os.path.join(output_folder, f"val_fold_{fold}.txt")

    # 写入文件，保留标题行
    train_df.to_csv(train_txt_path, sep="\t", index=False, header=True)
    val_df.to_csv(val_txt_path, sep="\t", index=False, header=True)

    print(f"Fold {fold} saved: {train_txt_path}, {val_txt_path}")

print(f"All folds saved in folder: {output_folder}")