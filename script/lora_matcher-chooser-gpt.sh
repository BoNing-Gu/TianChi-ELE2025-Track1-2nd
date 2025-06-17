#!/bin/bash

# 限制 CPU 使用（可选）：
# 方式 1：设置 OpenMP/NumPy 等线程数（适合 Python）
export OMP_NUM_THREADS=26
export MKL_NUM_THREADS=26

# 方式 2（可选）：直接限制 CPU 核心（例如用 0~3 核）
# taskset -c 0-3 python ...   ← 用这种方式直接运行程序

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda init
conda activate lit

# 运行 Python 脚本
python lora_finetune.py -o matcher-chooser-gpt --per_device_train_batch_size 8 --gradient_accumulation_steps 4
