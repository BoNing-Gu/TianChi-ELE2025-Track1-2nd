import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
from peft import LoraConfig, TaskType, get_peft_model
import argparse
import glob
from functools import partial

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

def process_func(example, modeltype):
    input_ids, attention_mask, labels = [], [], []
    
    if modeltype == 'instruct':
        instruction = tokenizer(
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n", 
            add_special_tokens=False   # add_special_tokens 不在开头加 special_tokens
        )
        response = tokenizer(
            f"{example['output']}", 
            add_special_tokens=False
        )
        input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1] 
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
        if len(input_ids) > MAX_LENGTH: 
            print("input_ids length > MAX_LENGTH")
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    elif modeltype == 'reasoning':
        instruction = tokenizer(
            f"<｜User｜>\nInstruction:\n{system_prompt}\nQuestion:\n{example['instruction'] + example['input']}\n<think>\n<｜Assistant｜>\n", 
            add_special_tokens=False   # add_special_tokens 不在开头加 special_tokens
        )
        response_thinking = tokenizer(
            f"{example['reasoning']}\n</think>\n", 
            add_special_tokens=False
        )
        if len(response_thinking["input_ids"]) > MAX_REASONING_LEN:
            print("response_thinking length > MAX_REASONING_LEN")
            cropped_input_ids = response_thinking["input_ids"][:MAX_REASONING_LEN]
            cropped_text = tokenizer.decode(cropped_input_ids, skip_special_tokens=True)
            response_thinking = tokenizer(
                f"{cropped_text}\n</think>\n", 
                add_special_tokens=False
            )
        response = tokenizer(
            f"{example['output']}", 
            add_special_tokens=False
        )

        input_ids = instruction["input_ids"] + response_thinking["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response_thinking["attention_mask"] + response["attention_mask"] + [1] 
        labels = [-100] * len(instruction["input_ids"]) + response_thinking["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]  
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_model_name', '-o', 
                        type=str, 
                        choices=['solver-chooser-ds', 'solver-chooser-gemini', 'solver-chooser-gpt', 'matcher-chooser-gpt', 'gambler', 'end-2-end', 'end-2-end-large'], 
                        help='model')
    parser.add_argument('--method', 
                        type=str, 
                        choices=['Lora', 'LP', 'Fully'],
                        help='finetune method')
    parser.add_argument('--n_epoch', 
                        default=3,
                        type=int)
    parser.add_argument('--per_device_train_batch_size', 
                        default=8,
                        type=int)
    parser.add_argument('--gradient_accumulation_steps', 
                        default=4,
                        type=int)
    update_args = parser.parse_args()
    args = {
        # Train args
        "model_dir": "checkpoints",
        "model_id": "Qwen/Qwen3-0.6B",
        "train_ds_dir": "data/dataset_{}/",
        "output_model_name": "end-2-end", 
        "logger_dir": "logs",
        "ckpt_dir": "checkpoints",
    }
    args['output_model_name'] = update_args.output_model_name
    args['n_epoch'] = update_args.n_epoch
    args['per_device_train_batch_size'] = update_args.per_device_train_batch_size
    args['gradient_accumulation_steps'] = update_args.gradient_accumulation_steps
    args['method'] = update_args.method
    args['logger_dir'] = os.path.join(args['logger_dir'], args['method'])
    args['ckpt_dir'] = os.path.join(args['ckpt_dir'], args['method'])

    # set model, system_prompt, dataset 
    if args['output_model_name'] in ['solver-chooser-ds', 'solver-chooser-gemini', 'solver-chooser-gpt', 'matcher-chooser-gpt']:
        args['model_id'] = "Qwen/Qwen2.5-7B-Instruct"
        MAX_LENGTH = 2048
        modeltype = 'instruct'
        with open('src/system_prompt_lora_commender.txt', 'r') as f:
            system_prompt = f.read()
    elif args['output_model_name'] in ['gambler']:
        args['model_id'] = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        MAX_REASONING_LEN = 3072
        modeltype = 'reasoning'
        with open('src/system_prompt_lora_gambler.txt', 'r') as f:
            system_prompt = f.read()
    elif args['output_model_name'] in ['end-2-end']:
        args['model_id'] = "Qwen/Qwen3-0.6B"
        MAX_LENGTH = 256
        modeltype = 'instruct'
        with open('src/system_prompt_lora_end2end.txt', 'r') as f:
            system_prompt = f.read()
    elif args['output_model_name'] in ['end-2-end-large']:
        args['model_id'] = "Qwen/Qwen3-8B"
        MAX_LENGTH = 1024
        modeltype = 'instruct'
        with open('src/system_prompt_lora_end2end.txt', 'r') as f:
            system_prompt = f.read()
    print(system_prompt)
    if args['output_model_name'] not in ['end-2-end', 'end-2-end-large']:
        dfs = []
        for dataset in ['A', 'B']:
            train_ds_dir = os.path.join(args['train_ds_dir'].format(dataset), args['output_model_name'])
            for filename in os.listdir(train_ds_dir):
                if filename.endswith(".json"):
                    print(filename)
                    df = pd.read_json(os.path.join(train_ds_dir, filename))
                    print(df.shape)
                    dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        print(df.shape)
        ds = Dataset.from_pandas(df)
        print(ds)
        ds[:3]
    else:
        dfs = []
        train_ds_dir = os.path.join(args['train_ds_dir'].format('F'), args['output_model_name'])
        for filename in os.listdir(train_ds_dir):
            if filename.endswith(".json"):
                print(filename)
                df = pd.read_json(os.path.join(train_ds_dir, filename))
                print(df.shape)
                dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        print(df.shape)
        ds = Dataset.from_pandas(df)
        print(ds)
        ds[:3]

    # 加载模型地址    
    model_dir = os.path.join(args['model_dir'], args['model_id'])
    print(model_dir)

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
    tokenizer

    # 计算 token 数
    tokens = tokenizer(system_prompt, return_tensors="pt")["input_ids"]
    token_count = tokens.shape[1]
    print(f"Token 数: {token_count}")

    tokenized_id = ds.map(partial(process_func, modeltype=modeltype), remove_columns=ds.column_names)
    tokenized_id
    tokenizer.decode(tokenized_id[0]['input_ids'])
    tokenizer.decode(list(filter(lambda x: x != -100, tokenized_id[1]["labels"])))

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.float32)
    model
    model.enable_input_require_grads() # 开启梯度检查点时，要执行该方法
    model.dtype

    if args['method'] == 'Lora':
        # 加载 LoRA
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            inference_mode=False, # 训练模式
            r=16, # Lora 秩
            lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
            lora_dropout=0.1# Dropout 比例
        )
        config
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
    elif args['method'] == 'LP':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.lm_head.parameters():
            param.requires_grad = True
        model.train()
        print_trainable_parameters(model)
    else:
        model.train()
        print_trainable_parameters(model)

    # 配置训练参数
    train_args = TrainingArguments(
        output_dir=os.path.join(args['ckpt_dir'], args['output_model_name']),
        logging_dir=os.path.join(args['logger_dir'], args['output_model_name']),
        # overwrite_output_dir=True,
        per_device_train_batch_size=args['per_device_train_batch_size'],
        gradient_accumulation_steps=args['gradient_accumulation_steps'],
        logging_steps=10,
        num_train_epochs=args['n_epoch'],
        save_steps=250, 
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        dataloader_num_workers=8,
    )

    # 训练器
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    # 训练
    trainer.train()