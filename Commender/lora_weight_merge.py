import os
import torch
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_model_name', '-o', 
                        type=str, 
                        choices=['solver-chooser-ds', 'solver-chooser-gemini', 'solver-chooser-gpt', 'matcher-chooser-gpt', 'gambler'] , 
                        help='model name')
    parser.add_argument('--ckpt_step', '-c', 
                        type=str, 
                        help='ckpt step')
    update_args = parser.parse_args()
    args = {
        "model_dir": "checkpoints",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "output_model_name": "solver-chooser-ds",  
        "logger_dir": "logs/Lora",
        "ckpt_dir": "checkpoints/Lora",
        "ckpt_step": '1400'
    }
    args['output_model_name'] = update_args.output_model_name
    args['ckpt_step'] = update_args.ckpt_step
    model_dir = snapshot_download(args['model_id'], cache_dir=args['model_dir'], revision='master')
    print(model_dir)

    # 模型融合
    lora_path = os.path.join(args['ckpt_dir'], args['output_model_name'], f"checkpoint-{args['ckpt_step']}")

    # 加载tokenizer, 模型和lora权重
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
    model = PeftModel.from_pretrained(model, model_id=lora_path)

    # merge
    save_path = os.path.join(args['ckpt_dir'], args['output_model_name'])
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)