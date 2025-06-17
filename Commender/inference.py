# vllm_model.py
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import torch
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import json
import tqdm
import re
import pandas as pd
import argparse
from tqdm.auto import tqdm

def extract_json(text):
    # 去除 markdown 标记和 LaTeX 字符
    text = text.strip().strip("`").strip()
    text = text.replace("\\_", "_")

    # 尝试提取一个合法 JSON 字符串（数组或对象）
    candidates = re.findall(r'(\[.*\]|\{.*\})', text, re.DOTALL)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            return parsed  # 直接返回已解析对象
        except json.JSONDecodeError:
            continue

    raise ValueError("No valid JSON found in text.")

class FoodCommender:
    temperature = 0.6
    top_p = 0.95
    top_k = 20
    min_p = 0
    def __init__(self, args):
        # Set system prompt and arguments
        args['ckpt_dir'] = os.path.join(args['ckpt_dir'], args['method'])
        if args['output_model_name'] in ['solver-chooser-ds', 'solver-chooser-gemini', 'solver-chooser-gpt', 'matcher-chooser-gpt']:
            self.max_model_len = 2048
            self.max_tokens = 1024
            with open('src/system_prompt_lora_commender.txt', 'r') as f:
                self.system_prompt = f.read()
            model_dir = os.path.join(args['model_dir'], args['model_id'])
            if args['method'] == 'Lora':
                self.lora_path = os.path.join(args['ckpt_dir'], args['output_model_name'], f"checkpoint-{args['ckpt_step']}")
        elif args['output_model_name'] in ['end-2-end', 'end-2-end-large']:
            self.max_model_len = 256
            self.max_tokens = 48
            self.temperature = 0.3
            self.top_p = 0.8
            with open('src/system_prompt_lora_end2end.txt', 'r') as f:
                self.system_prompt = f.read()
            if args['method'] == 'Fully':
                model_dir = os.path.join(args['ckpt_dir'], args['output_model_name'], f"checkpoint-{args['ckpt_step']}")
            elif args['method'] == 'NoFT':
                model_dir = os.path.join(args['model_dir'], args['model_id'])
        else:
            raise ValueError(f"Unsupported model name: {args['output_model_name']}")
        
        # 初始化 vLLM 推理引擎
        print(f'Loading model from {model_dir}...')
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.llm = self.prepare_model(model_dir, enable_lora=True if args['method'] == 'Lora' else False)
 
    def prepare_model(self, model_path, enable_lora=True):
        # 初始化 vLLM 推理引擎
        llm = LLM(
            model=model_path, 
            tokenizer=model_path, 
            max_model_len=self.max_model_len, 
            trust_remote_code=True, 
            enable_lora=enable_lora, 
            max_lora_rank=32,
            disable_log_stats=True
        )
        return llm
    
    def get_completion(self, prompts):
        stop_token_ids = [151645]
        # 创建采样参数。temperature 控制生成文本的多样性，top_p 控制核心采样的概率
        sampling_params = SamplingParams(
            temperature=self.temperature, 
            max_tokens=self.max_tokens, 
            top_p=self.top_p, 
            top_k=self.top_k,
            min_p=self.min_p,
            stop_token_ids=stop_token_ids
        )
        # 初始化 vLLM 推理引擎
        outputs = self.llm.generate(
            prompts, 
            sampling_params, 
            use_tqdm=False, 
            lora_request=LoRARequest("adapter_name", 1, self.lora_path) if hasattr(self, 'lora_path') else None
        )
        return outputs

    def commend(self, text):
        text = text.replace('天猫精灵', '')
        text = text.strip('，。？！：；、')
        message = [
            {"role": "system", "content": self.system_prompt}, 
            {"role": "user", "content": text}
        ]
        inputs = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        outputs = self.get_completion(inputs)
        response = outputs[0].outputs[0].text
        try:
            # 尝试读取为json
            response_json = extract_json(response)
            result = {
                'text': text,
                'content': response,
                'parsed_content': response_json
            }
        except (json.JSONDecodeError, ValueError) as e:
            result = {
                'text': text,
                'content': response,
                'parsed_content': '无法解析'
            }
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_model_name', '-o', 
                        type=str, 
                        choices=['solver-chooser-ds', 'solver-chooser-gemini', 'solver-chooser-gpt', 'matcher-chooser-gpt', 'gambler', 'end-2-end', 'end-2-end-large'], 
                        help='model')
    parser.add_argument('--method', 
                        type=str, 
                        choices=['NoFT', 'Lora', 'LP', 'Fully'],
                        help='finetune method')
    parser.add_argument('--ckpt_step', '-c', 
                        default='1400',
                        type=str, 
                        help='ckpt step')
    parser.add_argument("--dataset", "-d", 
                        type=str, 
                        default="B", 
                        choices=['T', 'A', 'B'],
                        help="dataset")
    update_args = parser.parse_args()
    args = {
        # Basic args
        "result_dir": "results_{}/Commender",
        "model_dir": "checkpoints",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "output_model_name": "solver-chooser-ds",  
        "ckpt_dir": "checkpoints",
        "ckpt_step": '1400',
        # Text file args (ASR model output)
        "asroutput_path": "results_{}/Sensevoice/modelsoup.csv",
    }
    args['result_dir'] = args['result_dir'].format(update_args.dataset)
    args['output_model_name'] = update_args.output_model_name
    args['asroutput_path'] = args['asroutput_path'].format(update_args.dataset)
    args['ckpt_step'] = update_args.ckpt_step
    args['method'] = update_args.method

    # 初始化 vLLM 推理引擎
    commender = FoodCommender(args)
    
    # ASR output
    asr_data = pd.read_csv(args['asroutput_path'], header=0)
    
    # 处理数据并生成结果
    infer_filedir = f"{args['result_dir']}/infer"
    os.makedirs(infer_filedir, exist_ok=True)
    with open(f"{infer_filedir}/{args['output_model_name']}.jsonl", 'w', encoding='utf-8') as output_file:
        for _, row in tqdm(asr_data.iterrows(), total=len(asr_data), desc="Processing rows"):
            text = row['text']
            user = f'用户点餐需求：{text}'
    
            result = commender.commend(user)
                
            output_file.write(json.dumps(result, ensure_ascii=False) + '\n')