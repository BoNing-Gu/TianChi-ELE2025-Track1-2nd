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

def prepare_model(model_path, enable_lora, max_model_len=5120):
    # 初始化 vLLM 推理引擎
    llm = LLM(model=model_path, 
              tokenizer=model_path, 
              max_model_len=max_model_len, 
              trust_remote_code=True, 
              enable_lora=enable_lora, 
              max_lora_rank=32,
              disable_log_stats=True)
    return llm
    
def get_completion(prompts, llm, lora_path, max_tokens=5120, temperature=0.3, top_p=0.95):
    stop_token_ids = [151329, 151336, 151338]
    # 创建采样参数。temperature 控制生成文本的多样性，top_p 控制核心采样的概率
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop_token_ids=stop_token_ids)
    # 初始化 vLLM 推理引擎
    if lora_path is not None:
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False, lora_request=LoRARequest("adapter_name", 1, lora_path))
    else:
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    return outputs

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

def get_commender_result(infer_filedir, commender_name):
    def extract_top_food(food_list):
        try:
            if not isinstance(food_list, list) or len(food_list) == 0:
                return []
            max_score = max(item.get("评分", 0) for item in food_list)
            top_foods = [item.get("菜名") for item in food_list if item.get("评分", 0) == max_score]
            return top_foods
        except Exception:
            return []
    infer_filepath = os.path.join(infer_filedir, f'{commender_name}.jsonl')
    # 读取jsonl文件，形成数据框
    data = []
    with open(infer_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                # 确保每条记录都有所需的字段
                if all(key in item for key in ['custom_id', 'text', 'content']):
                    # 如果parsed_content不存在，尝试从content解析
                    if 'parsed_content' not in item:
                        try:
                            item['parsed_content'] = json.loads(item['content'])
                        except json.JSONDecodeError:
                            item['parsed_content'] = []
                    data.append({
                        'custom_id': item['custom_id'],
                        'text': item['text'],
                        f'top_food_pool_from_{commender_name}': extract_top_food(item['parsed_content'])
                    })
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line: {line}, error: {e}")
                continue
    
    df = pd.DataFrame(data, columns=['custom_id', 'text', f'top_food_pool_from_{commender_name}'])
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_model_name', '-o', 
                        type=str, 
                        choices=['gambler'] , 
                        help='model name')
    parser.add_argument('--ckpt_step', '-c', 
                        type=str, 
                        help='ckpt step')
    parser.add_argument("--dataset", "-d", 
                        type=str, 
                        default="B", 
                        choices=['T', 'A', 'B'],
                        help="dataset")
    parser.add_argument('--enable_lora', '-e', 
                        action='store_true',
                        default=False,
                        help='wether to enable lora (default: False)')
    parser.add_argument('--inference', '-i', 
                        action='store_true',
                        default=False,
                        help='wether to inference (default: False)')
    update_args = parser.parse_args()
    args = {
        # Basic args
        "result_dir": "results_{}/Commender",
        "model_dir": "checkpoints",
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "output_model_name": "gambler",  
        "logger_dir": "logs/Lora",
        "ckpt_dir": "checkpoints/Lora",
        "ckpt_step": '750',
        # Commender file args (Commender output)
        "Commender": ['solver-chooser-ds', 'solver-chooser-gemini', 'solver-chooser-gpt', 'matcher-chooser-gpt'],
    }
    args['result_dir'] = args['result_dir'].format(update_args.dataset)
    args['output_model_name'] = update_args.output_model_name
    args['ckpt_step'] = update_args.ckpt_step
    args['enable_lora'] = update_args.enable_lora
    args['inference'] = update_args.inference
    
    # 初始化 vLLM 推理引擎
    # save_path = os.path.join(args['ckpt_dir'], args['output_model_name'])
    model_dir = snapshot_download(args['model_id'], cache_dir=args['model_dir'], revision='master')
    print(model_dir)
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    # LoRA权重
    if args['enable_lora']:
        lora_path = os.path.join(args['ckpt_dir'], args['output_model_name'], f"checkpoint-{args['ckpt_step']}")
    else: 
        lora_path = None
    # 加载vllm
    llm = prepare_model(model_dir, args['enable_lora'])
    
    # read system_prompt & dataset
    with open('src/system_prompt_lora_gambler.txt', 'r') as f:
        system_prompt = f.read()
    print(system_prompt)
    
    # 处理数据并生成结果
    infer_filedir = f"{args['result_dir']}/infer"
    os.makedirs(infer_filedir, exist_ok=True)
    # Commender output
    commender_df = pd.DataFrame()
    for commender_name in args['Commender']:
        commend_result = get_commender_result(infer_filedir, commender_name)
        if commender_df.empty:
            commender_df = commend_result
        else:
            # 如果不为空，按照custom_id列进行合并
            commender_df = pd.merge(
                commender_df,
                commend_result[['custom_id', f'top_food_pool_from_{commender_name}']],
                on='custom_id',
                how='inner'
            )
    commender_df['food_pool'] = commender_df.apply(lambda x: list(set(x['top_food_pool_from_solver-chooser-ds'] + x['top_food_pool_from_solver-chooser-gemini'] + x['top_food_pool_from_solver-chooser-gpt'] + x['top_food_pool_from_matcher-chooser-gpt'])), axis=1)
    results_df = commender_df.copy().rename(columns={'custom_id': 'uuid'})
    results_df = results_df[['uuid', 'text', 'food_pool']]
    results_df.to_csv(os.path.join(args['result_dir'], 'gambler-lora_results.csv'), index=False, encoding='utf-8')

    if args['inference']:
        with open(f"{infer_filedir}/{args['output_model_name']}.jsonl", 'w', encoding='utf-8') as output_file:
            for _, row in tqdm(commender_df.iterrows(), total=len(commender_df), desc="Processing rows"):
                text = row['text']
                food_pool = row['food_pool']
                user = f'Instruction:\n{system_prompt}\nQuestion:\n用户点餐需求：{text}\n菜品列表：{food_pool}\n<think>\n'
        
                # 构建消息列表
                message = [
                    {"role": "user", "content": user}
                ]
        
                # 准备模型输入
                inputs = tokenizer.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=True
                )
        
                # 调用模型生成回答
                outputs = get_completion(inputs, llm, lora_path, max_tokens=1024, temperature=0.3, top_p=1)
                response = outputs[0].outputs[0].text
                if r"</think>" in response:
                    think_content, answer_content = response.split(r"</think>")
                try:
                    # 尝试读取为json
                    response_json = extract_json(answer_content)
                    result = {
                        'custom_id': row['custom_id'],
                        'text': row['text'],
                        'think_content': think_content,
                        'answer_content': answer_content,
                        'parsed_content': response_json
                    }
                except (json.JSONDecodeError, ValueError) as e:
                    result = {
                        'custom_id': row['uuid'],
                        'text': row['text'],
                        'think_content': think_content,
                        'answer_content': answer_content,
                        'parsed_content': '无法解析'
                    }
                    
                output_file.write(json.dumps(result, ensure_ascii=False) + '\n')