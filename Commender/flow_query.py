import os
import json
import uuid
import time
from openai import OpenAI
from pathlib import Path
from tqdm.auto import tqdm
import argparse
import re
os.environ['http_proxy'] = 'http://127.0.0.1:11000'
os.environ['https_proxy'] = 'http://127.0.0.1:11000'
print(f'Warning: proxy enabled -> {os.environ["http_proxy"]} \n {os.environ["https_proxy"]}')

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

class OpenAIProcessor:
    def __init__(self, llm, api_key):
        if llm == 'ds':
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
        elif llm == 'gemini':
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
        elif llm == 'gpt':
            self.client = OpenAI(
                api_key=api_key
            )
        
    def process_request(self, request_data):
        """处理单个OpenAI格式请求"""
        try:
            # print(request_data["body"]["messages"])
            response = self.client.chat.completions.create(
                model=request_data["body"]["model"],
                messages=request_data["body"]["messages"],
                temperature=request_data["body"].get("temperature", 0.3),
                max_tokens=request_data["body"].get("max_tokens", 512)
            )
            # print(f"Response: {response}")
            return self._format_response(request_data, response)
        
        except Exception as e:
            return self._format_error(request_data, str(e))

    def _format_response(self, request_data, response):
        """格式化OpenAI兼容响应"""
        return {
            "id": f"chatcmpl-{str(uuid.uuid4())}",
            "custom_id": request_data["custom_id"],
            "response": {
                "status_code": 200,
                "request_id": str(uuid.uuid4()),
                "body": {
                    "created": int(time.time()),
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    },
                    "model": response.model,
                    "choices": [{
                        "finish_reason": choice.finish_reason,
                        "index": index,
                        "message": {
                            "role": choice.message.role,
                            "content": choice.message.content
                        }
                    } for index, choice in enumerate(response.choices)],
                    "object": "chat.completion"
                }
            },
            "error": None
        }

    def _format_error(self, request_data, error_msg):
        """格式化错误响应"""
        return {
            "id": str(uuid.uuid4()),
            "custom_id": request_data.get("custom_id", ""),
            "response": None,
            "error": error_msg
        }


def process_file_solver(input_path, output_path, error_path, llm, api_key):
    processor = OpenAIProcessor(llm, api_key)

    with open(input_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    written_ids = set()
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                custom_id = json.loads(line).get("custom_id")
                if custom_id is not None:
                    written_ids.add(custom_id)
        print('已完成', len(written_ids))

    with open(input_path, 'r') as infile, \
         open(output_path, 'a') as outfile, \
         open(error_path, 'a') as errfile:
        
        # 设置tqdm进度条
        progress_bar = tqdm(
            infile, 
            total=total_lines,
            desc="Processing requests",
            unit="req",
            dynamic_ncols=True,  # 自动适应终端宽度
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        )

        success_count = 0
        error_count = 0
        pass_count = 0

        for line in progress_bar:
            line = line.strip()
            if not line:
                continue
            
            try:
                request_data = json.loads(line)
                if request_data['custom_id'] in written_ids:
                    pass_count += 1
                    continue
                # print(f"Processing request: {request_data}")
                result = processor.process_request(request_data)
                
                if result.get("error"):
                    errfile.write(json.dumps(result, ensure_ascii=False) + '\n')
                    error_count += 1
                else:
                    outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
                    success_count += 1
                
                # 更新进度条附加信息
                progress_bar.set_postfix({
                    'success': success_count,
                    'error': error_count,
                    'rate': f"{success_count/(success_count+error_count):.1%}",
                    'pass': pass_count,
                })
                
                # # 避免速率限制
                # time.sleep(1)
                
            except json.JSONDecodeError as e:
                error_result = {
                    "id": str(uuid.uuid4()),
                    "custom_id": "unknown",
                    "response": None,
                    "error": f"Invalid JSON: {str(e)}"
                }
                errfile.write(json.dumps(error_result, ensure_ascii=False) + '\n')
                error_count += 1
            # break
    print(f"\nProcessing completed. Success: {success_count}, Error: {error_count}")

def process_file_chooser(input_path, output_path, error_path, llm, api_key):
    processor = OpenAIProcessor(llm, api_key)

    with open(input_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    written_ids = set()
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                custom_id = json.loads(line).get("custom_id")
                if custom_id is not None:
                    written_ids.add(custom_id)
        print('已完成', len(written_ids))

    with open(input_path, 'r') as infile, \
         open(output_path, 'a') as outfile, \
         open(error_path, 'a') as errfile:
        
        # 设置tqdm进度条
        progress_bar = tqdm(
            infile, 
            total=total_lines,
            desc="Processing requests",
            unit="req",
            dynamic_ncols=True,  # 自动适应终端宽度
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        )

        success_count = 0
        error_count = 0
        attempts_count = 0
        pass_count = 0

        for line in progress_bar:
            line = line.strip()
            if not line:
                continue
            
            try:
                request_data = json.loads(line)
                if request_data['custom_id'] in written_ids:
                    pass_count += 1
                    continue
                
                max_attempts = 5
                attempt = 0
                success = False
                while attempt < max_attempts:
                    attempts_count += 1
                    try:
                        # 请求模型
                        result = processor.process_request(request_data)
                        content = result.get('response', {}).get('body', {}).get('choices', [{}])[0].get('message', {}).get('content', '')
                        content = content.strip()
                        # print(content)
                        parsed_content = extract_json(content)
                        # print(parsed_content)

                        # 如果成功，替换原始内容为解析后的，并写入输出
                        result['parsed_content'] = parsed_content
                        outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
                        success_count += 1
                        success = True
                        break
                    except Exception as e:
                        attempt += 1
                        print(f"[Attempt {attempt}] Parsing failed: {e}")
                        time.sleep(1)
                
                if not success:
                    # 尝试失败，写入错误文件
                    result["error"] = f"Failed to parse content after {max_attempts} attempts."
                    errfile.write(json.dumps(result, ensure_ascii=False) + '\n')
                    error_count += 1
                
                # 更新进度条附加信息
                progress_bar.set_postfix({
                    'success': success_count,
                    'error': error_count,
                    'rate': f"{success_count/(success_count+error_count):.1%}", 
                    'pass': pass_count,
                    'attempts': attempts_count
                })
                
                # # 避免速率限制
                # time.sleep(1)
                
            except json.JSONDecodeError as e:
                error_result = {
                    "id": str(uuid.uuid4()),
                    "custom_id": "unknown",
                    "response": None,
                    "error": f"Invalid JSON: {str(e)}",
                    "parsed_content": None
                }
                errfile.write(json.dumps(error_result, ensure_ascii=False) + '\n')
                error_count += 1
            # break
    print(f"\nProcessing completed. Success: {success_count}, Error: {error_count}")

def main(input_file_path, output_file_path, error_file_path, llm, step):
    key = json.load(open("src/key.json", "r"))
    if llm == 'ds':
        api_key=key['DASHSCOPE_API_KEY']
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY not set")
        print(f"正在使用Dashscope API Key")
    elif llm == 'gemini':
        api_key=key['GOOGLE_API_KEY']
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")
        print(f"正在使用Google API Key")
    elif llm == 'gpt':
        api_key=key['OPENAI_API_KEY']
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        print(f"正在使用OpenAI API Key")
    
    if step == 'solver':
        process_file_solver(input_file_path, output_file_path, error_file_path, llm, api_key)
    elif step == 'chooser':
        process_file_chooser(input_file_path, output_file_path, error_file_path, llm, api_key)
    print(f"Processing completed. Results saved to {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flow Query")
    parser.add_argument("--name", "-n", 
                        type=str, 
                        choices=[
                            # Solver
                            "asr-modelsoup+solver-gemini", 
                            "asr-modelsoup+solver-gpt",
                            # Chooser
                            "slover-gemini+cluster-1.1_1.2+searcher-bge", 
                            "slover-gpt+cluster-1.1_1.2+searcher-bge", 
                            # Matcher
                            "matcher-3ensemble-top3"], 
                        help="query file name")
    parser.add_argument("--version", "-v", 
                        type=str, 
                        default="0", 
                        help="query file version")
    parser.add_argument("--llm", "-l", 
                        type=str, 
                        choices=["ds", "gpt", "gemini"],
                        default="gemini",
                        help="query llm name")
    parser.add_argument("--dataset", "-d", 
                        type=str, 
                        default="b", 
                        choices=['T', 'A', 'B'],
                        help="dataset")
    update_args = parser.parse_args()
    args = {
        "result_dir": "results_{}/Commender",
        "name": "asr-modelsoup+solver-gemini",
        "version": "0",
    }
    args['result_dir'] = args['result_dir'].format(update_args.dataset)
    args['name'] = update_args.name
    args['version'] = update_args.version
    args['llm'] = update_args.llm
    if args['name'] in ["asr-modelsoup+solver-ds", "asr-modelsoup+solver-gemini", "asr-modelsoup+solver-gpt"]:
        step = 'solver'
    elif args['name'] in ["slover-ds+cluster-1.1_1.2+searcher-bge", "slover-gemini+cluster-1.1_1.2+searcher-bge", "slover-gpt+cluster-1.1_1.2+searcher-bge", "matcher-3ensemble-top3"]:
        step = 'chooser'

    query_filedir = f"{args['result_dir']}/query"
    os.makedirs(query_filedir, exist_ok=True)

    input_file_path = os.path.join(query_filedir, f"{args['name']}_{args['version']}.jsonl")
    output_file_path = os.path.join(query_filedir, f"{args['name']}_{args['version']}_output.jsonl")
    error_file_path = os.path.join(query_filedir, f"{args['name']}_{args['version']}_error.jsonl")

    main(input_file_path, output_file_path, error_file_path, args['llm'], step)