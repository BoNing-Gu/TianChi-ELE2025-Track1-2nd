import os
import json
from pathlib import Path
from openai import OpenAI
import time
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


class OpenAIBatchProcessor:
    def __init__(self, llm, api_key):
        if llm == 'ds' or llm == 'r1':
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
            )
        elif llm == 'gpt':
            self.client = OpenAI(
                api_key=api_key
            )

    def upload_file(self, file_path):
        print(f"正在上传包含请求信息的JSONL文件...")
        file_object = self.client.files.create(file=Path(file_path), purpose="batch")
        print(f"文件上传成功。得到文件ID: {file_object.id}\n")
        return file_object.id

    def create_batch_job(self, input_file_id):
        print(f"正在基于文件ID，创建Batch任务...")
        # 请注意：选择Embedding文本向量模型进行调用时，endpoint的值需填写"/v1/embeddings"。
        batch = self.client.batches.create(input_file_id=input_file_id, endpoint="/v1/chat/completions", completion_window="24h")
        print(f"Batch任务创建完成。 得到Batch任务ID: {batch.id}\n")
        return batch.id

    def check_job_status(self, batch_id):
        print(f"正在检查Batch任务状态...")
        batch = self.client.batches.retrieve(batch_id=batch_id)
        print(f"Batch任务状态: {batch.status}\n")
        return batch.status

    def get_output_id(self, batch_id):
        print(f"正在获取Batch任务中执行成功请求的输出文件ID...")
        batch = self.client.batches.retrieve(batch_id=batch_id)
        print(f"输出文件ID: {batch.output_file_id}\n")
        return batch.output_file_id

    def get_error_id(self, batch_id):
        print(f"正在获取Batch任务中执行错误请求的输出文件ID...")
        batch = self.client.batches.retrieve(batch_id=batch_id)
        print(f"错误文件ID: {batch.error_file_id}\n")
        return batch.error_file_id

    def download_results(self, output_file_id, output_file_path):
        print(f"正在打印并下载Batch任务的请求成功结果...")
        content = self.client.files.content(output_file_id)
        # 打印部分内容以供测试
        print(f"打印请求成功结果的前1000个字符内容: {content.text[:1000]}...\n")
        # 保存结果文件至本地
        content.write_to_file(output_file_path)
        print(f"完整的输出结果已保存至本地输出文件result.jsonl\n")

    def download_errors(self, error_file_id, error_file_path):
        print(f"正在打印并下载Batch任务的请求失败信息...")
        content = self.client.files.content(error_file_id)
        # 打印部分内容以供测试
        print(f"打印请求失败信息的前1000个字符内容: {content.text[:1000]}...\n")
        # 保存错误信息文件至本地
        content.write_to_file(error_file_path)
        print(f"完整的请求失败信息已保存至本地错误文件error.jsonl\n")

def main(input_file_path, output_file_path, error_file_path, llm):
    key = json.load(open("src/key.json", "r"))
    if llm == 'ds' or llm == 'r1':
        api_key=key['DASHSCOPE_API_KEY']
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY not set")
        print(f"正在使用Dashscope API Key")
        error_html = 'https://help.aliyun.com/zh/model-studio/developer-reference/error-code'
    elif llm == 'gpt':
        api_key=key['OPENAI_API_KEY']
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        print(f"正在使用OpenAI API Key")
        error_html = 'https://platform.openai.com/docs/guides/error-codes#api-errors'
        
    processor = OpenAIBatchProcessor(llm, api_key)
    
    try:
        # Step 1: 上传包含请求信息的JSONL文件，得到输入文件ID
        input_file_id = processor.upload_file(input_file_path)
        # Step 2: 基于输入文件ID，创建Batch任务
        batch_id = processor.create_batch_job(input_file_id)
        # Step 3: 检查Batch任务状态直到结束
        status = ""
        while status not in ["completed", "failed", "expired", "cancelled"]:
            status = processor.check_job_status(batch_id)
            print(f"等待任务完成...")
            time.sleep(10)  # 等待10秒后再次查询状态
        # 如果任务失败，则打印错误信息并退出
        if status == "failed":
            batch = processor.client.batches.retrieve(batch_id)
            print(f"Batch任务失败。错误信息为:{batch.errors}\n")
            print(f"参见错误码文档: {error_html}")
            return
        # Step 4: 下载结果：如果输出文件ID不为空，则打印请求成功结果的前1000个字符内容，并下载完整的请求成功结果到本地输出文件；
        # 如果错误文件ID不为空，则打印请求失败信息的前1000个字符内容，并下载完整的请求失败信息到本地错误文件。
        output_file_id = processor.get_output_id(batch_id)
        if output_file_id:
            processor.download_results(output_file_id, output_file_path)
        error_file_id = processor.get_error_id(batch_id)
        if error_file_id:
            processor.download_errors(error_file_id, error_file_path)
            print(f"参见错误码文档:{error_html}")
            # TODO:处理error数据（添加到output中）
    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"参见错误码文档:{error_html}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Query")
    parser.add_argument("--name", "-n", 
                        type=str, 
                        choices=[
                            # Solver
                            "asr-modelsoup+solver-ds",
                            # Chooser
                            "slover-ds+cluster-1.1_1.2+searcher-bge", 
                            # Gambler
                            "gambler-final_one_from_four", 
                            "gambler-lora_results"], 
                        help="query file name")
    parser.add_argument("--version", "-v", 
                        type=str, 
                        default="0", 
                        help="query file version")
    parser.add_argument("--llm", "-l", 
                        type=str, 
                        choices=["ds", "gpt", "r1"],
                        default="ds",
                        help="query llm name")
    parser.add_argument("--dataset", "-d", 
                        type=str, 
                        default="B", 
                        choices=['T', 'A', 'B'],
                        help="dataset")
    update_args = parser.parse_args()
    args = {
        "result_dir": "results_{}/Commender",
        "name": "asr-modelsoup+solver-ds", 
        "version": "0",
    }
    args['result_dir'] = args['result_dir'].format(update_args.dataset)
    args['name'] = update_args.name
    args['version'] = update_args.version
    args['llm'] = update_args.llm

    query_filedir = f"{args['result_dir']}/query"

    input_file_path = os.path.join(query_filedir, f"{args['name']}_{args['version']}" + '.jsonl')
    output_file_path = os.path.join(query_filedir, f"{args['name']}_{args['version']}" + '_output.jsonl')
    error_file_path = os.path.join(query_filedir, f"{args['name']}_{args['version']}" + '_error.jsonl')

    main(input_file_path, output_file_path, error_file_path, args['llm'])

    parsed_results = []
    parsed_errors = []
    with open(output_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            result = json.loads(line)
            content = result.get('response', {}).get('body', {}).get('choices', [{}])[0].get('message', {}).get('content', '')
            content = content.strip()
            try:
                parsed_content = extract_json(content)
                # print(parsed_content)
                result['parsed_content'] = parsed_content
                parsed_results.append(result)
            except:
                parsed_errors.append(result)
    
    with open(output_file_path, 'w') as f:
        for result in parsed_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    with open(error_file_path, 'w') as f:
        for result in parsed_errors:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

