import os
import json
import pandas as pd
import argparse

def main(input_file_path, output_file_path, ds_output_dir, ds_cat):
    # 确保输出目录存在
    ds_output_path = os.path.join(ds_output_dir, ds_cat)
    os.makedirs(ds_output_path, exist_ok=True)
    
    # 读取输入文件
    input_data = []
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            input_data.append(json.loads(line))
    
    # 读取输出文件
    output_data = []
    with open(output_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            output_data.append(json.loads(line))
    
    # 创建DataFrame
    records = []
    for inp, out in zip(input_data, output_data):
        # 确保custom_id匹配
        if inp['custom_id'] != out['custom_id']:
            continue
        
        # 提取system和user消息
        messages = inp['body']['messages']
        system = next((msg['content'] for msg in messages if msg['role'] == 'system'), "")
        user = next((msg['content'] for msg in messages if msg['role'] == 'user'), "")
        
        # 提取assistant回复
        assistant_thinking = out['response']['body']['choices'][0]['message']['reasoning_content']
        assistant_parsed = out.get('parsed_content', "")
        
        records.append({
            'custom_id': inp['custom_id'],
            'system': system,
            'user': user,
            'assistant_thinking': f'{assistant_thinking}',
            'assistant': json.dumps(assistant_parsed, ensure_ascii=False)
        })
    
    df = pd.DataFrame(records)
    
    # # 保存system prompt到txt文件
    # if not df.empty:
    #     system_prompt = df.iloc[0]['system']
    #     system_file = os.path.join(ds_output_path, f"system.txt")
    #     with open(system_file, 'w', encoding='utf-8') as f:
    #         f.write(system_prompt)
    
    # 构造并保存json数据集
    dataset = []
    for _, row in df.iterrows():
        dataset.append({
            "instruction": row['user'],
            "input": "",
            "reasoning": row['assistant_thinking'],
            "output": row['assistant']
        })
    
    output_file = os.path.join(ds_output_path, f"dataset.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"Dataset saved to {output_file}")
    # print(f"System prompt saved to {system_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dataset")
    parser.add_argument("--name", "-n", 
                        type=str, 
                        choices=[
                                 # Gambler * 1
                                 "gambler-final_one_from_four"], 
                        help="dataset name")
    parser.add_argument("--version", "-v", 
                        type=str, 
                        default='0', 
                        help="query file version")
    parser.add_argument("--cat", "-c", 
                        type=str, 
                        default='gambler', 
                        choices=['gambler'],
                        help="dataset category")
    parser.add_argument("--dataset", "-d", 
                        type=str, 
                        default="B", 
                        choices=['T', 'A', 'B'],
                        help="dataset")
    update_args = parser.parse_args()
    args = {
        "result_dir": "results_{}/Commender",
        "name": "gambler-final_one_from_four", 
        "version": "0",
        "ds_output_dir": "data/dataset_{}",
        "ds_cat": "gambler",
    }
    args['result_dir'] = args['result_dir'].format(update_args.dataset)
    args['ds_output_dir'] = args['ds_output_dir'].format(update_args.dataset)
    args['name'] = update_args.name
    args['version'] = update_args.version
    args['ds_cat'] = update_args.cat

    query_filedir = f"{args['result_dir']}/query"

    input_file_path = os.path.join(query_filedir, f"{args['name']}_{args['version']}" + '.jsonl')
    output_file_path = os.path.join(query_filedir, f"{args['name']}_{args['version']}" + '_output.jsonl')

    main(input_file_path, output_file_path, args['ds_output_dir'], args['ds_cat'])