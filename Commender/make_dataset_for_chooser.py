import os
import json
import pandas as pd
import argparse

def main(input_file_path, output_file_path, asroutput_path, ds_output_dir, ds_cat):
    # 确保输出目录存在
    ds_output_path = os.path.join(ds_output_dir, ds_cat)
    os.makedirs(ds_output_path, exist_ok=True)

    # ASR output
    asr_data = pd.read_csv(asroutput_path, header=0)
    
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
    inp_records = []
    for inp in input_data:
        text = asr_data[asr_data['uuid'] == inp['custom_id']]['text'].values[0]
        text = text.replace('天猫精灵', '')
        text = text.strip('，。？！：；、')
        
        # 提取system和user消息
        user = f'用户点餐需求：{text}'
        
        inp_records.append({
            'custom_id': inp['custom_id'],
            'user': user
        })
    out_records = []
    for out in output_data:
        # 提取assistant回复
        assistant = out.get('parsed_content', "")
        
        out_records.append({
            'custom_id': out['custom_id'],
            'assistant': json.dumps(assistant, ensure_ascii=False)
        })
    
    inp_df = pd.DataFrame(inp_records)
    out_df = pd.DataFrame(out_records)
    df = pd.merge(inp_df, out_df, on='custom_id', how='inner')
    
    # 构造并保存json数据集
    dataset = []
    for _, row in df.iterrows():
        dataset.append({
            "instruction": row['user'],
            "input": "",
            "output": row['assistant']
        })
    
    output_file = os.path.join(ds_output_path, f"dataset.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"Dataset saved to {output_file}, len: {len(df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dataset")
    parser.add_argument("--name", "-n", 
                        type=str, 
                        choices=[# Commender * 4
                                 "slover-ds+cluster-1.1_1.2+searcher-bge", 
                                 "slover-gemini+cluster-1.1_1.2+searcher-bge", 
                                 "slover-gpt+cluster-1.1_1.2+searcher-bge",
                                 "matcher-3ensemble-top3"], 
                        help="dataset name")
    parser.add_argument("--version", "-v", 
                        type=str, 
                        default='0', 
                        help="query file version")
    parser.add_argument("--cat", "-c", 
                        type=str, 
                        default='0', 
                        choices=['solver-chooser-ds', 'solver-chooser-gemini', 'solver-chooser-gpt', 'matcher-chooser-gpt'],
                        help="dataset category")
    parser.add_argument("--dataset", "-d", 
                        type=str, 
                        default="B", 
                        choices=['T', 'A', 'B'],
                        help="dataset")
    update_args = parser.parse_args()
    args = {
        "result_dir": "results_{}/Commender",
        "name": "matcher-3ensemble-top3", 
        "version": "0",
        "ds_output_dir": "data/dataset_{}",
        "ds_cat": "matcher-chooser-gpt",
        # Text file args (ASR model output)
        "asroutput_path": "results_{}/Sensevoice/modelsoup.csv",
    }
    args['result_dir'] = args['result_dir'].format(update_args.dataset)
    args['ds_output_dir'] = args['ds_output_dir'].format(update_args.dataset)
    args['asroutput_path'] = args['asroutput_path'].format(update_args.dataset)
    args['name'] = update_args.name
    args['version'] = update_args.version
    args['ds_cat'] = update_args.cat

    query_filedir = f"{args['result_dir']}/query"

    input_file_path = os.path.join(query_filedir, f"{args['name']}_{args['version']}" + '.jsonl')
    output_file_path = os.path.join(query_filedir, f"{args['name']}_{args['version']}" + '_output.jsonl')

    main(input_file_path, output_file_path, args['asroutput_path'], args['ds_output_dir'], args['ds_cat'])