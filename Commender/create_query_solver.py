import pandas as pd
import json
import os
import argparse
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

def messages_builder(row, choice):
    with open(f'src/system_prompt_solver_{choice}.txt', 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    text = row["text"]
    text = text.replace('天猫精灵', '')
    text = text.strip('，。？！：；、')

    user_prompt = f"""
    用户提出的点餐要求是："{text}"
    你的菜品推荐是：
    """

    return [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

def create_query_flie(args):
    # Get ASR text
    df = pd.read_csv(args['text_file'], header=0)
    dom = pd.read_csv(args['dom_file'], header=0)
    df = pd.merge(df, dom, on="uuid", how="left")
    print(f"Found {len(df)} texts")

    # Create query file
    dfs = []
    if args["query_mode"] == "dev":
        # first 100 data for dev
        df = df[:100]
        dfs.append(df)
    elif args["query_mode"] == "full":
        # if len(df) > args["query_batch_size"], split into multiple files
        if len(df) > args["query_batch_size"]:
            for i in range(0, len(df), args["query_batch_size"]):
                dfs.append(df[i:i + args["query_batch_size"]])
        else:
            dfs.append(df)
    else:
        raise ValueError("query_mode must be 'dev' or 'full'")

    os.makedirs(f"{args['result_dir']}/query", exist_ok=True)
    for i, df in enumerate(dfs):
        if args["query_mode"] == "dev":
            i = "dev"
        with open(f"{args['result_dir']}/query/{args['name']}+solver-ds_{i}.jsonl", 'w', encoding='utf-8') as fout:
            for idx, row in df.iterrows(): 
                # if row['dom'] == 0:  # 所有的text都要查询
                #     continue
                row = row.where(pd.notnull(row), None)
                
                request = {
                    "custom_id": f"{row['uuid']}", 
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "deepseek-v3",
                        "messages": messages_builder(row, 'a'),
                        "temperature": 0.2, 
                        "max_tokens": 24,
                        # "response_format": {"type": "json_object"}
                    }
                }
                fout.write(json.dumps(request, ensure_ascii=False, separators=(',', ':')) + '\n')
        print(f"Query file {i} created, saved to {args['result_dir']}/query/{args['name']}+solver-ds_{i}.jsonl")

        with open(f"{args['result_dir']}/query/{args['name']}+solver-gemini_{i}.jsonl", 'w', encoding='utf-8') as fout:
            for idx, row in df.iterrows(): 
                # if row['dom'] == 0:  # 所有的text都要查询
                #     continue
                row = row.where(pd.notnull(row), None)
                
                request = {
                    "custom_id": f"{row['uuid']}", 
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gemini-2.0-flash",
                        "messages": messages_builder(row, 'b'),
                        "temperature": 0.2, 
                        "max_tokens": 24,
                        # "response_format": {"type": "json_object"}
                    }
                }
                fout.write(json.dumps(request, ensure_ascii=False, separators=(',', ':')) + '\n')
        print(f"Query file {i} created, saved to {args['result_dir']}/query/{args['name']}+solver-gemini_{i}.jsonl")

        with open(f"{args['result_dir']}/query/{args['name']}+solver-gpt_{i}.jsonl", 'w', encoding='utf-8') as fout:
            for idx, row in df.iterrows(): 
                # if row['dom'] == 0:  # 所有的text都要查询
                #     continue
                row = row.where(pd.notnull(row), None)
                
                request = {
                    "custom_id": f"{row['uuid']}", 
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4.1-mini",
                        "messages": messages_builder(row, 'c'),
                        "temperature": 0.2, 
                        "max_tokens": 24,
                        # "response_format": {"type": "json_object"}
                    }
                }
                fout.write(json.dumps(request, ensure_ascii=False, separators=(',', ':')) + '\n')
        print(f"Query file {i} created, saved to {args['result_dir']}/query/{args['name']}+solver-gpt_{i}.jsonl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create query file for Solver")
    parser.add_argument("--name", "-n", 
                        type=str, 
                        default="asr-modelsoup", 
                        help="query file name")
    parser.add_argument("--version", "-v", 
                        type=str, 
                        default='0', 
                        help="query file version")
    parser.add_argument("--dataset", "-d", 
                        type=str, 
                        default="B", 
                        choices=['T', 'A', 'B'],
                        help="dataset")
    update_args = parser.parse_args()
    args = {
        "text_file": "results_{}/Sensevoice/modelsoup.csv",
        "dom_file": "results_{}/EfficientNet/dom.csv",
        "result_dir": "results_{}/Commender",
        "name": "asr-modelsoup",
        "version": "0",
        "query_mode": "full",  # dev for 100 data query or full query
        "query_batch_size": 50000
    }
    args['text_file'] = args['text_file'].format(update_args.dataset)
    args['dom_file'] = args['dom_file'].format(update_args.dataset)
    args['result_dir'] = args['result_dir'].format(update_args.dataset)
    args['name'] = update_args.name
    args['version'] = update_args.version

    create_query_flie(args)
    
     