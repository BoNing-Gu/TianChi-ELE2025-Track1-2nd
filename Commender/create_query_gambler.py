import pandas as pd
import json
import os
import argparse
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

def messages_builder(row):
    with open(f'src/system_prompt_gambler.txt', 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    text = row["text"]
    text = text.replace('天猫精灵', '')
    text = text.strip('，。？！：；、')

    food_pool = row['food_pool']

    user_prompt = f"""
    用户点餐需求："{text}"
    菜品列表："{food_pool}"
    你觉得哪种菜品最能接受？请根据用户需求进行评估和打分，然后按照从好到坏对菜品进行排序，最终以JSON格式返回结果字段（包含`排序`、`理由` 和 `最优推荐`三个字段）。
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
    # Gather all results
    result_dir = os.path.join(args['result_dir'], f"{args['name']}.csv")
    df = pd.read_csv(result_dir, header=0)
    print(f"Found {len(df)} results")

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

    for i, df in enumerate(dfs):
        if args["query_mode"] == "dev":
            i = "dev"
        if args['llm'] == "ds":
            model = 'deepseek-v3'
        elif args['llm'] == "r1":
            model = 'deepseek-r1'
        elif args['llm'] == "gpt":
            model = 'gpt-4.1-mini'
        elif args['llm'] == "gemini":
            model = 'gemini-2.0-flash'
        with open(f"{args['result_dir']}/query/{args['name']}_{i}.jsonl", 'w', encoding='utf-8') as fout:
            for idx, row in df.iterrows(): 
                row = row.where(pd.notnull(row), None)
                
                request = {
                    "custom_id": f"{row['uuid']}", 
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": messages_builder(row),
                        "temperature": 0.3, 
                        "max_tokens": 1024,
                        # "response_format": {"type": "json_object"}
                    }
                }
                fout.write(json.dumps(request, ensure_ascii=False, separators=(',', ':')) + '\n')
        print(f"Query file {i} created, saved to {args['result_dir']}/query/{args['name']}_{i}.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create query file for Chooser")
    parser.add_argument("--name", "-n", 
                        type=str, 
                        choices=["gambler-final_one_from_four", "gambler-lora_results"], 
                        help="query file name")
    parser.add_argument("--version", "-v", 
                        type=str, 
                        default='0', 
                        help="query file version")
    parser.add_argument("--llm", "-l", 
                        type=str, 
                        choices=["ds", "r1", "gpt", "gemini"],
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
        "name": "gambler-final_one_from_four", 
        "version": "0",
        "query_mode": "full",  # dev for 100 data query or full query
        "query_batch_size": 50000
    }
    args['result_dir'] = args['result_dir'].format(update_args.dataset)
    args['name'] = update_args.name
    args['version'] = update_args.version
    args['llm'] = update_args.llm

    create_query_flie(args)
    