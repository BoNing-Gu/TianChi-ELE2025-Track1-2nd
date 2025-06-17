import os
import json
import pandas as pd
import argparse

def main(result_path, ds_output_dir, ds_cat):
    # 确保输出目录存在
    ds_output_path = os.path.join(ds_output_dir, ds_cat)
    os.makedirs(ds_output_path, exist_ok=True)

    df = pd.DataFrame()
    for lb in ['A', 'B']:
        result_filepath = result_path.format(lb)
        result_data = pd.read_csv(result_filepath, header=0)
        result_data = result_data[['uuid', 'text', 'food']]
        df = pd.concat([df, result_data], ignore_index=True)
        print(f"Loaded result data from {result_path}, shape: {df.shape}")
    
    # 构造并保存json数据集
    dataset = []
    for _, row in df.iterrows():
        text = row["text"]
        text = text.replace('天猫精灵', '')
        text = text.strip('，。？！：；、')
        food = row["food"]
        if food == '无':
            food = '指令与菜品推荐无关，请重新输入指令'
        dataset.append({
            "instruction": text,
            "input": "",
            "output": food
        })
    
    output_file = os.path.join(ds_output_path, f"dataset.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"Dataset saved to {output_file}, len: {len(df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dataset")
    parser.add_argument("--cat", "-c", 
                        type=str, 
                        default='0', 
                        choices=['end-2-end'],
                        help="dataset category")
    update_args = parser.parse_args()
    args = {
        "result_path": 'results_{}/Commender/gambler-final_one_from_four+chooser-dsr1.csv',
        "ds_cat": "end-2-end",
        "ds_output_dir": "data/dataset_F",
    }

    main(args['result_path'], args['ds_output_dir'], args['ds_cat'])