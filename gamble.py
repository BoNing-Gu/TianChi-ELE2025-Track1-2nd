import os
import json
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import difflib
import ast
import argparse
from difflib import get_close_matches

def extract_first_food(food_str):
    """从字符串格式的列表中提取第一个食物"""
    try:
        food_list = ast.literal_eval(food_str)  # 安全解析字符串为列表
        return food_list[0] if isinstance(food_list, list) and len(food_list) > 0 else None
    except (ValueError, SyntaxError):
        return None  # 解析失败时返回None
    
def get_top_food(df):
    def extract_top_food(food_str):
        try:
            food_list = ast.literal_eval(food_str)
            if not isinstance(food_list, list):
                return []
            max_score = max(item.get("评分", 0) for item in food_list)
            top_foods = [item.get("菜名") for item in food_list if item.get("评分", 0) == max_score]
            return top_foods
        except Exception:
            return []

    def correct_top_food(row):
        top_foods = row['top_food_pool']
        try:
            food_pool = ast.literal_eval(row['food_pool'])
        except Exception:
            food_pool = []

        corrected = []
        mismatch_count = 0
        for food in top_foods:
            if food in food_pool:
                corrected.append(food)
            elif food == '无':
                corrected.append(food)
            else:
                closest = get_close_matches(food, food_pool, n=1, cutoff=0.6)
                if closest:
                    corrected.append(closest[0])
                else:
                    corrected.append(food)  # 保留原名
                mismatch_count += 1
        return corrected, mismatch_count

    df['top_food_pool'] = df['food_from_chooser'].apply(extract_top_food)
    df[['top_food_pool', 'mismatch_count']] = df.apply(
        lambda row: pd.Series(correct_top_food(row)),
        axis=1
    )
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gamble")
    parser.add_argument("--step", "-s", 
                        type=str, 
                        choices=['Gambler', 'LLMChooser'], 
                        default="Gambler", 
                        help="control step")
    parser.add_argument("--name", "-n", 
                        type=str, 
                        choices=["gambler-final_one_from_four", "gambler-lora_results"], 
                        default="gambler-final_one_from_four", 
                        help="query file name")
    parser.add_argument("--dataset", "-d", 
                        type=str, 
                        default="b", 
                        choices=['T', 'A', 'B'],
                        help="dataset")
    update_args = parser.parse_args()
    args = {
        # Basic args
        "result_dir": "results_{}/Commender",
        "food_dictionary": "data/智慧养老_label_B/dim_ai_exam_food_category_filter_out.txt",
        # Commender args
        "model_dir": "checkpoints",
        "Commender1-Chooser:name": "slover-ds+cluster-1.1_1.2+searcher-bge+chooser-ds", 
        "Commender1-Chooser:version": "0",
        "Commender2-Chooser:name": "slover-gemini+cluster-1.1_1.2+searcher-bge+chooser-gemini", 
        "Commender2-Chooser:version": "0",
        "Commender3-Chooser:name": "slover-gpt+cluster-1.1_1.2+searcher-bge+chooser-gpt", 
        "Commender3-Chooser:version": "0",
        # Matcher args
        "Matcher-Chooser:name": "matcher-3ensemble-top3+chooser-gpt", 
        "Matcher-Chooser:version": "0",
        # Gambler args
        "Gambler:name": "gambler-final_one_from_four", 
        "Gambler:version": "0",
        "Chooser:name": "gambler-final_one_from_four+chooser-dsr1",
        "Chooser:version": "0",
    }
    args['result_dir'] = args['result_dir'].format(update_args.dataset)
    args['Gambler:name'] = update_args.name

    if update_args.step == 'Gambler':
        commender1_result = pd.read_csv(os.path.join(args['result_dir'], args['Commender1-Chooser:name'] + '.csv'), header=0)
        commender2_result = pd.read_csv(os.path.join(args['result_dir'], args['Commender2-Chooser:name'] + '.csv'), header=0)
        commender3_result = pd.read_csv(os.path.join(args['result_dir'], args['Commender3-Chooser:name'] + '.csv'), header=0)
        matcher_result = pd.read_csv(os.path.join(args['result_dir'], args['Matcher-Chooser:name'] + '.csv'), header=0)
        commender1_result = get_top_food(commender1_result)
        commender2_result = get_top_food(commender2_result)
        commender3_result = get_top_food(commender3_result)
        matcher_result = get_top_food(matcher_result)
        
        result_df = commender1_result[['uuid', 'text']].copy()
        result_df = pd.merge(result_df, commender1_result[['uuid', 'top_food_pool']], on='uuid', how='left').rename(columns={'top_food_pool': 'top_food_pool_from_commender1'})
        result_df = pd.merge(result_df, commender2_result[['uuid', 'top_food_pool']], on='uuid', how='left').rename(columns={'top_food_pool': 'top_food_pool_from_commender2'})
        result_df = pd.merge(result_df, commender3_result[['uuid', 'top_food_pool']], on='uuid', how='left').rename(columns={'top_food_pool': 'top_food_pool_from_commender3'})
        result_df = pd.merge(result_df, matcher_result[['uuid', 'top_food_pool']], on='uuid', how='left').rename(columns={'top_food_pool': 'top_food_pool_from_matcher'})
        # 合并成food_pool
        result_df['food_pool'] = result_df.apply(lambda x: list(set(x['top_food_pool_from_commender1'] + x['top_food_pool_from_commender2'] + x['top_food_pool_from_commender3'] + x['top_food_pool_from_matcher'])), axis=1)
        output_path = os.path.join(args['result_dir'], args['Gambler:name'] + '.csv')
        print(output_path)
        result_df.to_csv(output_path, index=False, encoding='utf-8')
        
    elif update_args.step == 'LLMChooser':
        # 推荐池
        result_df = pd.read_csv(os.path.join(args['result_dir'], f"{args['Gambler:name']}.csv"), header=0)

        # 食品库
        food_dictionary = pd.read_csv(args['food_dictionary'], sep='\t', header=0)
        food_set = set(food_dictionary['item_name'])
        food_set.add('无')

        # 加载选择后的数据
        query_filedir = f"{args['result_dir']}/query"
        output_file_path = os.path.join(query_filedir, f"{args['Gambler:name']}_{args['Gambler:version']}" + '_output.jsonl')
        print(output_file_path)
        data = []
        with open(output_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                custom_id = item.get('custom_id', '')
                content = item.get('parsed_content', {})[0]
                rank = content.get('排序', '')
                reason = content.get('理由', '')
                final_food = content.get('最优推荐', '')
                data.append({'uuid': custom_id, 'rank': rank, 'reason': reason, 'food_from_gambler': final_food})
        data = pd.DataFrame(data)
        data['in_dictionary'] = data['food_from_gambler'].apply(lambda x: 1 if x in food_set else 0)
        print(data['in_dictionary'].value_counts())

        # merge
        merged_df = pd.merge(result_df, data, how='left', on='uuid')
        # merged_df['food_from_diff-gambler'] = merged_df['food_pool'].apply(extract_first_food)
        merged_df['food'] = merged_df.apply(lambda row: row['food_from_gambler'] if row['in_dictionary'] == 1 else eval(row['top_food_pool_from_matcher'])[0], axis=1)
        print(merged_df.head(10))
        output_path = os.path.join(args['result_dir'], f"{args['Chooser:name']}.csv")
        print(output_path)
        merged_df.to_csv(output_path, index=False, encoding='utf-8')
    