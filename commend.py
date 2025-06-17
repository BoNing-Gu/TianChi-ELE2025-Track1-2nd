from Commender.cluster import DishClusteringModel
from text2vec import SentenceModel, cos_sim, semantic_search, BM25
import os
import json
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import difflib
import ast
import argparse


def find_most_similar(food, menu):
    """将food和列表中的每一个比较，找到相似度最高的选项，返回这个相似度最高的选项"""
    if not menu:
        return food
    similarities = [(dish, difflib.SequenceMatcher(None, food, dish).ratio()) for dish in menu]
    most_similar_dish = max(similarities, key=lambda x: x[1])[0]
    return most_similar_dish

def extract_first_food(food_str):
    """从字符串格式的列表中提取第一个食物"""
    try:
        food_list = ast.literal_eval(food_str)  # 安全解析字符串为列表
        return food_list[0] if isinstance(food_list, list) and len(food_list) > 0 else None
    except (ValueError, SyntaxError):
        return None  # 解析失败时返回None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Commend")
    parser.add_argument("--step", "-s", 
                        type=str, 
                        choices=['Commender', 'LLMChooser'], 
                        default="Commender", 
                        help="control step")
    parser.add_argument("--dataset", "-d", 
                        type=str, 
                        default="B", 
                        choices=['T', 'A', 'B'],
                        help="dataset")
    update_args = parser.parse_args()
    args = {
        # Basic args
        "result_dir": "results_{}/Commender",
        "food_dictionary": "data/智慧养老_label_B/dim_ai_exam_food_category_filter_out.txt",
        # Solver file args
        "Solver:name": "asr-modelsoup+solver",
        "Solver:LLM": ["ds", "gemini", "gpt"],  # 
        "Solver:version": "0",
        # Commender args
        "model_dir": "checkpoints",
        "Cluster:model_id": "Cluster",
        "Cluster:version": "0",
        "Searcher:model_id" : "text2vec-bge-large-chinese",
        "Commender:name": "slover-{}+cluster-1.1_1.2+searcher-bge", 
        "Commender:version": "0",
        "Chooser:name": "slover-{}+cluster-1.1_1.2+searcher-bge+chooser-{}", 
        "Chooser:version": "0",
        # Text file args (ASR model output)
        "asroutput_path": "results_{}/Sensevoice/modelsoup.csv",
    }
    args['result_dir'] = args['result_dir'].format(update_args.dataset)
    args['asroutput_path'] = args['asroutput_path'].format(update_args.dataset)

    if update_args.step == 'Commender':
        # ASR output
        asr_data = pd.read_csv(args['asroutput_path'], header=0)
        # Cluster
        cluster_model = DishClusteringModel(category_weight=0.4, name_similarity_threshold=0.6)
        cluster_model.load(os.path.join(args['model_dir'], args['Cluster:model_id'], args['Cluster:version']))
        # Searcher
        embedder = SentenceModel(os.path.join(args['model_dir'], args['Searcher:model_id']))  # , device='cuda:6'

        # 食品库
        food_dictionary = pd.read_csv(args['food_dictionary'], sep='\t', header=0)
        food_set = set(food_dictionary['item_name'])
        corpus = food_dictionary['item_name'].unique().tolist()
        corpus_embeddings = embedder.encode(corpus)

        # 加载解析后的数据
        query_filedir = f"{args['result_dir']}/query"
        data = asr_data.copy()
        for llm in args['Solver:LLM']:
            output_file_path = os.path.join(query_filedir, f"{args['Solver:name']}-{llm}_{args['Solver:version']}" + '_output.jsonl')
            tmp = []
            with open(output_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    custom_id = item.get('custom_id', '')
                    content = item.get('response', {}).get('body', {}).get('choices', [{}])[0].get('message', {}).get('content', '')
                    content = content.strip().replace('【', '').replace('】', '')
                    tmp.append({'uuid': custom_id, f'food_from_query_{llm}': content})
            tmp = pd.DataFrame(tmp)
            data = pd.merge(data, tmp, on='uuid', how='left')
        
            # 生成推荐菜品池
            total_samples = len(data)
            result_df = []
            for i, row in tqdm(data.iterrows(), total=total_samples):
                # if row['dom'] == 0:  # 所有的text都要推荐
                #     continue
                initial_food_from_llm = row[f'food_from_query_{llm}']
                initial_food_in_dic = find_most_similar(initial_food_from_llm, food_set)
                food_pool = []

                # Cluster
                food_pool.extend(cluster_model.get_cluster(initial_food_in_dic, step=1.1))
                # print('1:', cluster_model.get_cluster(initial_food_in_dic, step=1.1))
                food_pool.extend(cluster_model.get_cluster(initial_food_in_dic, step=1.2))
                # print('2:', cluster_model.get_cluster(initial_food_in_dic, step=1.2))

                # Searcher
                query_embedding = embedder.encode([initial_food_from_llm])
                hits = semantic_search(query_embedding, corpus_embeddings, top_k=5)
                # print('3:', [corpus[hit['corpus_id']] for hit in hits[0]])
                food_pool.extend([corpus[hit['corpus_id']] for hit in hits[0]])

                # 取并集
                food_pool = set(food_pool)
                # print('4:', food_pool)
                result_df.append([row['uuid'], row['text'], initial_food_from_llm, initial_food_in_dic, list(food_pool)])

            result_df = pd.DataFrame(result_df, columns=['uuid', 'text', 'food_from_solver', 'food_from_solver-in_dic', 'food_pool'])
            os.makedirs(args['result_dir'], exist_ok=True)
            output_path = os.path.join(args['result_dir'], f"{args['Commender:name'].format(llm)}.csv")
            print(output_path)
            result_df.to_csv(output_path, index=False, encoding='utf-8')
            
    elif update_args.step == 'LLMChooser':
        # 食品库
        food_dictionary = pd.read_csv(args['food_dictionary'], sep='\t', header=0)
        food_set = set(food_dictionary['item_name'])

        for llm in args['Solver:LLM']:
            # 推荐池
            result_df = pd.read_csv(os.path.join(args['result_dir'], f"{args['Commender:name'].format(llm)}.csv"), header=0)

            # 加载选择后的数据
            query_filedir = f"{args['result_dir']}/query"
            output_file_path = os.path.join(query_filedir, f"{args['Commender:name'].format(llm)}_{args['Commender:version']}" + '_output.jsonl')
            print(output_file_path)
            data = []
            with open(output_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    custom_id = item.get('custom_id', '')
                    content = item.get('parsed_content', {})
                    data.append({'uuid': custom_id, 'food_from_chooser': content})
            data = pd.DataFrame(data)
            # data['in_dictionary'] = data['food_from_chooser'].apply(lambda x: 1 if x in food_set else 0)
            # print(data['in_dictionary'].value_counts())

            # merge
            merged_df = pd.merge(result_df, data, how='left', on='uuid')
            # merged_df['food_from_solver-in_dic'] = merged_df['food_from_solver-in_dic'].apply(extract_first_food)
            # merged_df['food'] = merged_df.apply(lambda row: row['food_from_chooser'] if row['in_dictionary'] == 1 else row['food_from_solver-in_dic'], axis=1)
            print(merged_df.head(10))
            output_path = os.path.join(args['result_dir'], f"{args['Chooser:name'].format(llm, llm)}.csv")
            print(output_path)
            merged_df.to_csv(output_path, index=False, encoding='utf-8')