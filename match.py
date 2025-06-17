import os
import sys
import json
import jieba
jieba.dt.tmp_dir = "./.jieba_cache"  # 设置缓存目录为当前文件夹下的 .jieba_cache
jieba.initialize()  # 重新初始化
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import difflib
import ast
import argparse
from text2vec import SentenceModel, cos_sim, semantic_search, BM25
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi


def calculate_tfidf(recipe,vectorizer):
    """计算TF-IDF向量"""
    tf_idf = vectorizer.fit_transform(recipe) 
    return tf_idf, vectorizer

def tfidf_way(qtf_idf, tf_idf):
    """TF-IDF相似度计算"""
    res = cosine_similarity(tf_idf, qtf_idf).flatten()  
    top3_indices = np.argsort(-res)[:3]  
    return top3_indices

def jieba_tokenizer(text):
    """使用精确模式进行分词"""
    return " ".join(jieba.cut(text))

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
    
def match(query, corpus, bge, tfidf, bm25, choose):
    # sentence_way_index = sentence_transformer_way(result,model_sentence,recipe)
    # print(sentence_way_index)
    # return recipe[sentence_way_index]
    if choose == "bge":
        query_embedding = bge.encode([query])
        hits = semantic_search(query_embedding, corpus_embeddings, top_k=3)
        hits = hits[0]
        foods = [corpus[hit['corpus_id']] for hit in hits]
    elif choose == "tfidf":
        result_use_1 = jieba_tokenizer(query)
        qtf_idf = vectorizer.transform([result_use_1])
        tfidf_way_index = tfidf_way(qtf_idf, tfidf)
        corpus = np.array(corpus)
        foods = corpus[tfidf_way_index].tolist()
    elif choose == "bm25":
        result_use_1 = jieba_tokenizer(query)
        doc_scores = bm25.get_scores(result_use_1)
        foods = bm25.get_top_n(result_use_1, corpus, n=3)
    return foods

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match")
    parser.add_argument("--step", "-s", 
                        type=str, 
                        choices=['Matcher', 'LLMChooser'], 
                        default="Matcher", 
                        help="control step")
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
        # Matcher args
        "model_dir": "checkpoints",
        "Matcher:name": "matcher-3ensemble-top3",
        "Matcher:version": "0",
        "Chooser:LLM": ["gpt"],
        "Chooser:name": "matcher-3ensemble-top3+chooser-{}",
        "Chooser:version": "0",
        # Text file args (ASR model output)
        "asroutput_path": "results_{}/Sensevoice/modelsoup.csv",
    }
    args['result_dir'] = args['result_dir'].format(update_args.dataset)
    args['asroutput_path'] = args['asroutput_path'].format(update_args.dataset)

    if update_args.step == 'Matcher':
        # ASR output
        asr_data = pd.read_csv(args['asroutput_path'], header=0)

        # 食品库
        food_dictionary = pd.read_csv(args['food_dictionary'], sep='\t', header=0)
        food_set = set(food_dictionary['item_name'])
        corpus = food_dictionary['item_name'].unique().tolist()
        recipe_use = [jieba_tokenizer(rec) for rec in corpus]

        # BGE Matcher
        bge = SentenceModel(os.path.join(args['model_dir'], "text2vec-bge-large-chinese"))
        corpus_embeddings = bge.encode(corpus)
        # TF-IDF Matcher
        vectorizer = TfidfVectorizer()
        tfidf, vectorizer = calculate_tfidf(recipe_use, vectorizer)
        # BM25 Matcher
        bm25 = BM25Okapi(recipe_use)

        # 查询
        data = asr_data.copy()
        total_samples = len(data)
        result_df = []
        for i, row in tqdm(data.iterrows(), total=total_samples):
            query = row["text"]
            foods = []
            foods_bge = match(query, corpus, bge=bge, tfidf=tfidf, bm25=bm25, choose="bge")
            foods_tfidf = match(query, corpus, bge=bge, tfidf=tfidf, bm25=bm25, choose="tfidf")
            foods_bm25 = match(query, corpus, bge=bge, tfidf=tfidf, bm25=bm25, choose="bm25")
            foods.extend(foods_bge)
            foods.extend(foods_tfidf)
            foods.extend(foods_bm25)
            foods = list(set(foods))
            result_df.append([row['uuid'], query, foods])
            # print(query)
            # print("/t")
            # print(foods)

        result_df = pd.DataFrame(result_df, columns=['uuid', 'text', 'food_pool'])
        os.makedirs(args['result_dir'], exist_ok=True)
        result_df.to_csv(os.path.join(args['result_dir'], f"{args['Matcher:name']}.csv"), index=False, encoding='utf-8')

    elif update_args.step == 'LLMChooser':
        # 推荐池
        result_df = pd.read_csv(os.path.join(args['result_dir'], args['Matcher:name'] + '.csv'), header=0)

        # 食品库
        food_dictionary = pd.read_csv(args['food_dictionary'], sep='\t', header=0)
        food_set = set(food_dictionary['item_name'])

        for llm in args['Chooser:LLM']:
            # 加载选择后的数据
            query_filedir = f"{args['result_dir']}/query"
            output_file_path = os.path.join(query_filedir, f"{args['Matcher:name']}_{args['Matcher:version']}" + '_output.jsonl')
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
            # merged_df['food_from_3-matcher'] = merged_df['food_pool'].apply(extract_first_food)
            # merged_df['food'] = merged_df.apply(lambda row: row['food_from_chooser'] if row['in_dictionary'] == 1 else row['food_from_3-matcher'], axis=1)
            print(merged_df.head(10))
            output_path = os.path.join(args['result_dir'], f"{args['Chooser:name'].format(llm)}.csv")
            print(output_path)
            merged_df.to_csv(output_path, index=False, encoding='utf-8')