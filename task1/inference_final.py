from funasr import AutoModel
import os
import pandas as pd
from tqdm.auto import tqdm
from funasr.utils.postprocess_utils import postprocess_clean
#from sentence_transformers import SentenceTransformer
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
#import jieba
#from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
#import difflib
#import time
#import jiwer

# # FireRedASR-LLM
# model = FireRedAsr.from_pretrained("llm", "checkpoints/FireRedASR-LLM-L")
# results = model.transcribe(
#     batch_uttid,
#     batch_wav_path,
#     {
#         "use_gpu": 1,
#         "beam_size": 3,
#         "decode_max_len": 0,
#         "decode_min_len": 0,
#         "repetition_penalty": 3.0,
#         "llm_length_penalty": 1.0,
#         "temperature": 1.0
#     }
# )
# print(results)
#def task_three(result_use_1,result_use,recipe_use,recipe,vectorizer,tf_idf,model_gemma,tokenizer,PROMPT_TEMPLATE_FIRST,PROMPT_TEMPLATE_SECOND):
    # sentence_way_index = sentence_transformer_way(result,model_sentence,recipe)
    # print(sentence_way_index)
    # return recipe[sentence_way_index]
    #qtf_idf = vectorizer.transform([result_use_1])
    #tfidf_way_index = tfidf_way(qtf_idf ,tf_idf)
    #recommend_tfidf = recipe[tfidf_way_index]
    #food = model_inference(PROMPT_TEMPLATE_FIRST,result_use,model_gemma,tokenizer)
    # print(food)
    #recommend_difflib = find_most_similar(food,recipe)
    #if recommend_tfidf == recommend_difflib:
        #return recommend_tfidf
    #else:
        #user_input_new = result_use + "，候选菜品：" + recommend_tfidf + "，" + recommend_difflib
        # print(user_input_new)
        #food_new = model_inference(PROMPT_TEMPLATE_SECOND,user_input_new,model_gemma,tokenizer)
        #if food_new not in recipe:
            #return recommend_difflib
        #else:
            #return food_new

# def sentence_transformer_way(result,model_sentence,recipe):
#     embeddings_result = model_sentence.encode(result)
#     embeddings_recipe = model_sentence.encode(recipe)
#     similarities = model_sentence.similarity(embeddings_result, embeddings_recipe)
#     # print(torch.max(similarities))
#     return torch.argmax(similarities[0])

# def tfidf_way(qtf_idf ,tf_idf):
#     res = cosine_similarity(tf_idf, qtf_idf)
#     return np.argmax(res)

# def calculate_tfidf(recipe,vectorizer):
#     tf_idf = vectorizer.fit_transform(recipe) 
#     return tf_idf,vectorizer

# def jieba_tokenizer(text):
#     # 使用精确模式进行分词
#     return " ".join(jieba.cut(text))


# def load_model(model_path,):
#     quantization_config = BitsAndBytesConfig(load_in_4bit=True)
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         device_map='auto',
#         quantization_config = quantization_config
#     ).eval()
#     model = model.to(DEVICE)
#     return model, tokenizer

# def model_inference(PROMPT_TEMPLATE,user_input,model_gemma,tokenizer):
#     prompt = PROMPT_TEMPLATE.format(user_input=user_input)
#     # print(prompt)
#     inputs = tokenizer(text=prompt, return_tensors="pt").to(DEVICE)
#     outputs = model_gemma.generate(
#         **inputs,
#         max_new_tokens=20,
#         temperature=0.8,
#     )
#     # print(outputs)
#     generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
#     # print(generated_tokens)
#     food = tokenizer.decode(generated_tokens, skip_special_tokens=True,clean_up_tokenization_spaces=False)
#     print(food)
#     return food

# def find_most_similar(food, recipe):
#     if food in recipe:
#         return food
#     similarities = [(dish, difflib.SequenceMatcher(None, food, dish).ratio()) for dish in recipe]
#     most_similar_dish = max(similarities, key=lambda x: x[1])[0]
#     return most_similar_dish

#def caculate_cer(text1,text2):
    #transformations = jiwer.Compose([
        #jiwer.RemovePunctuation(),
        #jiwer.RemoveMultipleSpaces(),
        #jiwer.Strip(),
        #jiwer.ReduceToListOfListOfChars()
        #])
    
    #stats = jiwer.process_words(text1, 
                                #text2, 
                                #reference_transform=transformations, 
                                #hypothesis_transform=transformations)
    #error_count = stats.substitutions + stats.deletions + stats.insertions
    
    #return error_count

#def find_best_string_numpy(strlist, calculate_cer):
    #n = len(strlist)
    #error_matrix = np.zeros((n, n), dtype=int)
    #for i in range(n):
        #for j in range(i + 1, n): 
            #error = calculate_cer(strlist[i], strlist[j])
            #error_matrix[i, j] = error
            #error_matrix[j, i] = error 
    #error_sums = error_matrix.sum(axis=1)
    #best_index = np.argmin(error_sums)
    #best_string = strlist[best_index]
    #return best_string

if __name__ == "__main__":
    args = {
        # "model_1_dir": "./archive(1)/outputs0",
        # "model_2_dir": "./archive(1)/outputs1",
        # "model_3_dir": "./archive(1)/outputs2",
        # "model_4_dir": "./archive(1)/outputs3",
        # "model_5_dir": "./archive(1)/outputs4",
        "model_dir": "./outputs_1",
        "train_file": "./data/智慧养老_label/train.txt",
        "test_file": "./data/智慧养老_label/A.txt",
        "audio_dir": "./data/train_audio",  
        "test_audio_dir": "./data/A_audio",
        "result_dir": "./results/fireredasr",
        "logger_dir": "./logs/fireredasr",
        "Train": False,
        "name":"sensevoice_modelsoup_b",
        "dom_csv_file": "./results/fireredasr/dom.csv",
        "recipe_file": "./data/智慧养老_label/dim_ai_exam_food_category_filter_out.txt",
        # "MODEL_PATH": "qwen2.5-transformers-14b-instruct-v1"
    }
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if args['Train']:
        pass
    else:
        # model_gemma,tokenizer = load_model(args["MODEL_PATH"])
#         PROMPT_TEMPLATE_FIRST = """<|im_start|>system
# 你是一个专业的食品解析助手，请根据用户输入，严格按以下规则响应：
# 1. 只返回一个中文食物名称
# 2. 不要标点符号和额外解释
# 3. 如果出现食物名称，则回答中一定要出现这个名称
# 4. 输入明确食物时直接提取，模糊需求按口味推荐
# 例如，当用户输入：天猫精灵，要碗番茄面疙瘩汤。输出应该为：番茄面疙瘩汤<|im_end|>
# <|im_start|>user
# {user_input}<|im_end|>
# <|im_start|>assistant
# """
#         PROMPT_TEMPLATE_SECOND = """<|im_start|>system
# 你是一个专业的食品解析助手，请根据用户输入以及候选菜品，从候选菜品中选择推荐的菜品，
# 严格按以下规则响应：
# 1. 只返回一个中文食物名称
# 2. 不要标点符号和额外解释
# 例如，当用户输入：天猫精灵，要碗番茄面疙瘩汤，候选菜品：番茄面疙瘩汤，手擀番茄面汤。输出应该为：番茄面疙瘩汤<|im_end|>
# <|im_start|>user
# {user_input}<|im_end|>
# <|im_start|>assistant
# """
        df = pd.read_csv(args['test_file'], encoding='utf-8', sep='\t', header=0)
        df['wav_path'] = df['uuid'].apply(lambda x: os.path.join(args['test_audio_dir'], x + '.wav'))
        #print(df.loc[0,'wav_path'])
        df_recipe = pd.read_csv(args["recipe_file"],encoding='utf-8', sep='\t', header=0)
        # recipe = df_recipe["item_name"].tolist()
        # recipe_use = [jieba_tokenizer(rec) for rec in recipe]
        # # recipe = list(map(lambda x:"天猫精灵,来一份"+x,recipe))
        # vectorizer = TfidfVectorizer()
        # tfidf,vectorizer = calculate_tfidf(recipe_use,vectorizer)
        result_df = []
        # FireRedASR-LLM
        # model_1 = AutoModel(model=args["model_1_dir"])
        # model_2 = AutoModel(model=args["model_2_dir"])
        # model_3 = AutoModel(model=args["model_3_dir"])
        # model_4 = AutoModel(model=args["model_4_dir"])
        # model_5 = AutoModel(model=args["model_5_dir"])
        
        # model_sentence = SentenceTransformer("all-MiniLM-L6-v2")
        # model_list = [model_1,model_2,model_3,model_4,model_5] 
        # for row in tqdm(df.itertuples(), total=len(df)):
        #     result_list = []
        #     for model in model_list:
        #         res = model_1.generate(input=row.wav_path)
        #         result_first = postprocess_clean(res[0]["text"])
        #         result_list.append(result_first.replace('"',''))
        #     result = find_best_string_numpy(result_list,caculate_cer)
        #     # result_use = result.replace("天猫精灵，","")
        #     # result_use = result_use.replace("天猫精灵","")
        #     # result_use_1 = jieba_tokenizer(result_use)
        #     # print(result)
        #     # result_use = [result_use]
        #     # recommend = task_three(result_use_1,result_use,model_sentence,recipe_use,recipe,vectorizer,tfidf)
        #     # time_start = time.time()
        #     # recommend = task_three(result_use_1,result_use,recipe_use,recipe,vectorizer,tfidf,model_gemma,tokenizer,PROMPT_TEMPLATE_FIRST,PROMPT_TEMPLATE_SECOND)
        #     # print(time.time() - time_start)
        #     # print(recommend)
        #     # res = model_1.generate(input=row.wav_path)
        #     # result_first = postprocess_clean(res[0]["text"])
        #     # result = result_first.replace('"','')
        #     result_df.append([row.uuid, result])
        model = AutoModel(model=args["model_dir"])
        for row in tqdm(df.itertuples(), total=len(df)):
            res = model.generate(input=row.wav_path)
            result_first = postprocess_clean(res[0]["text"])
            result = result_first.replace('"','')
            result_df.append([row.uuid, result])
        

        result_df = pd.DataFrame(result_df, columns=['uuid', 'text'])
        # dom_df = pd.read_csv(args["dom_csv_file"], encoding='utf-8')
        # result_df_save = pd.merge(result_df,dom_df,how = "inner",on = "uuid")
        # result_df_save = result_df_save[['uuid', 'text','dom','recommend']]
        os.makedirs(args['result_dir'], exist_ok=True)
        result_df.to_csv(os.path.join(args['result_dir'], f"text-{args['name']}.txt"), index = False,sep = '\t',encoding='utf-8',header = 0)
