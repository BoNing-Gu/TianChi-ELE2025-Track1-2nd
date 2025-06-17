def replace_food_pattern(sub, match_key_word, new_food):
    # 找出 text 列匹配 match_key_word 的行
    mask = sub['text'].str.contains(match_key_word, na=False)
    matched_rows = sub[mask]
    
    # 打印被替换的原始行及数量
    print("原始被匹配的行：")
    print(matched_rows)
    print(f"总共匹配并将被替换的行数：{matched_rows.shape[0]}")
    print(f"替换为{new_food}\n")
    
    # 替换 food 列的内容
    sub.loc[mask, 'food'] = new_food
    
    return sub

def replace_food_exact(sub, match_key_word, new_food):
    # 找出 food 列中严格等于 match_key_word 的行
    mask = sub['food'] == match_key_word
    matched_rows = sub[mask]

    # 打印被替换的原始行及数量
    print("原始被匹配的行：")
    print(matched_rows)
    print(f"总共匹配并将被替换的行数：{matched_rows.shape[0]}")
    print(f"替换为 {new_food}\n")
    
    # 替换 food 列的内容
    sub.loc[mask, 'food'] = new_food

    return sub

import pandas as pd
text = pd.read_csv('results_B/Sensevoice/modelsoup.csv')
dom = pd.read_csv('results_B/EfficientNet/dom.csv')
food = pd.read_csv('results_B/Commender/gambler-final_one_from_four+chooser-dsr1.csv')
sub = text[['uuid', 'text']].copy()
sub = pd.merge(sub, dom[['uuid', 'dom']], on='uuid', how='left')
sub = pd.merge(sub, food[['uuid', 'food']], on='uuid', how='left')
columns = ['uuid', 'text', 'dom', 'food']
sub = sub[columns]
print(sub.head())
print(sub[(sub['dom'] == 1) & (sub['food'] == '无')])
print(sub['dom'].value_counts())
# sub.to_csv('sub_b/0514-text_by_modelsoup-dom_by_effnet-food_by_gambler-final_one_from_four+chooser-ds.txt', sep='\t', index=False, header=False, encoding='utf-8')

# 模糊意向固定推荐
sub = replace_food_pattern(sub, '软烂', '软糯小米红枣粥')
sub = replace_food_pattern(sub, '健康点的', '健康蔬菜粥')
sub = replace_food_pattern(sub, '炒嫩黄瓜', '紫苏炒黄瓜')
sub = replace_food_pattern(sub, '百合鲜笃', '银杏百合汤')
sub = replace_food_pattern(sub, '小炒牛肉配香米饭', '青菜鲜牛肉蛋炒饭')
sub = replace_food_pattern(sub, '蛋茄炖炒肉', '番茄炖牛腩')
sub = replace_food_pattern(sub, '杂粮养生饭', '五谷杂粮饭')
sub = replace_food_pattern(sub, '简单饭菜', '黑木耳炒肉简餐')
sub = replace_food_pattern(sub, '少放辣', '不辣热汤荞麦面')

# 同类食物升级
sub = replace_food_exact(sub, '暖心小米粥', '养胃暖心小米粥')
sub = replace_food_exact(sub, '小米玉米粥', '养胃暖心小米粥')
sub = replace_food_exact(sub, '养胃小米粥', '养胃暖心小米粥')
sub = replace_food_exact(sub, '番茄炒蛋份', '番茄炒鸡蛋')

# 检查意图识别
sub.loc[(sub['dom'] == 1) & (sub['food'] == '无'), 'dom'] = 0
print(sub['dom'].value_counts())

# 检查食品库
food_dictionary = pd.read_csv("data/智慧养老_label_B/dim_ai_exam_food_category_filter_out.txt", sep='\t', header=0)
food_set = set(food_dictionary['item_name'])
food_set.add('无')
mask = sub['food'].isin(food_set)
in_dict_count = mask.sum()
not_in_dict_count = (~mask).sum()
print(f"在食品库中的food数量：{in_dict_count}")
print(f"不在食品库中的food数量：{not_in_dict_count}")

sub.to_csv('sub_B/0516-text_by_modelsoup-dom_by_effnet-food_by_gambler-final_one_from_four+chooser-r1+postprocess.txt', sep='\t', index=False, header=False, encoding='utf-8')