import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from collections import defaultdict
from difflib import SequenceMatcher

class DishClusteringModel:
    def __init__(self, category_weight=0.5, name_similarity_threshold=0.7, distance_threshold_for_step1=0.4, distance_threshold_for_step2=0.6):
        """
        初始化聚类模型
        
        参数:
            category_weight: 类别信息在feature聚类中的权重(0-1)
            name_similarity_threshold: 第一步公共子串聚类中的名称相似度阈值
            distance_threshold_for_step1: 第一步feature聚类中的距离阈值
            distance_threshold_for_step2: 第二步feature聚类中的距离阈值
        """
        self.category_weight = category_weight
        self.name_similarity_threshold = name_similarity_threshold
        self.distance_threshold_for_step1 = distance_threshold_for_step1
        self.distance_threshold_for_step2 = distance_threshold_for_step2
        self.clusters = None
        self.dish_to_cluster = None
        self.cluster_to_dishes = None
        self.category_hierarchy = None
        self.name_vectorizer = None
        self.initial_clusters_by_string = []
        self.initial_clusters_by_feature = []
        self.final_clusters = []
        
    def _preprocess_categories(self, df):
        """预处理类别信息，构建类别层次结构"""
        # 填充空值
        df = df.fillna({'cate_1_name': '', 'cate_2_name': '', 'cate_3_name': ''})
        
        # 构建类别层次结构
        category_mapping = {}
        for _, row in df.iterrows():
            # 从最具体到最不具体的类别
            hierarchy = [
                # row['category_name'],
                row['cate_3_name'],
                row['cate_2_name'],
                row['cate_1_name']
            ]
            # 去除空字符串并保持顺序
            hierarchy = [c for c in hierarchy if c]
            category_mapping[row['item_name']] = hierarchy
            
        return category_mapping
    
    def _compute_category_similarity(self, hierarchy1, hierarchy2):
        """计算两个菜品类别层次结构的相似度"""
        if not hierarchy1 or not hierarchy2:
            return 0.0
            
        # 找到共同前缀的长度
        common_length = 0
        for c1, c2 in zip(hierarchy1, hierarchy2):
            if c1 == c2:
                common_length += 1
            else:
                break
                
        # 相似度为共同前缀长度与最大深度的比值
        max_depth = max(len(hierarchy1), len(hierarchy2))
        return common_length / max_depth
    
    def _find_common_substring_clusters(self, dish_names):
        """基于公共子串的初步聚类"""
        clusters = []
        used = set()
        
        for i, name1 in enumerate(dish_names):
            if name1 in used:
                continue
                
            current_cluster = [name1]
            used.add(name1)
            
            for j, name2 in enumerate(dish_names):
                if i == j or name2 in used:
                    continue
                    
                # 计算最长公共子串比例
                match = SequenceMatcher(None, name1, name2).find_longest_match()
                common_ratio = match.size / min(len(name1), len(name2))
                
                if common_ratio >= self.name_similarity_threshold:
                    current_cluster.append(name2)
                    used.add(name2)
                    
            if current_cluster:
                clusters.append(current_cluster)
                
        return clusters
    
    def _create_combined_similarity_matrix(self, df):
        """创建结合名称和类别的特征向量"""
        # 1. 基于名称的TF-IDF特征相似度矩阵
        self.name_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
        name_features = self.name_vectorizer.fit_transform(df['item_name'])  # [n_dishes, n_features]
        name_sim_matrix = linear_kernel(name_features, name_features)        # [n_dishes, n_dishes]
        
        # 2. 基于类别的层次结构相似度矩阵
        n_dishes = len(df)
        category_sim_matrix = np.zeros((n_dishes, n_dishes))  # [n_dishes, n_dishes]
        
        dish_names = df['item_name'].tolist()
        self.category_hierarchy = self._preprocess_categories(df)
        
        for i in range(n_dishes):
            for j in range(i, n_dishes):
                sim = self._compute_category_similarity(
                    self.category_hierarchy[dish_names[i]],
                    self.category_hierarchy[dish_names[j]]
                )
                category_sim_matrix[i, j] = sim
                category_sim_matrix[j, i] = sim
                
        # 3. 结合两种特征
        combined_sim_matrix = (1 - self.category_weight) * name_sim_matrix + \
                          self.category_weight * category_sim_matrix 
        # operands could not be broadcast together with shapes (81,407) (81,81) 
                          
        return combined_sim_matrix

    def _build_cluster_mappings(self):
        # 构建聚类映射
        self.initial_dish_to_cluster_by_string = {}
        self.initial_cluster_to_dishes_by_string = defaultdict(list)
        for i, cluster in enumerate(self.initial_clusters_by_string):
            for dish in cluster:
                self.initial_dish_to_cluster_by_string[dish] = i
                self.initial_cluster_to_dishes_by_string[i].append(dish)
        print(f"Step1 clusters [by common substring]: {len(self.initial_clusters_by_string)}")

        self.initial_dish_to_cluster_by_feature = {}
        self.initial_cluster_to_dishes_by_feature = defaultdict(list)
        for i, cluster in enumerate(self.initial_clusters_by_feature):
            for dish in cluster:
                self.initial_dish_to_cluster_by_feature[dish] = i
                self.initial_cluster_to_dishes_by_feature[i].append(dish)
        print(f"Step1 clusters [by common substring]: {len(self.initial_clusters_by_feature)}")

        self.final_dish_to_cluster = {}
        self.final_cluster_to_dishes = defaultdict(list)
        for i, cluster in enumerate(self.final_clusters):
            for dish in cluster:
                self.final_dish_to_cluster[dish] = i
                self.final_cluster_to_dishes[i].append(dish)
        print(f"Step2 clusters [by HAC]: {len(self.final_clusters)}")

    def string_cluster(self, df):
        """基于公共子串的聚类"""
        dish_names = df['item_name'].unique().tolist()
        clusters = self._find_common_substring_clusters(dish_names)
        return clusters
    
    def feature_cluster(self, df, distance_threshold):
        """基于特征相似度的聚类"""
        dish_names = df['item_name'].unique().tolist()
        # 创建结合名称和类别的特征
        similarity_matrix = self._create_combined_similarity_matrix(df)
        
        # 使用层次聚类
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric='precomputed',
            linkage='average',
            distance_threshold=distance_threshold
        )
        distance_matrix = 1 - similarity_matrix
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        # 根据聚类结果分组
        clusters = []
        for label in set(cluster_labels):
            sub_cluster = np.array(dish_names)[cluster_labels == label].tolist()
            clusters.append(sub_cluster)
        return clusters
    
    def fit(self, df):
        """
        对菜品进行聚类
        
        参数:
            df: 包含菜品信息的DataFrame，包含列:
                item_name, category_name, cate_1_name, cate_2_name, cate_3_name
        """
        # 第一步: 基于公共子串的聚类
        self.initial_clusters_by_string = self.string_cluster(df)

        # 第一步：基于特征的聚类
        self.initial_clusters_by_feature = self.feature_cluster(df, self.distance_threshold_for_step1)
        
        # 第二步: 对每个粗聚类进行更精细的层次聚类
        for cluster in self.initial_clusters_by_string:
            if len(cluster) == 1:
                self.final_clusters.append(cluster)
                continue
            # 获取当前聚类对应的数据
            cluster_df = df[df['item_name'].isin(cluster)]
            sub_clusters = self.feature_cluster(cluster_df, self.distance_threshold_for_step2)
            self.final_clusters.extend(sub_clusters)

        self._build_cluster_mappings()

    def save(self, path):
        """
        保存模型到指定路径
        会生成三个文件：path/step1_by_string.json, path/step1_by_feature.json 和 path/step2.json
        """
        path = Path(path)
        os.makedirs(path, exist_ok=True)
        # 保存第一步聚类结果（公共子串）
        with open(os.path.join(path, "step1_by_string.json"), "w", encoding='utf-8') as f:
            json.dump(self.initial_clusters_by_string, f, ensure_ascii=False, indent=2)

        # 保存第一步聚类结果（特征）
        with open(os.path.join(path, "step1_by_feature.json"), "w", encoding='utf-8') as f:
            json.dump(self.initial_clusters_by_feature, f, ensure_ascii=False, indent=2)
        
        # 保存第二步聚类结果（层次聚类）
        with open(os.path.join(path, "step2.json"), "w", encoding='utf-8') as f:
            json.dump(self.final_clusters, f, ensure_ascii=False, indent=2)

    def load(self, path):
        """
        从指定路径加载模型
        会自动加载三个文件：path/step1_by_string.json, path/step1_by_feature.json 和 path/step2.json
        """
        path = Path(path)
        # 加载聚类结果
        with open(os.path.join(path, "step1_by_string.json"), "r", encoding='utf-8') as f:
            self.initial_clusters_by_string = json.load(f)
        with open(os.path.join(path, "step1_by_feature.json"), "r", encoding='utf-8') as f:
            self.initial_clusters_by_feature = json.load(f)
        with open(os.path.join(path, "step2.json"), "r", encoding='utf-8') as f:
            self.final_clusters = json.load(f)
        
        # 重建查询字典
        self._build_cluster_mappings()
        return self

    def get_cluster(self, dish_name, step=2):
        """
        获取指定菜品所属的聚类
        
        参数:
            dish_name: 菜品名称
            
        返回:
            包含该菜品所在聚类的所有菜品名称列表
        """
        if step == 1.1:
            if dish_name not in self.initial_dish_to_cluster_by_string:
                return []
            cluster_id = self.initial_dish_to_cluster_by_string[dish_name]
            return self.initial_cluster_to_dishes_by_string[cluster_id]
        elif step == 1.2:
            if dish_name not in self.initial_dish_to_cluster_by_feature:
                return []
            cluster_id = self.initial_dish_to_cluster_by_feature[dish_name]
            return self.initial_cluster_to_dishes_by_feature[cluster_id]
        elif step == 2:
            if dish_name not in self.final_dish_to_cluster:
                return []
            cluster_id = self.final_dish_to_cluster[dish_name]
            return self.final_cluster_to_dishes[cluster_id]
        else:
            raise ValueError('Invalid step value. Must be 1.1, 1.2, or 2.')
    
    def get_all_clusters(self, step=2):
        """
        获取聚类
        
        返回:
            包含所有聚类的列表，每个聚类是一个菜品名称列表
        """
        if step == 1.1:
            return self.initial_clusters_by_string
        elif step == 1.2:
            return self.initial_clusters_by_feature
        elif step == 2:
            return self.final_clusters
        else:
            raise ValueError('Invalid step value. Must be 1.1, 1.2, or 2.')

if __name__ == '__main__':
    df = pd.read_csv('data/智慧养老_label/dim_ai_exam_food_category_filter_out.txt', sep='\t', header=0)
    df = df.drop(columns=['category_name'], inplace=False)
    print(df.head())
    # 初始化并训练模型
    model = DishClusteringModel(category_weight=0.4, name_similarity_threshold=0.6, distance_threshold_for_step1=0.5, distance_threshold_for_step2=0.6)
    model.fit(df)
    model.save('checkpoints/Cluster/3')
    # or model.load('checkpoints/Cluster')
    # 获取所有聚类
    clusters = model.get_all_clusters(step=1.2)
    print("所有聚类:")
    for i, cluster in enumerate(clusters):
        print(f"聚类 {i}: {cluster}")
    # 查询特定菜品的聚类
    test_dish = '一杯小米粥'
    cluster = model.get_cluster(test_dish, step=1.2)
    print(f"\n'{test_dish}' 所在的聚类: {cluster}")
