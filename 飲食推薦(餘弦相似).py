import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# 假設你有一個包含食譜和用戶喜好的數據集
recipes = [
    '炒青菜 青菜 油 蒜 鹽',
    '紅燒肉 豬肉 醬油 糖 薑',
    '番茄炒蛋 蛋番茄 油 鹽',
    '炒豆腐 豆腐 青椒 醬油',
]

user_preferences = [
    '青菜 豬肉',
    '豆腐 青椒',
]

# 創建詞袋模型
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(recipes)

# 將用戶喜好轉換為特徵向量
user_vector = vectorizer.transform(user_preferences)


df_tf = pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names_out())

df_tf1 = pd.DataFrame(user_vector.toarray(),columns=vectorizer.get_feature_names_out())

# 計算食譜和用戶特徵向量之間的餘弦相似度
cosine_similarities = cosine_similarity(X, user_vector).sum(axis=1)

# 根據相似度排序，推薦前幾個食譜
top_recommendations = np.argsort(cosine_similarities)[::-1][:2]

# 輸出推薦結果
for i ,idx in enumerate(top_recommendations):
    print(f'第{i+1}推薦:',recipes[idx])
