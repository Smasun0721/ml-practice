import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jieba  # 分词器
from sklearn.feature_extraction.text import CountVectorizer  # 词频统计包
from sklearn.naive_bayes import MultinomialNB  # 朴素贝叶斯

# 加载数据
data = pd.read_csv("./data/书籍评价.csv", encoding="gbk")


# 数据预处理
# 提取标签列
y = data["评价"].replace({"好评": 1, "差评": 0}).astype(int).values
print(f"标签列:{y}")

# 提取特征列
words = data["内容"].values.tolist()
print(f"特征列:{words}")

# 分词
words = [" ".join(jieba.lcut(word)) for word in words]
print(f"分词后:{words}")

# 加载终止词
with open("./data/stopwords.txt", encoding="utf-8") as f:
    stopwords = f.readlines()
    stopwords = list(set(stopword.strip() for stopword in stopwords))
print(f"终止词:{stopwords}")

# 统计词频创建词频矩阵
transfer = CountVectorizer(stop_words=stopwords)
x = transfer.fit_transform(words).toarray()
print(f"剩余词汇:{transfer.get_feature_names_out()}")
print(f"词频矩阵:{x}")


# 模型训练
x_train = x[:10]
y_train = y[:10]
x_test = x[10:]
y_test = y[10:]

estimator = MultinomialNB()
estimator.fit(x_train, y_train)

# 模型预测
y_pred = estimator.predict(x_test)
print(f"预测结果:{y_pred}")

# 模型评估
print(estimator.score(x_test, y_test))
