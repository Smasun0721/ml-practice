# from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error

import pandas as pd
import numpy as np

data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
raw_df = pd.read_csv(data_url, sep="\\s+", skiprows=22, header=None, storage_options=headers)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# 数据分割
x_train,x_test,y_train,y_test = train_test_split(data, target, test_size=0.2, random_state=42)
# 数据标准化处理
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 创建评估器对象
estimator= LinearRegression(fit_intercept=True)
# 模型训练
estimator.fit(x_train, y_train)

# 模型预测
y_pre = estimator.predict(x_test)
print(f'模型预测结果:{y_pre}')
# 模型评估
print(f'模型准确率:{estimator.score(x_test, y_test):%}')
print(f'均方误差:{mean_squared_error(y_test, y_pre)}')
print(f'均方根误差:{root_mean_squared_error(y_test, y_pre)}')
print(f'平均绝对误差:{mean_absolute_error(y_test, y_pre)}')