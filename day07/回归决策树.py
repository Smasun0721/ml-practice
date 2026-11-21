import numpy as np
import pandas as pd
from pandas.core.reshape import reshape
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# 1.准备数据
x_train = np.array(range(10)).reshape(-1, 1)
y_train = np.array([5.56, 5.7, 5.91, 6.4, 6.8, 7.05, 8.9, 8.7, 9, 9.05])

# 2.数据预处理

# 3.特征工程

# 4.模型训练
model1 = LinearRegression()
model2 = DecisionTreeRegressor(max_depth=1)
model3 = DecisionTreeRegressor(max_depth=3)
model4 = DecisionTreeRegressor(max_depth=5)

model1.fit(x_train, y_train)
model2.fit(x_train, y_train)
model3.fit(x_train, y_train)
model4.fit(x_train, y_train)

# 5.模型预测
x_test = np.arange(0, 10, 0.1).reshape(-1, 1)
y_pre1 = model1.predict(x_test)
y_pre2 = model2.predict(x_test)
y_pre3 = model3.predict(x_test)
y_pre4 = model4.predict(x_test)

# 6.模型评估


# 7.绘制"决策树"图
plt.scatter(x_train, y_train, c="gray")
plt.plot(x_test, y_pre1, c="red", label="model1")
plt.plot(x_test, y_pre2, c="blue", label="model2")
plt.plot(x_test, y_pre3, c="green", label="model3")
plt.plot(x_test, y_pre4, c="pink", label="model4")
plt.legend()
plt.xlabel("data")
plt.ylabel("target")

plt.show()
