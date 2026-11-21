from dask.array import hstack
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

# 准备数据
np.random.seed(42)
x = np.random.uniform(low=-3, high=3, size=100)
y = x**2 + 10 + x * 2 + np.random.normal(0, 2, size=100)

# 数据预处理
X = x.reshape(-1, 1)
X = np.hstack([X, X**2, X**3, X**4, X**5, X**6, X**7, X**8, X**9, X**10])
# 数据标准化
transfer = StandardScaler()
X = transfer.fit_transform(X)

# # 创建线性回归模型
# estimator = LinearRegression()
# 创建L1正则化对象
estimator = Lasso(alpha=0.1, max_iter=10000, tol=1e-4)
# # 创建L2正则化对象
# estimator=Ridge(alpha=10)
# 模型训练
estimator.fit(X, y)

# 模型预测与评估
y_pre = estimator.predict(X)
score = estimator.score(X, y)
print(f"均方误差:{mean_squared_error(y, y_pre)}")
print(f"准确率:{score:%}")
print(f"权重:{estimator.coef_}")

# 可视化评估
plt.scatter(x, y)  # 散点图
plt.plot(np.sort(x), y_pre[np.argsort(x)], color="red")  # 折线图
plt.show()
