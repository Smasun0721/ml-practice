import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression  # 逻辑回归模型
from sklearn.preprocessing import StandardScaler  # 标准化
from sklearn.metrics import accuracy_score  # 预测精确率
from sklearn.model_selection import train_test_split  # 数据切分

# 1.加载数据
data = pd.read_csv("./data/breast-cancer-wisconsin.csv")

# 2.数据预处理
data = data.replace("?", np.nan).dropna()

# 3.特征工程
# 3.1.特征提取
x = data.iloc[:, :-1]
y = data.iloc[:, -1]
# 3.2.数据切分
train_x, test_x, train_y, test_y = train_test_split(
    x, y, test_size=0.2, random_state=10
)
# 3.3.数据标准化
transfer = StandardScaler()
train_x = transfer.fit_transform(train_x)
test_x = transfer.transform(test_x)

# 4.模型训练
estimator = LogisticRegression(fit_intercept=True)
estimator.fit(train_x, train_y)

# 5.模型预测
print(estimator.predict(test_x))

# 6.模型评估
accuracy = estimator.score(test_x, test_y)
print(f"准确率:{accuracy:%}")
