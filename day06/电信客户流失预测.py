from idlelib import replace

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

# 读取数据
data = pd.read_csv("./data/churn.csv")

# 数据预处理
# 字符串替换
data["gender"] = (
    data["gender"].replace({"Female": 0, "Male": 1}).astype(int)
)  # Female:0 , Male:1
data["Churn"] = data["Churn"].replace({"Yes": 1, "No": 0}).astype(int)  # Yes:1 , No:0


# 数据、标签分离
y = data["Churn"]
x = data.iloc[:, 1:]
feature_names = x.columns

# 数据切分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

# 数据标准化
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 利用L1正则提取特征列名
lasso_cv = LassoCV(cv=5, fit_intercept=True)
lasso_cv.fit(x_train, y_train)
select_features = np.array(feature_names)[lasso_cv.coef_ != 0]
print(lasso_cv.coef_)
print(f"提取特征:{select_features}")

x_train_selected = pd.DataFrame(x_train, columns=feature_names)[select_features]
x_test_selected = pd.DataFrame(x_test, columns=feature_names)[select_features]

# 训练模型
estimator = LogisticRegression(fit_intercept=True, class_weight="balanced").fit(
    x_train_selected, y_train
)

# 模型预测
y_pre = estimator.predict(x_test_selected)
print(y_pre)

# 模型评估
cm = confusion_matrix(y_test, y_pre)
cm_df = pd.DataFrame(
    cm,
    index=["Actual 0 (No Churn)", "Actual 1 (Churn)"],
    columns=["Predicted 0", "Predicted 1"],
)
print(f"混淆矩阵:\n{cm_df}")
print()
print(f"准确率:{accuracy_score(y_test, y_pre):%}")
print(f"f1值:{f1_score(y_test, y_pre, pos_label=1):%}")
print(f"精确值:{precision_score(y_test, y_pre, pos_label=1):%}")
print(f"召回率:{recall_score(y_test, y_pre, pos_label=1):%}")
print(f"分类评估报告:\n{classification_report(y_test, y_pre)}")
print()

