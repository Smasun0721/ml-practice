import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,  # 分层K折交叉评估
    GridSearchCV,
)
from sklearn.metrics import classification_report
from sklearn.utils import compute_sample_weight

# 加载数据
data = pd.read_csv("./data/红酒品质分类.csv")
x = data.iloc[:, :-1]
# y = data.iloc[:, -1] - 3
y = pd.cut(data["quality"], bins=[0, 4, 6, 10], labels=[0, 1, 2]).astype(int)
data.info()

# 数据预处理
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0, stratify=y
)
print()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
print("训练集分布:", y_train.value_counts().sort_index().to_dict())
print("测试集分布:", y_test.value_counts().sort_index().to_dict())

# 模型训练
# 分层采样
spliter = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
# 参数列表
param_dict = {
    "n_estimators": [60, 65, 70],
    "learning_rate": [0.15, 0.2, 0.25],
    "max_depth": [4, 5, 6],
}
estimator = xgb.XGBClassifier(
    objective="multi:softmax", num_class=3
)  # 多分类问题，使用多分类模型
gs_estimator = GridSearchCV(
    estimator=estimator,
    param_grid=param_dict,
    cv=spliter,
    n_jobs=-1,
    verbose=True,
)
sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
gs_estimator.fit(x_train, y_train, sample_weight=sample_weights)

# 模型预测
y_pred = gs_estimator.predict(x_test)
print(f"预测结果:{y_pred}")

# 模型评估
print(classification_report(y_test, y_pred))
print(f"最优参数:{gs_estimator.best_params_}")
