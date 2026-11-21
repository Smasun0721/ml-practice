import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import plot_tree, DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LassoCV

# 1.导入数据
train_data = pd.read_csv("./data/红酒品质分类_train.csv", index_col=0)
test_data = pd.read_csv("./data/红酒品质分类_test.csv", index_col=0)

# 2.数据预处理
x_train = train_data.iloc[:, :-1]
x_test = test_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]
y_test = test_data.iloc[:, -1]
feature_names = x_train.columns

print("训练集分布:", y_train.value_counts().sort_index().to_dict())
print("测试集分布:", y_test.value_counts().sort_index().to_dict())

# 3.特征工程
# 特征标准化
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 利用L1正则化提取特征
lasso_cv = LassoCV(cv=5, fit_intercept=True)
lasso_cv.fit(x_train, y_train)
select_features = np.array(feature_names)[lasso_cv.coef_ > 1e-3]
print(lasso_cv.coef_)
print(f"提取特征:{select_features}")

x_train = pd.DataFrame(x_train, columns=feature_names)[select_features]
x_test = pd.DataFrame(x_test, columns=feature_names)[select_features]

# 模型训练
estimator = DecisionTreeClassifier()
ab_estimator = AdaBoostClassifier(estimator,learning_rate=0.1,n_estimators=500)
ab_estimator.fit(x_train, y_train)

# 5.模型预测
y_pre = ab_estimator.predict(x_test)
print(f"预测结果:{y_pre}")

# 6.模型评估
print(classification_report(y_test, y_pre))
