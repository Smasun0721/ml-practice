import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

# 1.加载数据
data = pd.read_csv("./data/train.csv")

# 2.数据预处理
x = data[["Pclass", "Sex", "Age"]]
y = data["Survived"]
# 填充Age缺失值
x["Age"] = x["Age"].fillna(x["Age"].mean())
# 将Sex字段转换类型
x["Sex"] = x["Sex"].replace(["male", "female"], [1, 0]).astype(int)

# 3.特征工程
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# 4.模型训练
estimator = DecisionTreeClassifier(random_state=42)
estimator.fit(x_train, y_train)
y_pred = estimator.predict(x_test)
print("单棵决策树的预测:")
print(y_pred)
print(classification_report(y_test, y_pred))
print("-" * 23)
gbc = GradientBoostingClassifier()
params_dict = {
    "n_estimators": [93, 94, 95, 96],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.048, 0.05, 0.049],
}
gs_estimator = GridSearchCV(estimator=gbc, param_grid=params_dict, cv=3, n_jobs=-1)
gs_estimator.fit(x_train, y_train)
y_pred = gs_estimator.predict(x_test)
print("GBDT预测结果:")
print(y_pred)
print(classification_report(y_test, y_pred))
print(f"最佳参数:{gs_estimator.best_params_}")
