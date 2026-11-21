import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# 1.加载数据
data = pd.read_csv('./data/train.csv')

# 2.数据预处理
x=data[['Pclass','Sex','Age']]
y=data['Survived']
# 填充Age缺失值
x['Age']=x['Age'].fillna(x['Age'].mean())
# 将Sex字段转换类型
x['Sex'] = x['Sex'].replace(['male', 'female'], [1, 0]).astype(int)

# 3.特征工程
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 4.模型训练
estimator = RandomForestClassifier()
params = {'max_depth':[3,5,7,9,11],'n_estimators':[50,100,150,300,500]}
gs_estimator = GridSearchCV(estimator, param_grid=params, cv=3)
gs_estimator.fit(x_train, y_train)
print(f'最佳参数是:{gs_estimator.best_params_}')

best_estimator = gs_estimator.best_estimator_
best_estimator.fit(x_train, y_train)

# 5.模型预测
y_pre = best_estimator.predict(x_test)
print(y_pre)

# 6.模型评估
print(classification_report(y_test, y_pre))

