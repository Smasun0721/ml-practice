import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

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
estimator = DecisionTreeClassifier(max_depth=10)
estimator.fit(x_train, y_train)

# 5.模型预测
y_pre = estimator.predict(x_test)
print(y_pre)

# 6.模型评估
report = classification_report(y_test, y_pre)
print(report)

# 7.绘制"决策树"图
plt.figure(figsize=(80,50))
plot_tree(estimator,filled=True,max_depth=10)
plt.savefig('tree2.png')
plt.show()