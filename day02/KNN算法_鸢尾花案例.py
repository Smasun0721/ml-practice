# 导入工具包
from sklearn.datasets import load_iris  # 加载鸢尾花测试集的.
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV  # 分割训练集和测试集的
from sklearn.preprocessing import StandardScaler  # 数据标准化的
from sklearn.neighbors import KNeighborsClassifier  # KNN算法 分类对象
from sklearn.metrics import accuracy_score  # 模型评估的, 计算模型预测的准确率



# 1.加载数据并预览
def load_data_iris():
    iris_data = load_iris()
    print("前五条x_trains:")
    print(iris_data.feature_names)
    print(iris_data.data[:5])
    print("前五条y_trains:")
    print(iris_data.target_names)
    print(iris_data.target[:5])


# 2.数据可视化
def show_data_iris():
    iris_data = load_iris()
    iris_df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
    # 添加标签列
    iris_df["label"] = iris_data.target
    # 通过Seaborn绘制散点图
    sns.lmplot(data=iris_df, x="sepal width (cm)", y="sepal length (cm)", hue="label")
    plt.title("Iris Data")
    plt.rcParams["figure.dpi"] = 200
    plt.subplots_adjust(top=0.9)
    plt.show()


# 3.切分数据集
def split_train_test():
    iris_data = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris_data.data, iris_data.target, test_size=0.2, random_state=42
    )
    return x_train, x_test, y_train, y_test

# 4.训练模型
def iris_evaluate_test():
    # 加载数据集
    iris_data = load_iris()
    # 数据集预处理
    # 4.1.切分数据集
    x_train, x_test, y_train, y_test = train_test_split(
        iris_data.data, iris_data.target, test_size=0.2, random_state=42
    )
    # 4.2.标准化数据
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    # 4.3.设置网格搜索字典
    param_dict = {"n_neighbors": [i for i in range(1, 11)]}
    # 4.4.采用网格搜索 + 交叉验证的方式寻找最优超参
    estimator = GridSearchCV(KNeighborsClassifier(), param_grid=param_dict, cv=4)
    # 训练模型
    estimator.fit(x_train, y_train)
    # 模型评估
    print(f"最优评分:{estimator.best_score_ * 100}%")
    print(f"最优超参组合:{estimator.best_params_}")
    print(f"最优的评估器对象:{estimator.best_estimator_}")
    print(f"具体的交叉验证结果:{estimator.cv_results_}")

# 测试
if __name__ == "__main__":
    iris_evaluate_test()

