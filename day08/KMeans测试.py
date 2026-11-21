from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import calinski_harabasz_score

# 准备数据
x, y = make_blobs(
    n_samples=1000,
    n_features=2,
    centers=[[0, 0], [1, 1], [2, 1], [3, 3]],
    cluster_std=[0.3, 0.1, 0.3, 0.4],
    random_state=0,
)

# 绘制图像
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.show()

# 模型训练
km = KMeans(n_clusters=4)
y_pred = km.fit_predict(x)


plt.scatter(x[:, 0], x[:, 1], c=y_pred)
plt.show()

# 模型评估
score = calinski_harabasz_score(x, y_pred)
print(score)
