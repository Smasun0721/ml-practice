from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import calinski_harabasz_score, silhouette_score

# 准备数据
x, y = make_blobs(
    n_samples=1000,
    n_features=2,
    centers=[[0, 0], [1, 1], [2, 1], [3, 3]],
    cluster_std=[0.3, 0.1, 0.3, 0.4],
    random_state=0,
)

see_list=[]
# 训练模型
for k in range(2,20):
    estimator = KMeans(n_clusters=k, random_state=0)
    y_pred = estimator.fit_predict(x)
    # 获取每个k值对应的sse值
    see_list.append(estimator.inertia_)
    # see_list.append(calinski_harabasz_score(x, y_pred))
    # see_list.append(silhouette_score(x, y_pred))

print(see_list)

# 数据可视化
plt.figure(figsize=(20,10))
plt.plot(see_list)
plt.scatter(range(len(see_list)),see_list)
plt.xticks(range(len(see_list)))
plt.grid(True)
plt.show()