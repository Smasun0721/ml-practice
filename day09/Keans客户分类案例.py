import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import calinski_harabasz_score, silhouette_score

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号


# 准备数据
data = pd.read_csv("./data/customers.csv")
x = data.iloc[:, -2:]
print(x)

# 训练模型
# sse_list=[]
# ch_list=[]
# sc_list=[]
# for k in range(2,20):
#     estimator = KMeans(n_clusters=k, random_state=0)
#     estimator.fit(x)
#     sse_list.append(estimator.inertia_)
#     ch_list.append(calinski_harabasz_score(x, estimator.predict(x)))
#     sc_list.append(silhouette_score(x, estimator.predict(x)))

estimator = KMeans(n_clusters=5, random_state=0)
y_pred = estimator.fit_predict(x)
print(y_pred)

# 数据可视化
x_ticks = [i for i in range(2, 20)]
plt.figure(figsize=(12, 8))

# plt.grid(True)
# plt.xticks(x_ticks)
# plt.plot(sse_list)
# plt.show()
#
# plt.grid(True)
# plt.xticks(x_ticks)
# plt.plot(ch_list)
# plt.show()
#
# plt.grid(True)
# plt.xticks(x_ticks)
# plt.plot(sc_list)
# plt.show()
#
# print(x_ticks)

cluster_names = ["普通客户", "VIP客户",  "理性型客户", "节俭型客户","冲动型客户",]
for i in range(5):
    mask = y_pred == i
    plt.scatter(
        x.iloc[mask, 0],
        x.iloc[mask, 1],
        s=50,
        alpha=0.7,
        label=cluster_names[i],  # ← 图例文字
    )
plt.scatter(
    estimator.cluster_centers_[:, 0], estimator.cluster_centers_[:, 1], color="black"
)
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("KMeans Clustering (k=5)")
plt.legend()
plt.show()
