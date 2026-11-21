from sklearn.preprocessing import StandardScaler
import numpy as np

x_train = np.array([[1, 23, 5, 45, 4],
                    [4, 5, 6, 54, 2],
                    [7, 8, 9, 5, 65]])
transfer = StandardScaler()
x_train_new = transfer.fit_transform(x_train)
print(x_train_new)
print(f"均值：{transfer.mean_}")
print(f"标准差：{transfer.scale_}")
print(f"方差：{transfer.var_}")
