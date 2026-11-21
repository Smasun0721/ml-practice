import numpy as np
from sklearn.neighbors import KNeighborsRegressor

x_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
y_train = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
x_test = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
estimator = KNeighborsRegressor(n_neighbors=3)
estimator.fit(x_train, y_train)
y_pre = estimator.predict(x_test)
print(y_pre)
print(type(y_pre))

