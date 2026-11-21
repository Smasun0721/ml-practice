from sklearn.neighbors import KNeighborsClassifier

x_train = [[1], [2], [3], [4]]
y_train = [0, 0, 1, 1]
x_test = [[0]]
estimator = KNeighborsClassifier(n_neighbors=2)
estimator.fit(x_train, y_train)
y_pre = estimator.predict(x_test)
print(y_pre)


