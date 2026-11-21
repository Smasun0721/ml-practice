from sklearn.linear_model import LinearRegression

# 准备数据
x_train=[[1],[2],[4],[5],[3],[6]]
y_train=[1,2,4,5,3,6]

model=LinearRegression()
model.fit(x_train,y_train)

score = model.score(x_train, y_train)
print(score)
print(model.predict(x_train))