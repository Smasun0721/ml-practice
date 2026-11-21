from sklearn.preprocessing import MinMaxScaler

x_train=[[1,23,5,45,4],
         [4,5,6,54,2],
         [7,8,9,5,65]]
x_train_new=MinMaxScaler().fit_transform(x_train)
print(x_train_new)