import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

f = open("./project/admission_data.csv", 'r')
rdr = csv.reader(f)

data = []
for line in rdr:
    data.append(np.array(line))

feature_name = data[0]
data = np.array(data)
data_data = data[1:, :-1]
data_target =  data[1:, -1]

data_data = data_data.astype(float)
data_target = data_target.astype(float).reshape(-1, 1)

scalerX = StandardScaler()
scalerY = StandardScaler()

univ_scaler = MinMaxScaler()

data_univ_ranking = data_data[:, 2].reshape(-1, 1)
research_binary = data_data[:, -1].reshape(-1, 1)
data_data = np.concatenate((data_data[:, :2], data_data[:, 3:-1]), axis = 1)

data_univ_ranking = univ_scaler.fit_transform(data_univ_ranking)

data_data = scalerX.fit_transform(data_data)
data_target = scalerY.fit_transform(data_target)

data_data = np.concatenate((data_data[:, :2], data_univ_ranking, data_data[:, 2:], research_binary), axis = 1)

print(data_data.shape)



X_train, X_test, y_train ,y_test = train_test_split(data_data,
                                                     data_target, test_size= 0.2, 
                                                       random_state = 8)

weight_list = []

for i in range(200):
    lasso =Lasso(alpha = i / 150)
    lasso.fit(X_train, y_train)
    weight_list.append(lasso.coef_)

weight_list = np.array(weight_list).T
print(weight_list.shape)

for i in range(7):
    plt.plot(np.arange(200) / 1500, weight_list[i], label = feature_name[i])

plt.legend()
plt.xlabel("Regularization strength (alpha)")
plt.ylabel("Coefficients")
plt.show()