import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

f = open("admission_data.csv", 'r')
rdr = csv.reader(f)

data = []
for line in rdr:
    data.append(np.array(line))

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

Ridge_RMSE_error_list = []
Lasso_RMSE_error_list = []

train_Ridge_RMSE_error_list = []
train_Lasso_RMSE_error_list = []

Ridge_best=[float('inf'), -1]
Lasso_best=[float('inf'), -1]


Linear = LinearRegression()
Linear.fit(X_train, y_train)
predict = Linear.predict(X_test)

Linear_RMSE_error = Linear.score(X_test, y_test)
Linear_best=[Linear_RMSE_error, Linear]
train_Linear_RMSE_error = Linear.score(X_train, y_train)
decompose = 15000

plt.plot((np.arange(200)) / decompose, [Linear_RMSE_error] * 200,
          label = "Test Linear Regression R2 score", c="orange")



for i in range(200):
    ridge = Ridge(alpha = i / decompose)
    ridge.fit(X_train, y_train)
    predict = ridge.predict(X_test)
    Ridge_RMSE_error = ridge.score(X_test, y_test)
    if Ridge_RMSE_error < Ridge_best[0]:
        Ridge_best = [Ridge_RMSE_error, i / decompose, ridge]
    Ridge_RMSE_error_list.append(Ridge_RMSE_error)
    train_Ridge_RMSE_error_list.append(ridge.score(X_train, y_train))

# plt.plot((np.arange(len(Ridge_RMSE_error_list))) / decompose,
#          Ridge_RMSE_error_list, label = "Test Ridge R2 score")

for i in range(200):
    lasso =Lasso(alpha = i / decompose)
    lasso.fit(X_train, y_train)
    predict = lasso.predict(X_test)
    Lasso_RMSE_error = lasso.score(X_test, y_test)
    if Lasso_RMSE_error < Lasso_best[0]:
        Lasso_best = [Lasso_RMSE_error, i / decompose, lasso]
    Lasso_RMSE_error_list.append(Lasso_RMSE_error)
    train_Lasso_RMSE_error_list.append(lasso.score(X_train, y_train))

plt.plot((np.arange(len(Lasso_RMSE_error_list))) / decompose,
         Lasso_RMSE_error_list, label = "Test Lasso R2 score", c="g")


def R_square(model):
    y_mean = np.mean(y_test)
    SST = np.sum((y_test - y_mean) ** 2)

    y_predict = model.predict(X_test)
    SSE = np.sum((y_predict - y_mean) ** 2)
    
    return SSE / SST

def count_non_zero_param(weight):
    n = len(weight)
    for i in range(len(weight)):
        if weight[i] == 0:
            n -= 1
    return n

# print("Linear Regression best performance coef : ", Linear_best[-1].coef_)
# print("Linear Regression best performance intercept : ", Linear_best[-1].intercept_)
print("Linear Regression best performance R_square : ", R_square(Linear_best[-1]))
print("Linear Regression best performance RMSE : ", Linear_best[0])


# print("Ridge best performance Hyperparameter alpha : ", Ridge_best[1])
# print("Ridge best performance coef : ", Ridge_best[-1].coef_)
# print("Ridge best performance intercept : ", Ridge_best[-1].intercept_)
print("Ridge best performance R_square : ", R_square(Ridge_best[-1]))
print("Ridge best performance RMSE : ", Ridge_best[0])

# print("Lasso best performance Hyperparameter alpha : ", Lasso_best[1])
# print("Lasso best performance coef : ", Lasso_best[-1].coef_)
# print("Lasso best performance intercept : ", Lasso_best[-1].intercept_)
print("Lasso best performance R_square : ", R_square(Lasso_best[-1]))
print("Lasso best performance RMSE : ", Lasso_best[0])
print("Lasso best performance the number of non zero parameter : " ,
      count_non_zero_param(Lasso_best[-1].coef_))


plt.xlabel("Regularization strength (alpha)")
plt.ylabel("R2_score")
plt.legend()
plt.show()

plt.plot((np.arange(200)) / decompose, [Linear_RMSE_error] * 200, label = "Train Linear Regression R2 score")
plt.plot((np.arange(len(train_Ridge_RMSE_error_list))) / decompose,
         train_Ridge_RMSE_error_list, label = "Train Ridge R2 score")
plt.legend()
plt.xlabel("Regularization strength (alpha)")
plt.ylabel("R2_score")
plt.show()

plt.plot((np.arange(200)) / decompose, [Linear_RMSE_error] * 200,
          label = "Train Linear Regression R2 score", c= "orange")
plt.plot((np.arange(len(train_Lasso_RMSE_error_list))) / decompose,
         train_Lasso_RMSE_error_list, label = "Train Lasso R2 score", c= 'g')
plt.legend()
plt.xlabel("Regularization strength (alpha)")
plt.ylabel("R2_score")
plt.show()