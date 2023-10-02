import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.model_selection import train_test_split
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
svr_RMSE_list = []
svr_R2SCORE_C_list = []
svr_R2SCORE_epsilon_list = []

train_svr_R2SCORE_C_list = []
train_svr_R2SCORE_epsilon_list = []

decompose = 200

for i in range(200):
	svr = SVR(kernel='linear', C = (i + 1) / decompose)
	svr.fit(X_train, np.squeeze(y_train, axis = 1))

	y_pred = svr.predict(X_test).reshape(-1, 1)
	svr_R2SCORE_C_list.append(svr.score(X_test, y_test))
	train_svr_R2SCORE_C_list.append(svr.score(X_train, y_train))
        
for i in range(200):
	svr = SVR(kernel='linear', epsilon = (i + 1) / decompose)
	svr.fit(X_train, np.squeeze(y_train, axis = 1))

	y_pred = svr.predict(X_test).reshape(-1, 1)
	svr_R2SCORE_epsilon_list.append(svr.score(X_test, y_test))
	train_svr_R2SCORE_epsilon_list.append(svr.score(X_train, y_train))

        
plt.plot(np.arange(200) / decompose, svr_R2SCORE_C_list, label = "Test R2 score")
plt.plot(np.arange(200) / decompose, train_svr_R2SCORE_C_list, label = "Train R2 score")
# plt.plot(np.arange(200) / decompose, svr_R2SCORE_epsilon_list, label = "Test R2 score")
# plt.plot(np.arange(200) / decompose, train_svr_R2SCORE_epsilon_list, label = "Train R2 score")

# plt.plot(np.arange(200) / 10000, svr_RMSE_list, label = "RMSE error")
plt.legend()
plt.xlabel("Tuning parameter epsilon")
plt.ylabel("R2_score")
plt.show()

svr = SVR()
svr.fit(X_train, np.squeeze(y_train, axis = 1))

y_pred = svr.predict(X_test).reshape(-1, 1)

#print("SVR coefficient : ", svr.coef_)
print("SVR n_features : ", svr.n_features_in_)
print("SVR score: ", svr.score(X_test, y_test))