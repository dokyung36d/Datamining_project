# 한계
# 출처가 불명확


#### KNN-regression
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score


raw_data = pd.read_csv('admission_data.csv')
x_data = raw_data.iloc[:,[0,1,2,3,4,5,6]]
y_data = raw_data.iloc[:,[7]]

##scaling
# 대학랭킹은 minmax
# 나머지는 standard
# 연구경험은 scaling X



scaler_s = StandardScaler()
scaler_m = MinMaxScaler()
scaler_s.fit(x_data.iloc[:,[0,1,3,4,5]])
scaler_m.fit(x_data.iloc[:,[2]])

X_train_a = pd.DataFrame(scaler_s.transform(x_data.iloc[:,[0,1,3,4,5]]))
X_train_a.columns = ['GRE Score', 'TOEFL Score', 'SOP', 'LOR', 'CGPA']
X_train_b = pd.DataFrame(scaler_m.transform(x_data.iloc[:,[2]]))
X_train_b.columns = ['University Rating']
X_train_c = x_data.iloc[:,[6]]

x_data = pd.concat([X_train_a, X_train_b, X_train_c], axis = 1)

# split
X_train, X_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, random_state=8
)


training_accuracy = []
test_accuracy = []

k_dic = [1,3,5,7,9]
p_dic = [1,2,3,4,5,6,7,8,9,10]


# train
for k in k_dic:
    for p in p_dic:
        reg = KNeighborsRegressor(n_neighbors=k, p=p)
        reg.fit(X_train, y_train)
        training_accuracy.append(r2_score(y_train, reg.predict(X_train)))
    plt.plot(p_dic, training_accuracy, label = f'Train R2 Score k: {k}')
    training_accuracy = []

plt.title('Train')
plt.legend(loc=4)
plt.xlabel('Tuning Parameter p')
plt.ylabel('R2 Score')
plt.show()


# test
for k in k_dic:
    for p in p_dic:
        reg = KNeighborsRegressor(n_neighbors=k, p=p)
        reg.fit(X_train, y_train)
        test_accuracy.append(r2_score(y_test, reg.predict(X_test)))
    plt.plot(p_dic, test_accuracy, label = f'Test R2 Score k: {k}')
    test_accuracy = []

plt.legend(loc=4)
plt.title('Test')
plt.xlabel('Tuning Parameter p')
plt.ylabel('R2 Score')
plt.show()




