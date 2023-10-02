###############################################
## GBM test : 0.76이 best
# n_estimators : 높게 하면 train에 대해 overfitting되지만 일반화 성능 하락
# subsample
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import plot_tree

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


# train
reg = GradientBoostingRegressor(n_estimators= 100, subsample=0.8, min_samples_leaf=4, max_depth=3,  learning_rate=0.8, random_state=256)
reg.fit(X_train, y_train)
training_accuracy.append(reg.score(X_train, y_train))


#test
reg = GradientBoostingRegressor(n_estimators= 100,  subsample=0.8, min_samples_leaf=4, max_depth=3,  learning_rate=0.8, random_state=256)
reg.fit(X_train, y_train)
test_accuracy.append(reg.score(X_test, y_test))


# visualization
# 어느 변수가 작용하였는지 Tree화
n_estimator = len(reg.estimators_)
fig = plt.figure(figsize=(70, 50), facecolor='white')

row_num = 2
col_num = 2

x = [1, 4, 50, 200]
j = 0

for i in range(n_estimator):
    if i + 1 in x:
        ax = fig.add_subplot(row_num, col_num, j + 1)
        plot_tree(reg.estimators_[x[j] - 1][0],
              feature_names=X_train.columns,  ## 박스에 변수 이름 표시
              ax=ax
              )
        ax.set_title(f'{x[j]} tree')
        j += 1
plt.show()