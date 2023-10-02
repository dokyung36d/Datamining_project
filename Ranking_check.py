import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

f = open("admission_data.csv", 'r')
rdr = csv.reader(f)

univ_ranking_list = [0, 0, 0, 0, 0]

i = 0

for line in rdr:
    if i == 0:
        i = 1
        continue

    univ_ranking_list[int(line[-2]) - 1] += 1

print(univ_ranking_list)