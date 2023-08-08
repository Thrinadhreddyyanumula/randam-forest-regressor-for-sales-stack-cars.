# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 07:55:11 2023

@author: 91955
"""
import numpy as np
import pandas as pd
import
matplotlib.pyplot as plt
from sklearn.metrics import r2_score
data =
pd.read_csv(r"C:\Users\91955\AppData\Local\Temp\Rar$DIa8328.30807/mock_kaggle.csv")
data = data.rename(columns={'data':'Date', 'venda': 'Sales', 'estoque': 'Stock', 'preco':
'Price'})
newdata = data.drop(["Date"], axis=1)
X = newdata[['Stock',
'Price']]
y = newdata['Sales']
from sklearn.model_selection import train_test_split
X_train,
X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=51)
from
sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled =
scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Create and fit the
Random Forest Regressor model
from sklearn.ensemble import RandomForestRegressor
model =
RandomForestRegressor(n_estimators=100, max_depth=None,
max_features='auto')
model.fit(X_train_scaled, y_train)
y_pred =
model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
print("R-squared:",
r2)
n = len(y_test)
p = X_test_scaled.shape[1]
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n -
p - 1)
print("Adjusted R-squared:", adjusted_r2)
R-squared:
0.10271149504755461
Adjusted R-squared: 0.0971897196324627
Powered by TCPDF (www.tcpdf.org)