# -*- coding: utf-8 -*-
# 어느 범죄학 연구자가 인구밀도와 절도발생률 간의 관계를 연구하면서 다음의 16개 도시의 자료를 수집하였다. 
# X는 해당 도시의 단위면적당 인구밀도를, Y는 이전년도의 10만명당 절도범죄의 발생횟수를 조사한 것이다.
# 파이썬을 이용하여 회귀분석식을 구하고, 각자 회귀분석식에 대한 평가를 진행해보아라.
# 데이터:
# X:{ 59, 49, 75, 54, 78, 56, 60, 82, 69, 83, 88, 94, 47, 65, 89, 70}
# Y:{ 209, 180, 195, 192, 215, 197, 208, 189, 213, 201, 214, 212, 205, 186, 200, 204}


import warnings
warnings.filterwarnings(action='ignore') 

# 라이브러리 불러오기
from sklearn import linear_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 데이터를 준비
data = {'x' : [59, 49, 75, 54, 78, 56, 60, 82, 69, 83, 88, 94, 47, 65, 89, 70],
        'y' : [209, 180, 195, 192, 215, 197, 208, 189, 213, 201, 214, 212, 205, 186, 200, 204]}
data = pd.DataFrame(data)

# 선형회귀 모델을 생성하고 데이터를 학습
linear_regression = linear_model.LinearRegression()
linear_regression.fit(X = pd.DataFrame(data["x"]), y = data["y"])

# 선형 회귀식의 계수를 출력
print('a value = ', linear_regression.intercept_)
print('b value =', linear_regression.coef_[0])

# 예측값을 구하고, 잔차를 계산
prediction = linear_regression.predict(X = pd.DataFrame(data["x"]))
residuals = data["y"] - prediction
print(prediction)
print(residuals)

# SSE, SST, R_squared 값을 계산
SSE = (residuals**2).sum()
SST = ((data["y"]-data["y"].mean())**2).sum()
R_squared = 1 - (SSE/SST)
print('SSE = ', SSE)
print('SST = ', SST)
print('R_squared = ', R_squared)

# 회귀분석의 평가 지표를 출력
print('score = ', linear_regression.score(X = pd.DataFrame(data["x"]), y = data["y"]))
print('Mean_absolute_Error = ', mean_absolute_error(prediction, data['y']))
print('Mean_Squared_Error = ', mean_squared_error(prediction, data['y']))
print('r2_score = ', r2_score(prediction, data['y']))
print('RMSE = ', mean_squared_error(prediction, data['y'])**0.5)

# X값이 58일 때 Y값을 예측
y_pred = linear_regression.predict(np.array([58]).reshape((-1, 1)))
print("X가 58일 때의 Y 예측값은 ", y_pred[0], "입니다.")






