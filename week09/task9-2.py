# -*- coding: utf-8 -*-
"""
Created on Sun May  5 03:52:56 2024

@author: ATIV
"""

# 해당 데이터셋은 당뇨병 환자들의 정보를 담고 있는 데이터셋이다.
# 총 442명의 환자(행)와 10개의 변수로 이루어진 데이터이다.
# 10개의 변수 = 나이, 성별, BMI, 혈압, s1 ~ s6


from sklearn import linear_model
from sklearn import datasets
from sklearn.metrics import mean_squared_error
import pandas as pd

diabetes_data = datasets.load_diabetes()

X = pd.DataFrame(diabetes_data.data)
y = diabetes_data.target

linear_regression = linear_model.LinearRegression()
linear_regression.fit(X = pd.DataFrame(X), y = y)
prediction = linear_regression.predict(X = pd.DataFrame(X))

print('a value = ', linear_regression.intercept_)
print('b balue =', linear_regression.coef_)

residuals = y-prediction
SSE = (residuals**2).sum(); SST = ((y-y.mean())**2).sum()
R_squared = 1 - (SSE/SST)

print('R_squared = ', R_squared)
print('score = ', linear_regression.score(X = pd.DataFrame(X), y = y))
print('Mean_Squared_Error = ', mean_squared_error(prediction, y))
print('RMSE = ', mean_squared_error(prediction, y)**0.5)

#-----------------------------------------------------------------------------

from sklearn import linear_model
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd

# 데이터 로드
diabetes_data = datasets.load_diabetes()
X = pd.DataFrame(diabetes_data.data)
y = diabetes_data.target

# 데이터를 학습 데이터와 테스트 데이터로 분할
# 전체 데이터의 20%를 테스트 데이터로 사용
# random_state=42 -> 재현 가능한 결과를 얻기 위해 사용되는 랜덤 시드 값
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 선형 회귀 모델 생성 및 학습
linear_regression = linear_model.LinearRegression()
linear_regression.fit(X = pd.DataFrame(X_train), y = y_train)

# 테스트 데이터에 대한 예측 수행
prediction = linear_regression.predict(X = pd.DataFrame(X_test))

# 회귀 계수 출력
print('a value = ', linear_regression.intercept_)
print('b value =', linear_regression.coef_)

# 잔차 계산
residuals = y_test - prediction

# SSE, SST, R_squared 계산
SSE = (residuals**2).sum()
SST = ((y_test - y_test.mean())**2).sum()
R_squared = 1 - (SSE/SST)

# 결과 출력
print('R_squared = ', R_squared)
print('score = ', linear_regression.score(X = pd.DataFrame(X_test), y = y_test))
print('Mean_Squared_Error = ', mean_squared_error(prediction, y_test))
print('RMSE = ', mean_squared_error(prediction, y_test)**0.5)
