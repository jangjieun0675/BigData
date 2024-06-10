# 혼동 행렬을 보고 분석하기
TN = 2  # True Negative
FP = 1  # False Positive
FN = 0   # False Negative
TP = 3 # True Positive

# 정확도 계산
accuracy = (TP + TN) / (TP + TN + FP + FN)

# (2) Recall (Sensitivity, True Positive Rate)
recall = TP / (TP + FN)

# (3) Fall-out (False Alarm Ratio, False Positive Rate)
fall_out = FP / (FP + TN)

# (4) Specificity
specificity = TN / (TN + FP)

# (5) Precision
precision = TP / (TP + FP)

# (6) F1 Score
f1_score = 2 * (precision * recall) / (precision + recall)

print(f'정확도(Accuracy): {accuracy}')
print(f'재현율(Recall, Sensitivity, True Positive Rate): {recall}')
print(f'Fall-out (False Alarm Ratio, False Positive Rate): {fall_out}')
print(f'특이도(Specificity): {specificity}')
print(f'정밀도(Precision): {precision}')
print(f'F1 점수(F1 Score): {f1_score}')
# =============================================================================

# 선형회귀식에서 기울기와 절편을 구해보고 특정 x값의 결과를 추출하여라
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# 주어진 데이터
data = {'x': [59, 49, 75, 54, 78, 56, 60, 82, 69, 83, 88, 94, 47, 65, 89, 70],
        'y': [209, 180, 195, 192, 215, 197, 208, 189, 213, 201, 214, 212, 205, 186, 200, 204]}
data = pd.DataFrame(data)

# X와 Y로 분리
X = data[['x']]
Y = data['y']

# 회귀 모델 생성 및 학습
model = LinearRegression()
model.fit(X, Y)

# 회귀 계수 및 절편
slope = model.coef_[0] #기울기
intercept = model.intercept_ #절편

# 회귀분석 식 출력
print(f'회귀분석식: Y = {slope}X + {intercept}')

# 예측값 계산
Y_pred = model.predict(X)

# 회귀분석 결과 평가
mse = mean_squared_error(Y, Y_pred)
r2 = r2_score(Y, Y_pred)

print(f'평균 제곱 오차 (MSE): {mse}')
print(f'결정 계수 (R_squared 값): {r2}')

# X 값이 58일 때 Y 값 예측
new_X = np.array([[58]])
new_Y_pred = model.predict(new_X)
print(f'X 값이 58일 때 예측된 Y 값: {new_Y_pred[0]}')


# =============================================================================