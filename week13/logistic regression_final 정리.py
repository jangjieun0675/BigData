#!/usr/bin/env python
# coding: utf-8

# Y값이 0과 1로 구분되면 -> LogisticRegression(이진분류)
# Y값이 숫자(ex 키, 몸무게)이면 -> LinearRegression(선형회귀)
# Y값이 1,2,3 -> classification(분류)
# Y값이 없으면 -> cluster analysis(군집분석)

# 분류분석 중 이진 분류( logistic regression )
# 오차 행렬에 기반한 성능 지표인 정밀도, 재현율, F1 스코어, ROC_AUC를 사용
#해당 문제는 LogisticRegression(이진분류), 시그모이드 함수
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

X = pd.read_csv("XX_train.csv")
Y = pd.read_csv("YY_train.csv")

# X_train 정보 확인 후 
X.info()

# 분석에 필요하지 않은 컬럼 제거 => ID 항목 제거
X_ID = X.pop("ID")
Y_ID = Y.pop("ID")

# 라벨 인코딩 - 명목형 변수 => LabelEncoding(), get_dummies() 
label_enc = LabelEncoder()
X['Warehouse_block'] = label_enc.fit_transform(X['Warehouse_block'])
X['Mode_of_Shipment'] = label_enc.fit_transform(X['Mode_of_Shipment'])
X['Product_importance'] = label_enc.fit_transform(X['Product_importance'])
X['Gender'] = label_enc.fit_transform(X['Gender'])

#categorical_cols = ['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender']
#X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# 피처로 사용할 데이터를 평균이 0, 분산이 1이 되는 정규 분포 형태로 맞춤
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# train-test 검증 데이터 분리(split) 20%
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y['Target'], test_size=0.2, random_state=0)

# Logistic Regression 모델 생성 및 학습
log_reg = LogisticRegression()
log_reg.fit(X_train, Y_train)


#모델 분석
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# 검증용 데이터에 대한 예측
Y_predict = log_reg.predict(X_test)

confusion_matrix(Y_test, Y_predict) #혼동행렬(실제값, 예측값)

acccuracy = accuracy_score(Y_test, Y_predict) # 정확도
precision = precision_score(Y_test, Y_predict) # 정밀도
recall = recall_score(Y_test, Y_predict) #재현율
f1 = f1_score(Y_test, Y_predict) #f1_score
roc_auc = roc_auc_score(Y_test, Y_predict)

print('정확도: {0:.3f}, 정밀도: {1:.3f}, 재현율: {2:.3f},  F1: {3:.3f}'.format(acccuracy,precision,recall,f1))
print('ROC_AUC: {0:.3f}'.format(roc_auc))

#오차 행렬( 혼동행렬, Confusion Matrix )
# 각 지표 계산하는 식
# 정확도(Accuracy) : (TP + TN)/(TP + TN + FP + FN)
# 정밀도(Precision) : TP / (TP + FP)
# 재현율(Recall) : TP / (TP + FN)
# F1 Score : 2 * (Precision * Recall) / (Precision + Recall)
# FPR : FP / (FP + TN)
# 특이도(specificity) : 1 - FPR = TN / (TN + FP)

#EX) 100명 중 8명의 암환자가 있을 때 진단을 통해 10이 암이라고 판정했다.
# 10명중 6명은 실제 암이고 4명은 암환자가 아니었다.
#            예측(N)      예측(P)       # 긍정을 긍정으로 올바르게 예측 : TP
#실제(N=90)   TN = 88     FP = 4        # 부정을 부정으로 올바르게 예측 : TN
#실제(P=10)   FN = 2      TP = 6        # 긍정을 부정으로 잘못 예측 : FN
                                        # 부정을 긍정으로 잘못 예측 : FP

# 1에 가까울수록 정확히 예측하고 구분한다는 것
#정확도(Accuracy): 모델이 올바르게 분류한 샘플의 비율입니다. 이 값이 1에 가까울수록 모델이 더 많은 샘플을 올바르게 분류한 것입니다.
#정밀도(Precision): 양성으로 예측된 샘플 중 실제로 양성인 샘플의 비율입니다. 이 값이 1에 가까울수록 모델이 더 적은 수의 거짓 양성(False Positive)을 만들었다는 것을 의미합니다.
#재현율(Recall)= 민감도(sensitivity): 실제 양성 샘플 중 모델이 양성으로 예측한 샘플의 비율입니다. 이 값이 1에 가까울수록 모델이 더 적은 수의 거짓 음성(False Negative)을 만들었다는 것을 의미합니다.
#F1 점수(F1 Score): 1에 가까울수록 모델의 정밀도와 재현율이 모두 높다는 것을 의미합니다.
#ROC_AUC:1에 가까울수록 모델이 양성 클래스와 음성 클래스를 잘 구분한다는 것을 의미합니다.




