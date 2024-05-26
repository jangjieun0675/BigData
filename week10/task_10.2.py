# -*- coding: utf-8 -*-
"""
Created on Sun May 26 15:54:57 2024

@author: ATIV
"""

# 필요한 라이브러리 불러오기
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score

# 데이터 불러오기
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

# 데이터를 학습 데이터와 테스트 데이터로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 로지스틱 회귀 모델 생성 및 학습
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# 모델을 테스트 데이터에 적용하여 예측값 생성
y_pred = model.predict(X_test)

# 모델 성능 평가
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'ROC AUC: {roc_auc:.2f}')
