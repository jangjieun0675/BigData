# -*- coding: utf-8 -*-
"""
Created on Sun May 26 17:05:41 2024

@author: ATIV
"""

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 데이터 로드
iris = load_iris()
X = iris.data
y = iris.target

# 훈련 세트와 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# n_estimators 값 설정
n_estimators_values = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

for n in n_estimators_values:
    # 모델 생성 및 훈련
    model = RandomForestClassifier(n_estimators=n, random_state=42)
    model.fit(X_train, y_train)
    
    # 예측 및 정확도 계산
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f'n_estimators: {n}, Accuracy: {accuracy}')
    
from sklearn import tree
from IPython.display import Image
import pydotplus

rfc = RandomForestClassifier(n_estimators=n, random_state=42)
rfc.fit(X_train, y_train)

# 특정 의사결정트리 선택
model = rfc.estimators_[5]

# 의사결정트리 시각화
dt_dot_data = tree.export_graphviz(model,
                                   feature_names = iris.feature_names,
                                   class_names = iris.target_names,
                                   filled = True, rounded = True,
                                   special_characters = True)
dt_graph = pydotplus.graph_from_dot_data(dt_dot_data)
Image(dt_graph.create_png())

import numpy as np

# 테스트 데이터 설정
myX_test = np.array([[5.6, 2.9, 3.6, 1.3]])

# 예측 수행
myprediction = rfc.predict(myX_test)

# 예측 결과 출력
print('다음 꽃의 종은:', iris.target_names[myprediction])




    
    
    
    
    
