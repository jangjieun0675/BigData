#LogisticRegression_이진분류 -> X 값은 숫자 데이터, Y값 0과1

# Y값이 0과 1로 구분되면 -> LogisticRegression(이진분류) -> 출력 값이 0 또는 1인 두 가지 범주 중 하나에 속하는 경우
# Y값이 숫자(ex 키, 몸무게)이면 -> LinearRegression(선형회귀) -> 사람의 키나 몸무게와 같은 연속적인 값을 예측하는 경우
# Y값이 1,2,3 -> classification(분류) -> 고양이, 개, 또는 새 중 어떤 것을 나타내는지 예측하는 경우
# Y값이 없으면 -> cluster analysis(군집분석)->레이블이 없는 데이터를 비슷한 특성(비지도학습)

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

#만약 종속변수와 독립변수를 나눠야 한다면
#Y=X['Warehouse_block'] #종족변수
#X=X.drop(columns='Warehouse_block') #독립변수

# 분석에 필요하지 않은 컬럼 제거 => ID 항목 제거
X_ID = X.pop("ID")
Y_ID = Y.pop("ID")

# =============================================================================
# 라벨 인코딩 - 명목형 변수 => LabelEncoding(), get_dummies() 
label_enc = LabelEncoder()
X['Warehouse_block'] = label_enc.fit_transform(X['Warehouse_block'])
X['Mode_of_Shipment'] = label_enc.fit_transform(X['Mode_of_Shipment'])
X['Product_importance'] = label_enc.fit_transform(X['Product_importance'])
X['Gender'] = label_enc.fit_transform(X['Gender'])

#원 핫 인코딩
#categorical_cols = ['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender']
#X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# =============================================================================
# 피처로 사용할 데이터를 평균이 0, 분산이 1이 되는 정규 분포 형태로 맞춤
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =============================================================================
# train-test 검증 데이터 분리(split) 20%
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y['Target'], test_size=0.2, random_state=0)

# Logistic Regression 모델 생성 및 학습
log_reg = LogisticRegression()
log_reg.fit(X_train, Y_train)

# =============================================================================

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
#1에 가까울수록 좋음
# =============================================================================

#<<예측하기>>
#
# new_data = pd.DataFrame({
#     'Warehouse_block': ['B'],
#     'Mode_of_Shipment': ['Flight'],
#     'Customer_care_calls': [5],
#     'Customer_rating': [2],
#     'Cost_of_the_Product': [164],
#     'Prior_purchases': [4],
#     'Product_importance': ['high'],
#     'Gender': ['F'],
#     'Discount_offered': [16],
#     'Weight_in_gms': [3759]
# })
# 
# 
# # 각 명목형 변수에 대해 기존의 LabelEncoder를 사용하여 변환
# new_data['Warehouse_block'] = label_enc.fit_transform(new_data['Warehouse_block'])
# new_data['Mode_of_Shipment'] = label_enc.fit_transform(new_data['Mode_of_Shipment'])
# new_data['Product_importance'] = label_enc.fit_transform(new_data['Product_importance'])
# new_data['Gender'] = label_enc.fit_transform(new_data['Gender'])
# 
# # 기존에 학습된 StandardScaler를 사용하여 새로운 데이터도 스케일링
# new_data_scaled = scaler.fit_transform(new_data)
# 
# # 새로운 데이터 예측
# new_prediction = log_reg.predict(new_data_scaled)
# 
# print("새로운 데이터 예측 결과:", new_prediction)
# =============================================================================


# =============================================================================
# #예시1) 로지스틱 회귀 분석( 유방암 진단 )
# import numpy as np
# import pandas as pd
# 
# from sklearn.datasets import load_breast_cancer
# b_cancer = load_breast_cancer()
# print(b_cancer.DESCR) #데이터셋에 대한 설명
# 
# b_cancer_df = pd.DataFrame(b_cancer.data, columns = b_cancer.feature_names)
# # 유방암 유무를 확인할 컬럼 추가
# b_cancer_df['diagnosis']= b_cancer.target
# b_cancer_df.head()
# print('유방암 진단 데이터셋 크기 : ', b_cancer_df.shape)
# b_cancer_df.info()
# 
# # 피처로 사용할 데이터를 평균이 0, 분산이 1이 되는 정규 분포 형태로 맞춤
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# b_cancer_scaled = scaler.fit_transform(b_cancer.data)
# print(b_cancer.data[0])
# print(b_cancer_scaled[0])
# 
# #로지스틱 회귀 분석
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# # X, Y 설정하기
# Y = b_cancer_df['diagnosis']
# X = b_cancer_scaled 
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
# lr_b_cancer = LogisticRegression()
# lr_b_cancer.fit(X_train, Y_train)
# Y_predict = lr_b_cancer.predict(X_test)
# 
# #모델 분석
# from sklearn.metrics import confusion_matrix, accuracy_score
# from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
# confusion_matrix(Y_test, Y_predict) #혼동행렬(실제값, 예측값)
# 
# acccuracy = accuracy_score(Y_test, Y_predict)
# precision = precision_score(Y_test, Y_predict)
# recall = recall_score(Y_test, Y_predict)
# f1 = f1_score(Y_test, Y_predict)
# roc_auc = roc_auc_score(Y_test, Y_predict)
# 
# print('정확도: {0:.3f}, 정밀도: {1:.3f}, 재현율: {2:.3f},  F1: {3:.3f}'.format(acccuracy,precision,recall,f1))
# print('ROC_AUC: {0:.3f}'.format(roc_auc))
# =============================================================================


# =============================================================================
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




