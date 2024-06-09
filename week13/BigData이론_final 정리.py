# =============================================================================
# ※ 빅데이터 분석 기초 지식
# (1) 범주형 변수(질적 자료) : 범주형은 모두 숫자로 바꿔야 데이터 저리가 가능하다
#
# 명목형 변수 : 순서 없음, 값을 구분하기 위한 변수 (성별, 혈액형, 국가, 직업)
# 서열형(순위형) 변수 : 순서 있음, 순위형 (학점, 제품 만족도)
# (2) 수치형 변수(양적 자료)
# 
# 구간형(정수형) 변수 : 등간형(Interval), 셀 수는 있지만 특정 구간이 존재하는 변수, 사칙연산 (년도, 발생횟수, 자녀수)
# 비율형 변수 : 연속형(continuous), 연속적인 값을 가지며 차이와 비율에 의미가 있는 변수(소득, 키, 몸무게)
# ※ 독립변수와 종속변수
# - 독립변수 : 어떤 실험에서 실험자가 직접 변경하는 변수(결과에 영향을 주는 변수들)
# - 종속변수 : 독립변수의 값이 변함에 따라 달라지는 수량(결과값)
# 
# ex) 용돈을 벌기 위해 집안일을 돕는다고 가정합시다. 집안일 한 개당 용돈 300원을 받는다.
# 이때, 독립변수는 집안일의 양이고 종속변수는 집안일을 해서 버는 용돈(독립변수 * 300)이다.
# 
# ※ 상관관계와 인과관계
# 양의 상관관계	하나의 변수가 증가할 때 다른 변수가 함께 증가하는 경우
# 음의 상관관계	하나의 변수가 증가할 때 다른 변수가 함께 감소하는 경우
# 무 상관관계	두 변수 간의 아무런 증감이 없는 경우
#  
# 
# - 인과관계 분석(회귀 분석) : 선행하는 변수가 후행하는 다른 변수의 원인이 되고 있다고 믿어지는 관계
# 
# ▶ 인과관계가 있다면  → 상관관계도 존재, 그러나 상관관계가 있다고 해서 인과관계가 존재한다고는 할 수 없다.
# 
# 종속변수가 1개, 독립변수가 1개 → 단변량 단순 선형 회귀 모델
# 종속변수가 1개, 독립변수가 2개 → 단변량 다중 선형 회귀 모델
# 종속변수가 2개, 독립변수가 1개 → 다변량 단순 선형 회귀 모델
# 종속변수가 2개, 독립변수가 2개 → 다변량 다중 선형 회귀 모델
# ※ 선형 회귀 분석 : 주어진 데이터가 이루는 하나의 선을 찾는 것
# 
# 위와 같은 수식을 나타내며 x라는 독립변수가 Y라는 종속변수에 주는 영향력을 나타낸다.
# 
# 해당 수식에서 a는 x가 변해도 Y에 영향을 주지 않는 회귀 계수를 말하며 E는 오차항 b는 X에 영향력을 주는 계수이다.
# 
# ※ 분석 평가 지표
# 회귀 모델 성능 평가 기준
# 
# 0에 가까울수록 좋은 성능 : MAE, MSE, RMSE
# 1에 가까울수록 좋은 성능 : R-Squared
# 
# MAE : 실제값과 모델 예측값 차이를 절대값으로 변환한 후 평균낸 값
# 
# MSE : 실제값과 모델 예측값 차이를 제곱한 후 평균낸 값
# 
# RMSE : 예측 대상 크기의 영향을 많이 받아 MAE보다 특이치에 강하다
# 
# R2_score : 독립 변수가 종속 변수를 얼마나 잘 설명해주는지 보여주는 지표

# y값에서 예측한 y값의 평균은 항상 0이다. 그러므로 제곱하여 SSE를 구해야한다.
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