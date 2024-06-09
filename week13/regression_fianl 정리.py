#회귀 분석_regression : X, Y값 모두 숫자데이터로 바꾸어줘야 함
import pandas as pd
data=pd.read_csv("mtcars.csv")
print(data)
print(data.info())

#상관계수 출력 : 문자도 같이 계산하여 출력
print(data.corr(numeric_only=True))

# =============================================================================
# X, Y 분할하기 => 종속변수와 독립변수 분할
X=data.drop(columns='mpg') #종족변수
Y=data['mpg'] #독립변수

# 필요없는 열 삭제 - 'Unnamed: 0' 자동차 이름(분석에 필요하지 않은 컬럼 제거)
X=X.iloc[:, 1:]
#X_0= X.pop("Unnamed: 0")

# 결측치(null)  확인 - 평균값으로 대치
X.isnull().sum()
X_cyl_mean=X['cyl'].mean() # cyl의 평균을 구함
X['cyl']=X['cyl'].fillna(X_cyl_mean) #cyl의 null값을 cyl행의 평균으로 바꿈

X.isnull().sum()
X_cyl_median=X['qsec'].median()
X['qsec']=X['qsec'].fillna(X_cyl_median)

X=X.dropna() #또는 아예 결측값(null) 있는 행 제거

# 잘못된 값 바꾸기 '*3'->'3'
print(X['gear'].unique())
X['gear']=X['gear'].replace('*3', '3').replace('*5', 5)
print(X.info())

# 'cyl' 최대치 이상값 처리
X_describe=X.describe()
X_iqr=X_describe.loc['75%']-X_describe.loc['25%']
X_iqrmax=X_describe.loc['75%']+(1.5*X_iqr)
X_cyl_list=X.index[X['cyl']>X_iqrmax['cyl']].tolist()
X.loc[X_cyl_list, 'cyl']=X_iqrmax['cyl']

# 'hp' 최소치 이상값 처리
X_describe=X.describe()
#X_iqr=X_describe.loc['75%']-X_describe.loc['25%']
X_hpmin=X_describe.loc['25%']-(1.5*X_iqr)
X_hp_list=X.index[X['hp']<X_hpmin['hp']].tolist()
X.loc[X_hp_list, 'hp']=X_hpmin['hp']

# =============================================================================
# 데이터타입 변경(인코딩)
# 'gear' 자료형을 정수형으로 변환 또는 Label Encoding, One-Hot Encoding
#X['gear'] = X['gear'].astype(int)
print(X.info())
X['gear']=X['gear'].astype('object')

# (1) 인코딩 - Label Encoding
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder() 
X['am'] = encoder.fit_transform(X['am'])
#X = encoder.fit_transform(X) #전체데이터 Label Encoding

# (2) 인코딩 - One-Hot Encoding
#X['am'] = pd.get_dummies(X['am'])
#X['am'] = pd.get_dummies(X['am'], drop_first=True)
X=pd.get_dummies(X, drop_first=True)

# (3) 인코딩 - replace
X['am'].replace('manual',0).replace('auto', 1) 
X=X.drop(columns='am')

# =============================================================================
# 피처로 사용할 데이터를 평균이 0, 분산이 1이 되는 정규 분포 형태로 맞춤
import sklearn
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X = scaler.fit_transform(X) #데이터 전체 스케일링 -> 모든 데이터 타입이 숫자일때 가능
#X[['qsec']] = scaler.fit_transform(X[['qsec']]) #부분만 스케일링

# =============================================================================
# 학습데이터/테스트데이터 분리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(X, Y, test_size=0.3, random_state=10)

# =============================================================================
# 모델학습 Linear Regression
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train, y_train)

# 선형회귀분석 : 평가 데이터에 대한 예측 수행 -> 예측 결과 Y_predict 구하기
y_predict = model.predict(x_test); 
y_train_predicted=model.predict(x_train)
y_test_predicted=model.predict(x_test)

# ## 4) 결과 분석
# RSME
from sklearn.metrics import r2_score
#print('r2_score =  : ',r2_score(y_train, y_train_predicted)) #훈련 데이터에 대한 모델의 성능
print('r2_score =  : ',r2_score(y_test, y_test_predicted)) #모델의 성능을 측정
from sklearn.metrics import mean_absolute_error
#print('MAE(mean_absolute_error) : ',mean_absolute_error(y_test, y_test_predicted))
from sklearn.metrics import mean_squared_error
import numpy as np
print('MSE(mean_squared_error) : ',mean_squared_error(y_test, y_test_predicted))
print('RMSE(Root mean_squared_error) : ',np.sqrt(mean_squared_error(y_test, y_test_predicted)))
print('Y 절편 값: ', model.intercept_)
print('회귀 계수 값: ', np.round(model.coef_, 1))

# =============================================================================
# 회귀 모델 성능 평가 기준
# 0에 가까울수록 좋은 성능 : MAE, MSE, RMSE
# 1에 가까울수록 좋은 성능 : R-Squared
# MAE : 실제값과 모델 예측값 차이를 절대값으로 변환한 후 평균낸 값
# MSE : 실제값과 모델 예측값 차이를 제곱한 후 평균낸 값
# RMSE : 예측 대상 크기의 영향을 많이 받아 MAE보다 특이치에 강하다
# R2_score : 독립 변수가 종속 변수를 얼마나 잘 설명해주는지 보여주는 지표

# =============================================================================
# <<예측하기>>
# print("연비를 예측하고 싶은 차의 정보를 입력해주세요.")
# 
# cylinders_1 = int(input("cylinders : "))
# displacement_1 = int(input("displacement : "))
# weight_1 = int(input("weight : "))
# acceleration_1 = int(input("acceleration : "))
# model_year_1 = int(input("model_year : "))
# 
# mpg_predict = model.predict([[cylinders_1, displacement_1, weight_1, acceleration_1 , model_year_1]])
# 
# print("이 자동차의 예상 연비(mpg)는 %.2f 입니다." %mpg_predict)
# =============================================================================





# =============================================================================
# 예시2) 주택가격 분석( 다중 선형 회귀 모델 )
# import numpy as np
# import pandas as pd
# 
# from sklearn.datasets import fetch_openml
# boston = fetch_openml(name='boston')
# 
# # ## 2) 데이터 준비 및 탐색
# # In[2]:
# print(boston.DESCR)
# 
# # In[3]:
# boston_df = pd.DataFrame(boston.data, columns = boston.feature_names)
# boston_df.head()
# 
# # In[4]:
# boston_df['PRICE'] = boston.target
# boston_df.head()
# boston_df.to_csv("./DATA/BostonHousing.csv", index=False)
# # In[6]:
# print('보스톤 주택 가격 데이터셋 크기 : ', boston_df.shape)
# 
# # In[7]:
# boston_df.info() # 데이터 type 확인
# boston_df['CHAS']=boston_df['CHAS'].astype('int') #category 티입을 int형으로 변환
# boston_df['RAD']=boston_df['RAD'].astype('int')
# # ## 3) 분석 모델 구축
# # In[8]:
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score
# 
# # In[10]:
# # X, Y 분할하기 => 종속변수와 독립변수 분할
# Y = boston_df['PRICE']
# X = boston_df.drop(['PRICE'], axis=1, inplace=False)
# 
# # In[10]:
# # 훈련용 데이터와 평가용 데이터 분할하기
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=156)
# 
# # In[11]:
# # 선형회귀분석 : 모델 생성
# lr = LinearRegression()
# 
# # In[12]:
# # 선형회귀분석 : 모델 훈련
# lr.fit(X_train, Y_train)
# 
# # In[13]:
# # 선형회귀분석 : 평가 데이터에 대한 예측 수행 -> 예측 결과 Y_predict 구하기
# Y_predict = lr.predict(X_test); 
# 
# # ## 4) 결과 분석 및 시각화
# # In[14]:
# mse = mean_squared_error(Y_test, Y_predict)
# rmse = np.sqrt(mse)
# 
# print('MSE : {0:.3f}, RMSE : {1:.3f}'.format(mse, rmse))
# print('R^2(Variance score) : {0:.3f}'.format(r2_score(Y_test, Y_predict)))
# 
# # In[15]:
# print('Y 절편 값: ', lr.intercept_)
# print('회귀 계수 값: ', np.round(lr.coef_, 1))
# 
# # In[16]:
# coef = pd.Series(data = np.round(lr.coef_, 2), index=X.columns)
# coef.sort_values(ascending = False)
# 
# # ## - 회귀 분석 결과를 산점도 + 선형 회귀 그래프로 시각화하기
# # In[17]:
# import matplotlib.pyplot as plt
# import seaborn as sns
# 
# # In[18]:
# fig, axs = plt.subplots(figsize=(16, 16), ncols=3, nrows=5)
# x_features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
# 
# for i, feature in enumerate(x_features):
#       row = int(i/3)
#       col = i%3
#       sns.regplot(x=feature, y='PRICE', data=boston_df, ax=axs[row][col])
# =============================================================================

# =============================================================================
# 예시3) 자동차 연비 예측
# # ## 1) 데이터 수집
# # In[4]:
# import numpy as np
# import pandas as pd 
# data_df = pd.read_csv('./DATA/auto-mpg.csv', header=0, engine='python')
# 
# # ## 2) 데이터 준비 및 탐색
# # In[5]:
# print(' 데이터셋 크기 : ', data_df.shape)
# data_df.head()
# 
# # #### - 분석하지 않을 변수 제외하기
# # In[4]:
# data_df = data_df.drop(['car_name', 'origin', 'horsepower'], axis=1, inplace=False)
# 
# # In[5]:
# print(' 데이터세트 크기 : ', data_df.shape)
# data_df.head()
# 
# # In[6]:
# data_df.info()
# 
# # ## 3) 분석 모델 구축
# # In[7]:
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score
# 
# # In[8]:
# # X, Y 분할하기
# Y = data_df['mpg']
# X = data_df.drop(['mpg'], axis=1, inplace=False)
# 
# # In[9]:
# # 훈련용 데이터와 평가용 데이터 분할하기
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
# 
# # In[10]:
# # 선형회귀분석 : 모델 생성
# lr = LinearRegression()
# 
# # In[11]:
# # 선형회귀분석 : 모델 훈련
# lr.fit(X_train, Y_train)
# 
# # In[12]:
# # 선형회귀분석 : 평가 데이터에 대한 예측 수행 -> 예측 결과 Y_predict 구하기
# Y_predict = lr.predict(X_test)
# 
# # ## 4) 결과 분석 및 시각화
# # In[13]:
# mse = mean_squared_error(Y_test, Y_predict)
# rmse = np.sqrt(mse)
# 
# print('MSE : {0:.3f}, RMSE : {1:.3f}'.format(mse, rmse))
# print('R^2(Variance score) : {0:.3f}'.format(r2_score(Y_test, Y_predict)))
# 
# # In[14]:
# print('Y 절편 값: ',  np.round(lr.intercept_, 2))
# print('회귀 계수 값: ', np.round(lr.coef_, 2))
# 
# # In[15]:
# coef = pd.Series(data=np.round(lr.coef_, 2), index=X.columns)
# coef.sort_values(ascending=False)
# 
# # ### - 회귀 분석 결과를 산점도 + 선형 회귀 그래프로 시각화하기
# # In[16]:
# import matplotlib.pyplot as plt
# import seaborn as sns
# 
# # In[17]:
# fig, axs = plt.subplots(figsize=(16, 16), ncols=3, nrows=2)
# 
# x_features = ['model_year', 'acceleration', 'displacement', 'weight', 'cylinders']
# plot_color = ['r', 'b', 'y', 'g', 'r']
# 
# for i, feature in enumerate(x_features):
#       row = int(i/3)
#       col = i%3
#       sns.regplot(x=feature, y='mpg', data=data_df, ax=axs[row][col], color=plot_color[i])
# 
# # ###   <<<< 연비 예측하기  >>>>
# # In[18]:
# print("연비를 예측하고 싶은 차의 정보를 입력해주세요.")
# 
# cylinders_1 = int(input("cylinders : "))
# displacement_1 = int(input("displacement : "))
# weight_1 = int(input("weight : "))
# acceleration_1 = int(input("acceleration : "))
# model_year_1 = int(input("model_year : "))
# 
# # In[19]:
# mpg_predict = lr.predict([[cylinders_1, displacement_1, weight_1, acceleration_1 , model_year_1]])
# 
# # In[20]:
# 
# print("이 자동차의 예상 연비(mpg)는 %.2f 입니다." %mpg_predict)
# =============================================================================















# random forest regressor
# =============================================================================
# from sklearn.ensemble import RandomForestRegressor
# model=RandomForestRegressor(random_state=10)
# model.fit(x_train, y_train)
# y_train_predicted=model.predict(x_train)
# y_test_predicted=model.predict(x_test)
# # RSME
# from sklearn.metrics import r2_score
# print(r2_score(y_train, y_train_predicted))
# print(r2_score(y_test, y_test_predicted))
# from sklearn.metrics import mean_absolute_error
# print(mean_absolute_error(y_test, y_test_predicted))
# from sklearn.metrics import mean_squared_error
# print(mean_squared_error(y_test, y_test_predicted))
# 
# =============================================================================
# =============================================================================
# dir(pd)
# pd.read_csv.__doc__
# print(pd.read_csv.__doc__)
# print(pd.DataFrame.head.__doc__)
# 
# print(data.head())
# print(data.shape)
# print(type(data))
# print(data.columns)
# print(data.describe())
# print(data['hp'].describe())
# print(data['gear'].unique())
# print(data['cyl'].unique())
# =============================================================================

# outlier 찾아보기
# =============================================================================
# def outlier(data, column):
#     mean=data[column].mean()
#     std=data[column].std()
#     lowest=mean-(std*1.5)
#     highest=mean+(std*1.5)
#     print('최소', lowest, '최대', highest)
#     outlier_index=data[column][ (data[column] < lowest) | (data[column] > highest)].index
#     return outlier_index
# print(outlier(X,'qsec'))
# X.loc[24,'qsec']=42.245
# #print(X.loc[outlier(X,'qsec'), 'qsec'])
# print(outlier(X,'carb'))
# X.loc[29,'carb']=5.235
# X.loc[30,'carb']=5.235
# #print(X.loc[outlier(X,'carb'), 'carb'])
# =============================================================================

# 데이터 스케일링
# =============================================================================
# dir(sklearn)
# print(sklearn.__doc__)
# sklearn.__all__ 
# import sklearn.preprocessing
# dir(sklearn.preprocessing)
# =============================================================================


# 피처로 사용할 데이터를 평균이 0, 분산이 1이 되는 정규 분포 형태로 맞춤
# =============================================================================
# import sklearn
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# X = scaler.fit_transform(X)
# X[['qsec']] = scaler.fit_transform(X[['qsec']]) #부분만 스케일링
# X = scaler.fit_transform(X) #데이터 전체 스케일링 -> 모든 데이터 타입이 숫자일때 가능
# =============================================================================
#temp=X[['qsec']]
#scaler.fit_transform(temp)
#qsec_s_scaler=pd.DataFrame(scaler.fit_transform(temp))
#print(qsec_s_scaler.describe())
#X['qsec']=qsec_s_scaler

#파생변수 만들기
# =============================================================================
# X['wt'] < 3.3
# X.loc[ X['wt'] < 3.3, 'wt_class']=0
# X.loc[ X['wt'] >= 3.3, 'wt_class']=1
# X['qsec_4']=X['qsec']*4
# X.drop(columns='qsec')

# =============================================================================
# dir(LinearRegression)
# print(sklearn.linear_model.LinearRegression.__doc__)
# print(model.intercept_)
# print(model.coef_)
# print(model.score(x_train, y_train))
# print(model.score(x_test, y_test))
# 
# =============================================================================
