import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("mtcartrain.csv")

data.info()

# X, Y 분할하기 => 종속변수와 독립변수 분할
X=data.drop(columns='mpg') #종족변수
Y=data['mpg'] #독립변수

# 분석에 필요하지 않은 컬럼 제거
X=X.drop(columns='cyl')
X=X.drop(columns='disp')
X=X.drop(columns='hp')
X=X.drop(columns='qsec')
X=X.drop(columns='vs')
X=X.drop(columns='am')

X_test = pd.read_csv("mtcartest.csv")

# 분석에 필요하지 않은 컬럼 제거
X_test=X_test.drop(columns='cyl')
X_test=X_test.drop(columns='disp')
X_test=X_test.drop(columns='hp')
X_test=X_test.drop(columns='qsec')
X_test=X_test.drop(columns='vs')
X_test=X_test.drop(columns='am')

# train-test 검증 데이터 분리(split) 25%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)


# 모델학습 Linear Regression
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train, Y_train)
Y_predict_train=model.predict(X_test) # 훈련모델의 테스트 결과

from sklearn.metrics import mean_squared_error
mean_squared_error = mean_squared_error(Y_test, Y_predict_train)
print('MSE(mean_squared_error) : {0:.3f}'.format(mean_squared_error)) #5.782


# 새로운 데이터 예측
new_prediction = model.predict(X_test)
print("데이터 예측 결과:", new_prediction[0:5])
# 데이터 예측 결과: [30.51494956 17.29295267 28.04312105 19.35885383 18.84715435]



















































