# 7번
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# 1. 데이터 읽기
X=pd.read_csv("X.csv")
Y=pd.read_csv("y.csv")
X.info()
# 분석에 필요하지 않은 컬럼 제거 => ID 항목 제거
X_ID = X.pop("Unnamed: 0")
Y_ID = Y.pop("Unnamed: 0")


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
# 2. 모델 학습
model=RandomForestClassifier(n_estimators=200, # Number of trees
                               max_depth=5,    # Num features considered
                                  oob_score=True)

# 3. 예측 및 평가
model.fit(X_train, Y_train)
Y_predict=model.predict(X_test)

f1 = f1_score(Y_test, Y_predict) #f1_score
print(f1)
