#classification_분류 분석 -> X 값은 수치형 또는 범주형 데이터일 수 있지만, y 값은 항상 범주형 데이터
# DecisionTree와 RandomForest
import pandas as pd
data=pd.read_csv("mtcars.csv")
print(data)
print(data.info())

# =============================================================================
# (1) Decision Tree

# 필요없는 열 삭제 - 'Unnamed: 0' 자동차 이름
X=data.iloc[:, 1:]
X=X.dropna() #아예 결측값(null) 있는 행 제거
# 잘못된 값 바꾸기 '*3'->'3'
print(X['gear'].unique())
X['gear']=X['gear'].replace('*3', '3').replace('*5', 5)
print(X.info())

# 종속변수 am을 예측해야하므로 Y값 설정 후 X에서 제거
X['am']=X['am'].replace('manual',0).replace('auto', 1) 
Y=X['am']
X=X.drop(columns='am')
print(X.info())

# =============================================================================
# 학습데이터/테스트데이터 분리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(X, Y, test_size=0.3, random_state=10)

# =============================================================================
# DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(x_train, y_train)
y_test_predicted=model.predict(x_test)
# =============================================================================
#4 결과 분석
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score

acccuracy = accuracy_score(y_test, y_test_predicted) # 정확도
precision = precision_score(y_test, y_test_predicted) # 정밀도
recall = recall_score(y_test, y_test_predicted) #재현율
roc_auc = roc_auc_score(y_test, y_test_predicted)

print('정확도: {0:f}, 정밀도: {1:f}, 재현율: {2:f}'.format(acccuracy,precision,recall))
print('ROC_AUC: {0:f}'.format(roc_auc))

# =============================================================================
# =============================================================================

# (2) RandomForest
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(x_train, y_train)
y_test_predicted=model.predict(x_test)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
random_acccuracy = accuracy_score(y_test, y_test_predicted) # 정확도
random_precision = precision_score(y_test, y_test_predicted) # 정밀도
random_recall = recall_score(y_test, y_test_predicted) #재현율
random_roc_auc = roc_auc_score(y_test, y_test_predicted)

print('random_정확도: {0:f}, random_정밀도: {1:f}, random_재현율: {2:f}'.format(random_acccuracy,random_precision,random_recall))
print('random_ROC_AUC: {0:f}'.format(random_roc_auc))

# =============================================================================
# =============================================================================

# LogisticRegression
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(max_iter=1000) # lbfgs failed to converge
model.fit(x_train, y_train)
y_test_predicted=model.predict(x_test)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
Logistic_acccuracy = accuracy_score(y_test, y_test_predicted) # 정확도
Logistic_precision = precision_score(y_test, y_test_predicted) # 정밀도
Logistic_recall = recall_score(y_test, y_test_predicted) #재현율
Logistic_roc_auc = roc_auc_score(y_test, y_test_predicted)

print('Logistic_정확도: {0:f}, Logistic_정밀도: {1:f}, Logistic_재현율: {2:f}'.format(Logistic_acccuracy,Logistic_precision,Logistic_recall))
print('Logistic_ROC_AUC: {0:f}'.format(Logistic_roc_auc))

# 확률값 예측
print('****', model.predict_proba(x_test))

# 결과
print(y_test_predicted)
result=pd.DataFrame(y_test_predicted)
#result.to_csv('result.csv', index=False) #파일로 저장

# =============================================================================
# #예시1) DecisionTree 분류 분석( playtennis )
# from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn import tree
# 
# from IPython.display import Image
# 
# import pandas as pd
# import numpy as np
# get_ipython().system('pip install pydotplus')
# import pydotplus
# import os
# 
# tennis_data = pd.read_csv('playtennis.csv')
# 
# # 범주형 변수 변경(인코딩)
# tennis_data.Outlook = tennis_data.Outlook.replace('Sunny', 0)
# tennis_data.Outlook = tennis_data.Outlook.replace('Overcast', 1)
# tennis_data.Outlook = tennis_data.Outlook.replace('Rain', 2)
# 
# tennis_data.Temperature = tennis_data.Temperature.replace('Hot', 3)
# tennis_data.Temperature = tennis_data.Temperature.replace('Mild', 4)
# tennis_data.Temperature = tennis_data.Temperature.replace('Cool', 5)
# 
# tennis_data.Humidity = tennis_data.Humidity.replace('High', 6)
# tennis_data.Humidity = tennis_data.Humidity.replace('Normal', 7)
# 
# tennis_data.Wind = tennis_data.Wind.replace('Weak', 8)
# tennis_data.Wind = tennis_data.Wind.replace('Strong', 9)
# 
# tennis_data.PlayTennis = tennis_data.PlayTennis.replace('No', 10)
# tennis_data.PlayTennis = tennis_data.PlayTennis.replace('Yes', 11)
# 
# tennis_data
# 
# X = np.array(pd.DataFrame(tennis_data, columns = ['Outlook', 'Temperature', 'Humidity', 'Wind']))
# y = np.array(pd.DataFrame(tennis_data, columns = ['PlayTennis']))
# print(X)
# print(y)
# 
# X_train, X_test, y_train, y_test = train_test_split(X, y)
# 
# print(X_train)
# print(X_test)
# dt_clf = DecisionTreeClassifier()
# dt_clf = dt_clf.fit(X_train, y_train)
# dt_prediction = dt_clf.predict(X_test)
# dt_prediction
# print(confusion_matrix(y_test, dt_prediction))
# print(classification_report(y_test, dt_prediction))
# 
# feature_names = tennis_data.columns.tolist()
# feature_names = feature_names[0:4]
# target_name = np.array(['Play No', 'Play Yes'])
# 
# dt_dot_data = tree.export_graphviz(dt_clf, out_file = None,
#                                   feature_names = feature_names,
#                                   class_names = target_name,
#                                   filled = True, rounded = True,
#                                   special_characters = True)
# dt_graph = pydotplus.graph_from_dot_data(dt_dot_data)
# =============================================================================

# #예시2) RandomForest 분류 분석( iris )
# =============================================================================
# from sklearn.datasets import load_iris
# from sklearn import tree
# from sklearn.metrics import accuracy_score
# import numpy as np
# import pandas as pd
# 
# iris = load_iris()
# 
# import pandas as pd 
# df = pd.DataFrame(data=iris['data'], columns = iris['feature_names'])
# df.to_excel('iris.xlsx', index=False)
# 
# #training data 설정 -> 잘못 설정한 예
# x_train = iris.data[:-30]
# y_train = iris.target[:-30]
# #test data 설정
# x_test = iris.data[-30:] # test feature data  
# y_test = iris.target[-30:] # test target data
# 
# #RandomForestClassifier libary를 import
# from sklearn.ensemble import RandomForestClassifier # RandomForest
# #tree 의 개수 Random Forest 분류 모듈 생성
# rfc = RandomForestClassifier(n_estimators=10) 
# rfc
# rfc.fit(x_train, y_train)
# #Test data를 입력해 target data를 예측 
# prediction = rfc.predict(x_test)
# #예측 결과 precision과 실제 test data의 target 을 비교 
# print (prediction==y_test)
# #Random forest 정확도 츶정
# rfc.score(x_test, y_test)
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import classification_report
# print ("Accuracy is : ",accuracy_score(prediction, y_test))
# print ("=======================================================")
# print (classification_report(prediction, y_test))
# 
# from sklearn.model_selection import train_test_split
# x = iris.data
# y = iris.target
# X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)
# print (y_test)
# print (Y_test)
# clf = RandomForestClassifier(n_estimators=10) # Random Forest
# clf.fit(X_train, Y_train)
# prediction_1 = clf.predict(X_test)
# #print (prediction_1 == Y_test)
# print ("Accuracy is : ",accuracy_score(prediction_1, Y_test))
# print ("=======================================================")
# print (classification_report(prediction_1, Y_test))
# 
# # Initialize the model
# #트리 개수 200개
# clf_2 = RandomForestClassifier(n_estimators=200, # Number of trees
#                                max_features=4,    # Num features considered
#                                   oob_score=True)    # Use OOB scoring*
# clf_2.fit(X_train, Y_train)
# prediction_2 = clf_2.predict(X_test)
# print (prediction_2 == Y_test)
# print ("Accuracy is : ",accuracy_score(prediction_2, Y_test))
# print ("=======================================================")
# print (classification_report(prediction_2, Y_test))
# 
# for feature, imp in zip(iris.feature_names, clf_2.feature_importances_):
#     print(feature, imp)
#     
# import graphviz
# import os
# #os.environ['PATH'] += os.pathsep + 'c:\programdata\anaconda3\lib\site-packages'
# estimator = clf_2.estimators_[5]
# from sklearn.tree import export_graphviz
# export_graphviz(estimator, out_file='tree.dot', 
#                 feature_names = iris.feature_names,
#                 class_names = iris.target_names,
#                 rounded = True, proportion = False, 
#                 precision = 2, filled = True)
# 
# # 생성된 .dot 파일을 .png로 변환
# from subprocess import call
# call(['dot', '-Tpng', 'tree.dot', '-o', 'decistion-tree.png', '-Gdpi=50'])
# 
# # jupyter notebook에서 .png 직접 출력
# from IPython.display import Image
# 
# # 다음 명령어는 따로 실행해본다.
# Image(filename = 'decistion-tree.png')
# =============================================================================





# 다음 명령어는 따로 실행해본다.
#Image(dt_graph.create_png())
# =============================================================================
# 기타 분류 알고리즘
# SVM, KNN, MLPClassifier

# =============================================================================
# dir(pd)
# pd.read_csv.__doc__
# print(pd.read_csv.__doc__)
# print(pd.DataFrame.head.__doc__)
# print(data.info())
# =============================================================================


