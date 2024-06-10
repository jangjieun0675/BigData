import pandas as pd

df=pd.read_csv("emp.csv")
data=df.groupby('job').agg({'job': 'count'})
data.sort_index(ascending=False)


df=pd.read_csv("emp.csv")
data=df.groupby('job').agg({'job': 'count'})
data.sort_index(ascending=False)


import pandas as pd
X=pd.read_csv("mtcars.csv")
X.info()
X_ID = X.pop("Unnamed: 0")
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)



TN = 2  # True Negative
FP = 1  # False Positive
FN = 0   # False Negative
TP = 3 # True Positive
accuracy = (TP + TN) / (TP + TN + FP + FN)
recall = TP / (TP + FN)
precision = TP / (TP + FP)
print(f'정확도(Accuracy): {accuracy}')
print(f'재현율(Recall, Sensitivity, True Positive Rate): {recall}')
print(f'정밀도(Precision): {precision}')



import pandas as pd
df=pd.read_csv("smoke.csv")
cond1=df.smoker=='no'; 
cond2=df.charges <= df.charges.quantile(.10);
df[cond1 & cond2]




















































































