import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, rc

# 한글 폰트 설정
font_path = "c:/Windows/fonts/malgun.ttf"  # 폰트 경로 설정
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# 데이터 로드
file_path = 'forest_fire_data_20160101_20231231.csv'
data = pd.read_csv(file_path, encoding='cp949')

# 필요한 열 선택 및 결측치 제거
features = ['발생연도', '발생월', '발생시간', '발생장소_시도', '피해면적']
data = data[features + ['발생원인']].dropna()

# 발생시간을 분 단위로 변환하는 함수
def time_to_minutes(time_str):
    try:
        h, m, s = map(int, time_str.split(':'))
        return h * 60 + m
    except:
        return 0

# 발생시간을 분 단위로 변환
data['발생시간'] = data['발생시간'].apply(time_to_minutes)

# 특성과 타겟 변수 정의
X = data[features]
y = data['발생원인']

# 범주형 데이터 인코딩 및 스케일링 설정
categorical_features = ['발생장소_시도']
categorical_transformer = OneHotEncoder()
numerical_features = ['발생연도', '발생월', '발생시간', '피해면적']
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numerical_transformer, numerical_features)
    ]
)

# 데이터 전처리
X = preprocessor.fit_transform(X)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 기본 모델 정의
rf_clf = RandomForestClassifier(n_estimators=200, random_state=42)
gb_clf = GradientBoostingClassifier(n_estimators=200, random_state=42)
lr_clf = LogisticRegression(max_iter=1000, random_state=42)

# 투표 기반 앙상블 모델 정의
voting_clf = VotingClassifier(
    estimators=[('rf', rf_clf), ('gb', gb_clf), ('lr', lr_clf)],
    voting='soft'
)

# 모델 학습
voting_clf.fit(X_train, y_train)

# 예측
y_pred_train = voting_clf.predict(X_train)
y_pred_test = voting_clf.predict(X_test)

# 모델 평가
print("Classification Report for Training Set")
print(classification_report(y_train, y_pred_train))
print("Classification Report for Test Set")
print(classification_report(y_test, y_pred_test))

# 정확도, 정밀도, 재현율, F1 스코어, ROC AUC 스코어 계산
accuracy = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test, average='weighted')  # 다중 클래스의 경우 평균 방법을 지정해야 합니다.
recall = recall_score(y_test, y_pred_test, average='weighted')
f1 = f1_score(y_test, y_pred_test, average='weighted')
roc_auc = roc_auc_score(y_test, voting_clf.predict_proba(X_test), multi_class='ovr', average='weighted')

print('정확도: {0:.3f}, 정밀도: {1:.3f}, 재현율: {2:.3f}, F1: {3:.3f}'.format(accuracy, precision, recall, f1))
print('ROC AUC: {0:.3f}'.format(roc_auc))

# 혼동 행렬 시각화
conf_matrix = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=voting_clf.classes_, yticklabels=voting_clf.classes_)
plt.xlabel('예측된 값')
plt.ylabel('실제 값')
plt.title('혼동 행렬')
plt.show()
