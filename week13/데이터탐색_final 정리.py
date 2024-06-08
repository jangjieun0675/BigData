# 데이터 탐색
import pandas as pd

#데이터프레임 만들기
df = pd.DataFrame([[60,61,62],[70,71,72],[80,81,82],[90,91,92]],
                  index=['1반','2반','3반','4반'],columns=['퀴즈1','퀴즈2','퀴즈3'])
#데이터 시리즈 만들기
col2 = pd.Series([3,2,0,1],name='oranges',index=['June', 'Robert', 'Lily','David'])

data=pd.read_csv('Ex_CEOSalary.csv', encoding='utf-8')
# In[2]: 컬럼, NUll값, 데이터 type 출력
data.info()
# In[3]:데이터 위에서부터 10개 출력
data.head(10)
#컬럼 출력 -> 인덱스 형식
print(data.columns)
# ## 1-1. 범주형 자료의 탐색
# In[4]:
data['industry'].value_counts() #컬럼별 갯수
data['industry'].count() # 총개수==행 갯수
print(data['industry'].unique())#값의 종류 출력
#get_ipython().run_line_magic('matplotlib', 'inline') 컬럼별 그래프 그리기
data['industry'].value_counts().plot(kind="pie")
# In[7]:
data['industry'].value_counts().plot(kind="bar")

import matplotlib.pyplot as plt 
data.hist(bins=50, figsize=(20,15)) # 히스토그램 그리기
# In[13]:
data['salary'].hist(bins=50, figsize=(5,5))# 컬럼별 히스토그램 그리기

#갯수, 평균, 표준편차, 최소값, 최대값, 중간값(50%), 시리즈 형태로 출력
data.describe()

#상관계수 출력 : 문자도 같이 계산하여 출력
data=pd.read_csv("mtcars.csv")
print(data.corr(numeric_only=True))

#그룹화 : groupby
#SELECT industry,
#       COUNT(salary) AS count,
#       AVG(salary) AS mean,
#       MIN(salary) AS min,
#       MAX(salary) AS max
#FROM data
#GROUP BY industry;
data.groupby('industry')[['salary']].describe()

#SELECT COUNT(*), SUM(sal) FROM Emp GROUP BY job;
df.groupby('JOB').agg({'JOB': 'count', 'SAL': 'sum'})

# 산업(industry)별 평균 확인
data.groupby('industry').mean()

# # 5. 결측치(null값) 처리
# In[45]:
import pandas as pd
data1=pd.read_csv('Ex_Missing.csv')
# ## 5-1. 결측치 확인 
# ### 가. 전체 및 변수별 결측 확인
# isnull(): 결측이면 True, 결측이 아니면 False 값 반환
pd.isnull(data1)
# notnull(): 결측이면 False, 결측이 아니면 True 값 반환
pd.notnull(data1)
# 변수(컬럼)별로 결측값 개수 확인: df.isnull().sum()
data1.isnull().sum()
# 특정 변수(컬럼)의 결측 아닌 값의 개수 확인: df.notnull().sum()
data1['salary'].notnull().sum()

# ## 5-2. 결측값 제거: dropna(), axis=0이면 행, axis=1이면 열 전체 제거
# ### 가. 결측값 있는 행(row/case) 제거
data_del_row=data1.dropna(axis=0)
data_del_row

# ### 다. 결측값 있는 특정 행 제거
data_del_row1=data1[['salary']].dropna()

# 결측값을 0을 대체
data_0 = data1.fillna(0)
# 결측값을 평균으로 대체
data_mean=data1.fillna(data.mean())
# sales의 결측값을 salary 값으로 대체
import numpy as np
data2=data1.copy()
data2['sales_new'] = np.where(pd.notnull(data2['sales']) == True, data2['sales'], data2['salary'])


# In[5-1]: SQL 질의문으로 나타내기
# csv파일 (data.csv) 생성하기
import pandas as pd
import numpy as np
data = {
    "메뉴":['아메리카노','카페라떼','카페모카', '바닐라라떼', '녹차', '초코라떼', '바닐라콜드브루'],
    "가격":[4100, 4600, 4600, 5100, 4100, 5000, 5100],
    "할인율":[0.5, 0.1, 0.2, 0.3, 0, 0, 0],
    "칼로리":[10, 180, 420, 320, 20, 500, 400],
    "원두":['콜롬비아', np.NaN, '과테말라', np.NaN, '한국', '콜롬비아', np.NaN],
    "이벤트가":[1900, 2300, np.NaN, 2600, np.NaN, 3000, 3200],
}
df = pd.DataFrame(data)

# 데이터 불러오기
df

# 문제 [1] 1개 조건, 할인율 > 0.2인 메뉴에 대한 정보를 보기
cond=df.할인율 > 0.2
df[cond]

# 문제 [2] 2개 조건, 할인율 > 0.2 AND 칼로리 < 200인 메뉴에 대한 정보를 보기
cond1=df.할인율 >= 0.2; cond2=df.칼로리 < 200;
df[cond1 & cond2]

# 문제 [3] 2개 조건, 할인율 > 0.2 OR 칼로리 < 200인 메뉴에 대한 정보를 보기
cond1=df.할인율 >= 0.2; cond2=df.칼로리 < 200;
df[cond1 | cond2]

# 문제 [4] 문자열 1개 조건, 원두 이름이 "콜롬비아"인 메뉴에 대한 정보를 보기
cond1=df.원두=='콜롬비아'
df[cond1]

# 문제 [5] 문자열 조건, 원두 이름이 "콜롬비아" AND 가격 < 4500인 메뉴에 대한 정보를 보기
cond1=df.원두=='콜롬비아'; cond2=df.가격 < 4500;
df[cond1 & cond2]

# 문제 [6] 컬럼별 결측치 갯수 확인
df.isnull().sum()

# 문제 [7] 결측값 채우기, 원두 결측치를 "코스타리카"로 채우기
df['원두']=df['원두'].fillna('코스타리카')

# 문제 [8] 결측값 채우기, '이벤트가'컬럼 결측치는 1900으로 결측치 채움
df['이벤트가']=df['이벤트가'].fillna(1900)

# 문제 [9] 데이터 값 변경하기 : 아메리카노 -> 룽고, 녹차 -> 그린티, replace 문 사용해보기
df.메뉴.replace('아메리카노', '룽고', inplace=True)
df.메뉴.replace('녹차', '그린티', inplace=True)

# 문제 [10] 이벤트가격 1900 -> 1500 변경하기, replace 사용
df.이벤트가.replace(1900, 1500, inplace=True)

# 문제 [11] 메뉴 "바닐라라떼"의 원두를 "과테말라"로 변경, df.loc 사용
df.loc[3, "원두"]="과테말라"

# 문제 [12] 이벤트가 전체를 1000으로 변경
df['이벤트가']=1000

# 문제 [13] 컬럼별 데이터 갯수 카운트, NULL 값은 제외됨, count() 사용, 기본값 axis=0
df.count()

# 행별 데이터 갯수 카운트, NULL 값은 제외됨, count() 사용, axis=1
df.count(axis=1)

# 문제 [14] 데이터 갯수 len() 함수 사용, shape 사용
len(df)
df.shape
df.shape[0]

# 가격 컬럼의 최대값 구하기
df.가격.max()

# 문제 [15] 가격 컬럼의 최소값 구하기
df.가격.min()

# 가격 컬럼의 평균값 구하기
df.가격.mean()

# 문제 [16] 가격 컬럼의 중앙값 구하기
df.가격.median()

# 합계
df.가격.sum()

# 가격 컬럼의 표준편차 구하기
df.가격.std()

# 분산
df.가격.var()

# 백분위수 알아보기
df['가격'].describe()

# 가격 컬럼의 하위 25% 값 구하기, quantile() 함수 사용
df.가격.quantile(.25)

# 가격 컬럼의 하위 25% 데이터 모두 보기
cond=(df.가격 <= df.가격.quantile(.25))
df[cond]

# 최빈값 구하기, mode(), 주의 : 결과는 Series이다.
df.원두.mode()[0]

# apply 함수 예시
def cal(x):
  if x>=300:
    return 'No'
  else:
    return 'Yes'
df['칼로리'].apply(cal)

# apply 적용해서 새로운 컬럼 생성 (칼로리 컬럼 활용)
df['먹어도될까요']=df['칼로리'].apply(cal)

#그룹(groupby)
# 원두별 가격 평균 구하기
df.groupby('원두').mean()

# (원두와 할인율)별 평균값 구하기
df.groupby(['원두', '할인율']).mean()

# (원두와 할인율)별 가격 평균값 구하기
group1= df.groupby(['원두', '할인율'])['가격'].mean()

# 1개 인덱스 형태로 리셋
group2=df.groupby(['원두', '할인율'])['가격'].mean().reset_index()

# group2에서 원두='콜롬비아' AND 할인율=0.0 의 가격
print(group2.index)
cond1= (group2.원두=='콜롬비아')
cond2= (group2.할인율==0.0)
group2[cond1 & cond2]['가격']

# 결측치 채우기 : '이벤트가' 컬럼 결측치는 이벤트가격 데이터 중 최소값으로 결측치 채움
m=df.이벤트가.min()
df.이벤트가.fillna(m, inplace=True)

# 결측치 채우기 :'원두' 컬럼 결측치는 원두 데이터 중 최빈값으로 결측치 채움
m=df.원두.mode()[0]
df.원두.fillna(m, inplace=True)

# 가격이 5000 이상인 데이터의 개수를 구하시오
cond=df.가격 >= 5000
len(df[cond])


# In[5]: SQL 질의문으로 나타내기

#[질의 3-1] emp.csv를 읽어서 DataFrame emp 만들기
df = pd.read_csv('emp.csv')

#[질의 3-2] SELECT * FROM Emp;
df

#[질의 3-3] SELECT ename FROM Emp;
df['ENAME']

#[질의 3-4] SELECT ename, sal FROM Emp;
df[['ENAME','SAL']]

#[질의 3-5] SELECT DISTINCT job FROM Emp;
df['JOB'].unique()

#[질의 3-6] SELECT * FROM Emp WHERE sal < 2000;
cond = df.SAL < 2000
df[cond]

#[질의 3-7] SELECT * FROM Emp WHERE sal BETWEEN 1000 AND 2000;
cond1 = df['SAL'] >= 1000
cond2 = df['SAL'] <= 2000
df[cond1 & cond2]

#[질의 3-8] SELECT * FROM Emp WHERE sal >= 1500 AND job= ‘SALESMAN’;
cond3 = df['SAL'] >= 1500
cond4 = df['JOB'] == 'SALESMAN'
df[cond3 & cond4]

#[질의 3-9] SELECT * FROM Emp WHERE job IN ('MANAGER', 'CLERK');
cond5 = df['JOB'] == 'MANAGER'
cond6 = df['JOB'] == 'CLERK'
df[cond5 | cond6]

#[질의 3-10] SELECT * FROM Emp WHERE job NOT IN ('MANAGER', 'CLERK');
cond7 = ~(df['JOB'].isin(['MANAGER', 'CLERK']))
df[cond7]

#[질의 3-11] SELECT ename, job FROM Emp WHERE ename LIKE 'BLAKE';
df.loc[df['ENAME'] == 'BLAKE', ['ENAME', 'JOB']]

#[질의 3-12] SELECT ename, job FROM Emp WHERE ename LIKE '%AR%';
cond8 = df[df['ENAME'].str.contains('AR')]
df[[cond8], ['ENAME', 'JOB']]

#[질의 3-13] SELECT * FROM Emp WHERE ename LIKE '%AR%' AND sal >= 2000;
df[(df['ENAME'].str.contains('AR')) & (df['SAL'] >= 2000)]

#[질의 3-14] SELECT * FROM Emp ORDER BY ename;
df.sort_values(by='ENAME')

#[질의 3-15] SELECT SUM(sal) FROM Emp;
df['SAL'].sum()

#[질의 3-16] SELECT SUM(sal) FROM Emp WHERE job LIKE 'SALESMAN';
df[df['JOB'] == 'SALESMAN']['SAL'].sum()

#[질의 3-17] SELECT SUM(sal), AVG(sal), MIN(sal), MAX(sal) FROM Emp;
df['SAL'].agg(['sum', 'mean', 'max', 'min'])

#[질의 3-18] SELECT COUNT(*) FROM Emp;
df.count()

#[질의 3-19] SELECT COUNT(*), SUM(sal) FROM Emp GROUP BY job;
df.groupby('JOB').agg({'JOB': 'count', 'SAL': 'sum'})

#[질의 3-20] SELECT * FROM Emp WHERE comm IS NOT NULL;
df[df['COMM'].notnull()]

#[질의 4-0] emp.csv를 읽어서 DataFrame emp 만들기
df = pd.read_csv('emp.csv')

#[질의 4-1] emp에 age 열을 만들어 다음을 입력하여라(14명) 
df['AGE']=[30,40,50,30,40,50,30,40,50,30,40,50,30,40]


#[질의 4-2] INSERT INTO Emp(empno, ename, job) Values (9999, ‘ALLEN’, ‘SALESMAN’)
df1 = pd.DataFrame({'EMPNO': [9999], 'ENAME': ['ALLEN'], 'JOB': ['SALESMAN']})
df2 = pd.concat([df, df1])

#[질의 4-3] emp의 ename=‘ALLEN’ 행을 삭제하여라
#(DELETE FROM emp WHERE ename LIKE ‘ALLEN’;)
df2 = df[df['ENAME'] != 'ALLEN']

#[질의 4-4] emp의 hiredate 열을 삭제하여라
#(ALTER TABLE emp DROP COLUMN hiredate;)
df2 = df2.drop('HIREDATE', axis=1)

#[질의 4-5] emp의 ename=‘SCOTT’의 sal을 3000으로 변경하여라
#(UPDATE emp SET sal=3000 WHERE ename LIKE ‘SCOTT’;
cond = df2['ENAME']=='SCOTT'
df2.loc[cond,'SAL']=3000

#[질의 5-1] emp의 sal 컬럼을 oldsal 이름으로 변경하여라. 
#(ALTER TABLE emp RENAME sal TO oldsal;)
df2 = df2.rename(columns={'SAL':'OLDSAL'})

#[질의 5-2] emp에 newsal 컬럼을 추가하여라, 값은 oldsal 컬럼값
#(ALTER TABLE emp ADD newsal …;)
df2['NEWSAL'] = df2['OLDSAL']

#[질의 5-3] emp의 oldsal 컬럼을 삭제하여라
#(ALTER TABLE emp DROP COLUMN oldsal;
df2 = df2.drop('OLDSAL', axis=1)



# ### pandas 제공 기술통계 함수 
# 
# - count:  NA 값을 제외한 값의 수를 반환 
# - describe:  시리즈 혹은 데이터프레임의 각 열에 대한 기술 통계 
# 
# - min, max: 최소, 최대값 
# - argmin, argmax:  최소, 최대값을 갖고 있는 색인 위치 반환 
# - idxmin, idxmanx:  최소 최대값 갖고 있는 색인의 값 반환 
# - quantile:  0부터 1까지의 분위수 계산 
# - sum: 합 
# - mean: 평균 
# - median: 중위값 
# - mad: 평균값에서 절대 평균편차 
# - var: 표본 분산 
# - std: 표본 정규분산 
# - skew: 표본 비대칭도 
# - kurt: 표본 첨도 
# - cumsum: 누적 합 
# - cummin, cummax: 누적 최소값, 누적 최대값 
# - cumprod: 누적 곱 
# - diff: 1차 산술차 (시계열 데이터 사용시 유용) 
# - pct_change: 퍼센트 변화율 계산 
# - corr: 데이터프레임의 모든 변수 간 상관관계 계산하여 반환
# - cov: 데이터프레임의 모든 변수 간 공분산을 계산하여 반환









