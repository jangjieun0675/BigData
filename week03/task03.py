# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 14:02:53 2024

@author: ATIV
"""

import numpy as np
import pandas as pd

[질의 3-1] emp.csv를 읽어서 DataFrame emp 만들기
df = pd.read_csv('emp.csv')

[질의 3-2] SELECT * FROM Emp;
df

[질의 3-3] SELECT ename FROM Emp;
df['ENAME']

[질의 3-4] SELECT ename, sal FROM Emp;
df[['ENAME','SAL']]

[질의 3-5] SELECT DISTINCT job FROM Emp;
df['JOB'].unique()

[질의 3-6] SELECT * FROM Emp WHERE sal < 2000;
cond = df.SAL < 2000
df[cond]

[질의 3-7] SELECT * FROM Emp WHERE sal BETWEEN 1000 AND 2000;
cond1 = df['SAL'] >= 1000
cond2 = df['SAL'] <= 2000
df[cond1 & cond2]

[질의 3-8] SELECT * FROM Emp WHERE sal >= 1500 AND job= ‘SALESMAN’;
cond3 = df['SAL'] >= 1500
cond4 = df['JOB'] == 'SALESMAN'
df[cond3 & cond4]

[질의 3-9] SELECT * FROM Emp WHERE job IN ('MANAGER', 'CLERK');
cond5 = df['JOB'] == 'MANAGER'
cond6 = df['JOB'] == 'CLERK'
df[cond5 | cond6]

[질의 3-10] SELECT * FROM Emp WHERE job NOT IN ('MANAGER', 'CLERK');
cond7 = ~(df['JOB'].isin(['MANAGER', 'CLERK']))
df[cond7]

[질의 3-11] SELECT ename, job FROM Emp WHERE ename LIKE 'BLAKE';
df.loc[df['ENAME'] == 'BLAKE', ['ENAME', 'JOB']]

[질의 3-12] SELECT ename, job FROM Emp WHERE ename LIKE '%AR%';
cond8 = df[df['ENAME'].str.contains('AR')]
df[[cond8], ['ENAME', 'JOB']]

[질의 3-13] SELECT * FROM Emp WHERE ename LIKE '%AR%' AND sal >= 2000;
df[(df['ENAME'].str.contains('AR')) & (df['SAL'] >= 2000)]

[질의 3-14] SELECT * FROM Emp ORDER BY ename;
df.groupby('ENAME')

[질의 3-15] SELECT SUM(sal) FROM Emp;
df['SAL'].sum()

[질의 3-16] SELECT SUM(sal) FROM Emp WHERE job LIKE 'SALESMAN';
df[df['JOB'] == 'SALESMAN']['SAL'].sum()

[질의 3-17] SELECT SUM(sal), AVG(sal), MIN(sal), MAX(sal) FROM Emp;
df['SAL'].agg(['sum', 'mean', 'max', 'min'])


[질의 3-18] SELECT COUNT(*) FROM Emp;
df.count()

[질의 3-19] SELECT COUNT(*), SUM(sal) FROM Emp GROUP BY job;
df.groupby('JOB').agg({'JOB': 'count', 'SAL': 'sum'})

[질의 3-20] SELECT * FROM Emp WHERE comm IS NOT NULL;
df[df['COMM'].notnull()]


[질의 4-0] emp.csv를 읽어서 DataFrame emp 만들기
df = pd.read_csv('emp.csv')

[질의 4-1] emp에 age 열을 만들어 다음을 입력하여라(14명) 
df['AGE']=[30,40,50,30,40,50,30,40,50,30,40,50,30,40]


[질의 4-2] INSERT INTO Emp(empno, ename, job) Values (9999, ‘ALLEN’, ‘SALESMAN’)
df1 = pd.DataFrame({'EMPNO': [9999], 'ENAME': ['ALLEN'], 'JOB': ['SALESMAN']})
df2 = pd.concat([df, df1])

[질의 4-3] emp의 ename=‘ALLEN’ 행을 삭제하여라
(DELETE FROM emp WHERE ename LIKE ‘ALLEN’;)
df2 = df2[df2['ENAME'] != 'ALLEN']

[질의 4-4] emp의 hiredate 열을 삭제하여라
(ALTER TABLE emp DROP COLUMN hiredate;)
df2 = df2.drop('HIREDATE', axis=1)

[질의 4-5] emp의 ename=‘SCOTT’의 sal을 3000으로 변경하여라
(UPDATE emp SET sal=3000 WHERE ename LIKE ‘SCOTT’;
 cond = df2['ENAME']=='SCOTT'
 df2.loc[cond,'SAL']=3000

 
[질의 5-1] emp의 sal 컬럼을 oldsal 이름으로 변경하여라. 
(ALTER TABLE emp RENAME sal TO oldsal;)
df2 = df2.rename(columns={'SAL':'OLDSAL'})

[질의 5-2] emp에 newsal 컬럼을 추가하여라, 값은 oldsal 컬럼값
(ALTER TABLE emp ADD newsal …;)
df2['NEWSAL'] = df2['OLDSAL']

[질의 5-3] emp의 oldsal 컬럼을 삭제하여라
(ALTER TABLE emp DROP COLUMN oldsal;
 df2 = df2.drop('OLDSAL', axis=1)





















