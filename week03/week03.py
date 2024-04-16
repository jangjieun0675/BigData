# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 13:13:01 2024

@author: ATIV
"""

import pandas as pd

data = {
        'apple' : [3,2,0,1],
        'orange' : [0,3,7,2]
        }

purchases = pd.DataFrame(data)

purchases = pd.DataFrame(data, index=['June', 'Robert', 'Lily','David'])

col1 = pd.Series([3,2,0,1],name='apples')

col2 = pd.Series([3,2,0,1],name='oranges',index=['June', 'Robert', 'Lily','David'])
col2 = pd.Series([3,2,0,1],name='oranges')
                 
col1.name
col1.values

x=col1.values

purchases2 = pd.DataFrame(col1)

purchases4 = pd.concat([col1,col2], axis=1)

purchases4.columns

df = pd.DataFrame([[60,61,62],[70,71,72],[80,81,82],[90,91,92]],
                  index=['1반','2반','3반','4반'],columns=['퀴즈1','퀴즈2','퀴즈3'])

df['퀴즈1'] #Series
df.loc['2반']
df.loc['2반':'4반', '퀴즈1']
df.loc['2반':'4반', '퀴즈1':'퀴즈3']

df.iloc[2]
df.iloc[2:4,0:2]



import numpy as np
import pandas as pd

df = pd.DataFrame([[89.2,92.5,'B'],
                   [90.8,92.8,'A'],
                   [89.9,95.2,'A'],
                   [89.9,85.2,'C'],
                   [89.9,90.2,'B']],
                  columns=['중간고사','기말고사','성적'],
                  index=['1반','2반','3반','4반','5반'])

type(df)

df['중간고사']['1반']
df['중간고사']
type(df['중간고사'])
df['중간고사'][0:2]
df['중간고사']['1반':'3반']



df.loc['1반']
df.loc[:,'중간고사']
df.loc['1반','중간고사']



df.iloc[0]
df.iloc[0]['중간고사']


cond = df['성적'] =='B'
df.loc[cond]


cond1 = df['성적'] == 'A'
cond2 = df['중간고사'] >= 90
df[cond1 & cond2]


df.describe()#갯수, 평균, 최소, 최대 등
df.info()
df.중간고사.describe()
df['중간고사'].describe()

df.head(1)

df['성적'].unique()
df_mean = df['중간고사'].mean() #평균 계산
df['중간고사'].value_counts() #각 점수의 갯수 출력
df['중간고사'].map(lambda p: p-df_mean)

# 그룹 & 정렬

g1 = df.groupby('성적')
g2 = df.groupby('성적').count()['중간고사']
g3 = df.groupby('성적').sum()
g4 = df.groupby('중간고사').중간고사.agg([len, max, min])


df.sort_values('기말고사') #오름차순
df.sort_values('기말고사',ascending=False) #내림차순



#데이터 추가
df.loc['6반']=[10, 10, np.nan] # 행 추가


df[pd.isnull(df.성적)] # Null값을 찾는 방법


# 열 이름 바꾸기
df = df.rename(columns={'성적':'등급'})

#인덱스 이름 바꾸기
df.rename_axis('반이름',axis='rows')


# 데이터 프레임 합치기
df1 = pd.DataFrame([[89.2,92.5,'B'],
                   [90.8,92.8,'A'],
                   [89.9,95.2,'A'],
                   [89.9,85.2,'C'],
                   [89.9,90.2,'B']],
                  columns=['중간고사','기말고사','성적'],
                  index=['1반','2반','3반','4반','5반'])

df2 = pd.concat([df,df1]) #combining -> 아래로 붙이기
df3 = pd.concat([df,df1],axis=1) #combining -> 옆으로 붙이기

df.shape #데이터 차원 출력
len(df) #데이터 길이 출력










