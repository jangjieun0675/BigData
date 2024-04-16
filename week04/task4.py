# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 18:33:54 2024

@author: ATIV
"""

import csv

f  = open('201402_201402_people.csv')
data = csv.reader(f)
for row in data :
    print(row)
    
#----------------------------------------------------------

import csv

file_2014 = open('201402_201402_people.csv')
reader_2014 = csv.reader(file_2014)
next(reader_2014)


file_2024 = open('202402_202402_people.csv')
reader_2024 = csv.reader(file_2024)
next(reader_2024)

data_2014 = list(reader_2014)
data_2024 = list(reader_2024)

result = []

for row_2014, row_2024 in zip(data_2014, data_2024):
   
    population_2014 = int(row_2014[1].replace(',', ''))
    population_2024 = int(row_2024[1].replace(',', ''))
    
    population_change = population_2024 - population_2014
    result.append(population_change)

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import font_manager, rc
import platform
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

#스타일설정
plt.style.use('ggplot')
#이미지 사이즈 설정 - 단위는 인치
plt.figure(figsize = (15,15)) #데이터 많으면 여기 조정 해야 한다.


bars = ('전국', '서울특별시', '부산광역시', '대구광역시', '인천광역시', '광주광역시', '대전광역시', '울산광역시', '세종특별자치시',
        '경기도', '강원도', '충청북도', '충청남도', '전라북도', '전라남도', '경상북도', '경상남도', '제주특별자치도')
x_pos = np.arange(len(bars))

plt.bar(x_pos, result)

# Create names on the x-axis
plt.xticks(x_pos, bars)

# Show graphic
plt.show()

plt.savefig('Q4-1.png')




import pandas as pd

df_2014 = pd.read_csv('201402_201402_people.csv', encoding='cp949')
df_2024 = pd.read_csv('202402_202402_people.csv', encoding='cp949')


df_2014['2014년02월_총인구수'] = df_2014['2014년02월_총인구수'].str.replace(',', '').astype(int)
df_2024['2024년02월_총인구수'] = df_2024['2024년02월_총인구수'].str.replace(',', '').astype(int)

result = df_2024['2024년02월_총인구수'] - df_2014['2014년02월_총인구수']

print(result)



import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import font_manager, rc
import platform
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

df_2014 = pd.read_csv('201402_201402_people.csv', encoding='cp949')
df_2024 = pd.read_csv('202402_202402_people.csv', encoding='cp949')

df_2014['2014년02월_총인구수'] = df_2014['2014년02월_총인구수'].str.replace(',', '').astype(int)
df_2024['2024년02월_총인구수'] = df_2024['2024년02월_총인구수'].str.replace(',', '').astype(int)

result = df_2024['2024년02월_총인구수'] - df_2014['2014년02월_총인구수']
result.index = ['전국', '서울특별시', '부산광역시', '대구광역시', '인천광역시', '광주광역시', '대전광역시', '울산광역시', '세종특별자치시',
                '경기도', '강원도', '충청북도', '충청남도', '전라북도', '전라남도', '경상북도', '경상남도', '제주특별자치도']

plt.figure(figsize = (12,18))
result.plot(kind='bar')

plt.title('2014년부터 2024년까지의 인구 변화')

plt.show()






