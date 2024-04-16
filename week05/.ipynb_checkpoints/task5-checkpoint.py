# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 00:09:55 2024

@author: ATIV
"""

#과제

import csv
file_2018 = open('subwaytime_201803.csv', encoding='UTF8')
data_2018 = csv.reader(file_2018)
next(data_2018)
next(data_2018)
s2018_in = [0] * 24
s2018_out = [0] * 24
for row in data_2018 :
    row[4:] = map(int, row[4:]) 
    for i in range(24) :
        s2018_in[i] += row[4 + i * 2] # 2018년 시간대별 승차인원
        s2018_out[i] += row[5 + i * 2] # 2018년 시간대별 하차인원


file_2020 = open('subwaytime_202003.csv', encoding='UTF8')
data_2020 = csv.reader(file_2020)
next(data_2020)
next(data_2020)
s2020_in = [0] * 24
s2020_out = [0] * 24
for row in data_2020 :
    row[4:] = map(int, row[4:]) 
    for i in range(24) :
        s2020_in[i] += row[4 + i * 2] # 2020년 시간대별 승차인원
        s2020_out[i] += row[5 + i * 2] # 2020년 시간대별 하차인원


file_2023 = open('subwaytime_202303.csv', encoding='UTF8')
data_2023 = csv.reader(file_2023)
next(data_2023)
next(data_2023)
s2023_in = [0] * 24
s2023_out = [0] * 24
for row in data_2023 :
    row[4:] = map(int, row[4:]) 
    for i in range(24) :
        s2023_in[i] += row[4 + i * 2] # 2023년 시간대별 승차인원
        s2023_out[i] += row[5 + i * 2] # 2023년 시간대별 하차인원


import matplotlib.pyplot as plt
plt.figure(dpi = 100)
plt.rc('font', family = 'Malgun Gothic')
plt.title('지하철 시간대별 승하차 인원 추이')
plt.plot(s2018_in, label = '201803승차')
plt.plot(s2018_out, linestyle='--', label = '201803하차')
plt.plot(s2020_in, label = '202003승차')
plt.plot(s2020_out, linestyle='--', label = '202003하차')
plt.plot(s2023_in, label = '202303승차')
plt.plot(s2023_out, linestyle='--', label = '202303하차')
plt.legend()
plt.xticks(range(24), range(4,28))
plt.show()






import numpy as np
import csv
f = open('age.csv')
data = csv.reader(f)
next(data)

name = input('인구 구조가 알고 싶은 지역의 이름(읍면동 단위)을 입력해주세요 : ')
mn = 1 
result_name = ''
result = 0 
home = []

for row in data :
    if name in row[0]: 
        areaname=row[0]
        for i in row[3:]: 
            home.append(int(i)) 
        hometotal=int(row[2])
for k in range(len(home)):
    home[k]=(home[k]/hometotal)
result_list=[]
f = open('age.csv')
data = csv.reader(f)
next(data)
for row in data : 
    away=[]
    for i in row[3:]:
        away.append(int(i)) 
    awaytotal=int(row[2])
    for k in range(len(away)):
        away[k]=(away[k]/awaytotal)
    s=0
    for j in range(len(away)):
        s=s+(home[j]-away[j])**2
    result_list.append([row[0], away, s])
result_list.sort(key=lambda s: s[2])


import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.figure(figsize = (10,5), dpi=100)            
plt.rc('font', family ='Malgun Gothic')
plt.title(name +' 지역과 가장 비슷한 인구 구조를 가진 지역')
plt.plot(home, label = name)
plt.plot(result_list[1][1], label = result_list[1][0])
plt.plot(result_list[2][1], label = result_list[2][0])
plt.plot(result_list[3][1], label = result_list[3][0])
plt.plot(result_list[4][1], label = result_list[4][0])
plt.plot(result_list[5][1], label = result_list[5][0])
plt.legend()
plt.show()




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('age.csv', encoding='cp949')
data = data.iloc[1:]

name = input('인구 구조가 알고 싶은 지역의 이름(읍면동 단위)을 입력해주세요: ')

home = data[data['행정구역'].str.contains(name)].iloc[0, 3:].astype(int)
home_total = home.sum()
home_result = home / home_total

away = data.iloc[:, 3:].astype(int)
away_total = away.sum(axis=1)
away_result = away.div(away_total, axis=0)

s = ((home_result - away_result) ** 2).sum(axis=1)
result_list = pd.DataFrame({'행정구역': data['행정구역'], 'away_result': away_result.values.tolist(), 's': s})
result_list = result_list.sort_values(by='s')

plt.style.use('ggplot')
plt.figure(figsize=(10, 5), dpi=100)
plt.rc('font', family='Malgun Gothic')
plt.title(name + ' 지역과 가장 비슷한 인구 구조를 가진 지역')

x_ticks = np.arange(0, len(home), 20)
plt.xticks(x_ticks)

plt.plot(home_result, label=name)

for idx in range(1, 6):
    plt.plot(result_list.iloc[idx]['away_result'], label=result_list.iloc[idx]['행정구역'])

plt.legend()
plt.show()












