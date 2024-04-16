# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 20:33:29 2024

@author: ATIV
"""

def common_data(list1, list2):
    for x in list1:
        if x in list2:
            return True
    return False

print(common_data([1,2,3,4,5],[5,6,7,8,9]))


dic={'name':'jang', 'phone':'01058130553','birth':'0116'}
dic.keys()
dic.values()
dic.items()
dic['name']
dic['pet']='dog'
dic.pop('name')

f = open('a.txt', 'w')
for i in range(1,5):
    data = '%d번째 줄입니다\n' %i
    f.write(data)
f.close()

import numpy as np
ar1 = np.array([1,2,3,4,5])
list1 = [1,2,3,4,5]

ar2 = np.array(list1)
ar3 = np.arange(1,11,2)
ar4 = np.array([1,2,3,4,5,6])
ar4 = np.array([1,2,3,4,5,6]).reshape(3, 2)

import pandas as pd
pd.__version__

data1 = [10,20,30,40,50]

sr1 = pd.Series(data1)

sr3 = pd.Series([10,20,30,40,50])

sr5 = pd.Series(data1, index = [1000,1001,1002,1003,1004])

sr7 = pd.Series(data1, index = sr5)

sr9 = sr1+sr3

