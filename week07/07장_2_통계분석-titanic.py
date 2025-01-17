# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 22:32:51 2021

@author: Park
"""
import seaborn as sns
import pandas as pd
titanic = sns.load_dataset("titanic") #타이타닉 데이터프레임
titanic.to_csv('./DATA/titanic.csv', index = False)

titanic.isnull().sum()# null의 개수
#age의 중간값을 null에 채움
titanic['age'] = titanic['age'].fillna(titanic['age'].median())
titanic['embarked'].value_counts()
#Southampton으로 embark_town의 null값을 채움
titanic['embark_town'] = titanic['embark_town'].fillna('Southampton')
titanic['deck'].value_counts()
#C로 deck의 null값을 채움
titanic['deck'] = titanic['deck'].fillna('C')
titanic.isnull().sum()

titanic.info()
titanic.survived.value_counts()#사망자 : 549명, 생존자 : 342명, 총 891명

import matplotlib.pyplot as plt
f,ax = plt.subplots(1, 2, figsize = (10, 5))
#남성 생존자와 여성 생존자를 나누어 그래프 출력
titanic['survived'][titanic['sex'] == 'male'].value_counts().plot. \
    pie(explode = [0,0.1], autopct = '%1.1f%%', ax = ax[0], shadow = True)
titanic['survived'][titanic['sex'] == 'female'].value_counts().plot. \
       pie(explode = [0,0.1], autopct = '%1.1f%%', ax = ax[1], shadow = True)
ax[0].set_title('Survived (Male)')
ax[1].set_title('Survived (Female)')
plt.show()

sns.countplot(x='pclass', hue = 'survived', data = titanic)
plt.title('Pclass vs Survived')
plt.show()

titanic_corr = titanic.corr(method = 'pearson', numeric_only=1)
titanic_corr
#survived에 제일 영향을 미치는 것은 fare이다.
#pclass가 높을수록 생존률이 낮아진다
titanic_corr.to_csv('DATA/titanic_corr.csv', index = False)

titanic['survived'].corr(titanic['adult_male'])
titanic['survived'].corr(titanic['fare'])
sns.pairplot(titanic, hue = 'survived')
plt.show()
sns.catplot(x = 'pclass', y = 'survived', hue = 'sex', data = titanic, kind = 'point')
plt.show()

def category_age(x):
        if x < 10:
           return 0
        elif x < 20:
           return 1
        elif x < 30:
           return 2
        elif x < 40:
           return 3
        elif x < 50:
            return 4
        elif x < 60:
           return 5
        elif x < 70:
           return 6
        else:
           return 7

titanic['age2'] = titanic['age'].apply(category_age)
titanic['sex'] = titanic['sex'].map({'male':1, 'female':0})
titanic['family'] = titanic['sibsp'] + titanic['parch'] + 1
titanic.to_csv('./DATA/titanic3.csv', index = False)
heatmap_data = titanic[['survived', 'sex', 'age2', 'family', 'pclass', 'fare']]
colormap = plt.cm.RdBu
sns.heatmap(heatmap_data.astype(float).corr(), linewidths = 0.1, vmax
        = 1.0, square = True, cmap = colormap, linecolor = 'white', annot = True,
        annot_kws = {"size": 10})
plt.show()













