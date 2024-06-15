import pandas as pd
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# 1. 데이터 준비

# CSV 파일 읽기
file_path = 'forest_fire_data_20160101_20231231.csv'
data = pd.read_csv(file_path, encoding='cp949')

# 2. 데이터 전처리

# '발생원인' 컬럼에서 데이터를 추출
causes = data['발생원인'].dropna().tolist()

# 발생원인 데이터를 하나의 문자열로 결합
causes_text = ' '.join(causes)

# 비 알파벳 문자를 공백으로 대체
causes_text = re.sub(r'[^\w]', ' ', causes_text)

# 단어 빈도 계산
words = causes_text.split()
count = Counter(words)

# 빈도수가 높은 단어만 추출
word_count = dict()
for tag, counts in count.most_common(80):
    if len(tag) > 1:  # 단어 길이가 1보다 큰 경우만 추출
        word_count[tag] = counts

# 3. 데이터 시각화

# 한글 폰트 설정
font_path = "c:/Windows/fonts/malgun.ttf"  # 폰트 경로 설정
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# 단어 빈도에 대한 히스토그램
plt.figure(figsize=(10, 10))
plt.xlabel('키워드')
plt.ylabel('빈도수')
plt.grid(True)

sorted_Keys = sorted(word_count, key=word_count.get, reverse=True)
sorted_Values = sorted(word_count.values(), reverse=True)

plt.bar(range(len(word_count)), sorted_Values, align='center')
plt.xticks(range(len(word_count)), list(sorted_Keys), rotation='vertical')

plt.show()

# 4. 워드 클라우드 생성

wc = WordCloud(font_path=font_path, background_color='ivory', width=800, height=600)
cloud = wc.generate_from_frequencies(word_count)

plt.figure(figsize=(8, 8))
plt.imshow(cloud)
plt.axis('off')
plt.show()

# 워드 클라우드 이미지 파일로 저장
output_file = 'forest_fire_data_20160101_20231231_cloud.jpg'
cloud.to_file(output_file)
