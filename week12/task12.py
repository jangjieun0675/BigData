import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer

# 각 문서의 용어 빈도를 딕셔너리로 정의
term_frequencies = {
    'Doc1': [27, 3, 0, 14],
    'Doc2': [4, 33, 33, 0],
    'Doc3': [24, 0, 29, 17]
}
# 용어를 리스트로 정의
terms = ['car', 'auto', 'insurance', 'best']

# 용어 빈도를 데이터프레임으로 변환
tf_df = pd.DataFrame(term_frequencies, index=terms)

print("Term Frequency DataFrame:")
print(tf_df)

# 각 용어가 나타나는 문서의 수를 계산
df = (tf_df > 0).sum(axis=1)

# 문서 빈도를 데이터프레임으로 변환
df_df = pd.DataFrame(df, columns=['DF'])

# 전체 문서 수를 계산
n_docs = len(tf_df.columns)

# 역문서 빈도를 계산
idf = np.log((n_docs + 1) / (df + 1)) + 1

# 역문서 빈도를 데이터프레임으로 변환
idf_df = pd.DataFrame(idf, columns=['IDF'])

print("Document Frequency (DF) DataFrame:")
print(df_df)

print("\nInverse Document Frequency (IDF) DataFrame:")
print(idf_df)

# TF-IDF 변환기를 생성
tfidf_transformer = TfidfTransformer(norm=None, use_idf=True, smooth_idf=False)

# TF-IDF 행렬을 계산
tfidf_matrix = tfidf_transformer.fit_transform(tf_df.T).toarray().T

# TF-IDF 행렬을 데이터프레임으로 변환
tfidf_df = pd.DataFrame(tfidf_matrix, columns=['Doc1', 'Doc2', 'Doc3'], index=terms)

print("\nTF-IDF DataFrame:")
print(tfidf_df)
