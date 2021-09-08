import numpy as np
import pandas as pd

df = pd.read_csv('movie.csv', index_col='序号')
movies = df.values # np格式
print('电影如下：')
print(movies)
old = movies[:-1, 1:4]
new = movies[-1, 1:4]
name = movies[-1, 0]
L = np.sqrt(np.array([np.sum(np.square(old - new), axis=1)],dtype=float).T)
K = 5
top = np.argsort(L[:,0])[0:K]
print('---------------------------')
print('与', name, '最相近的5个电影：')
print(np.column_stack((movies[top], np.round(L[top]))))
result = {}
for i in movies[top][:, -1]:
    if i not in result:
        result[i] = 1
    else:
        result[i] += 1
print('**********************')
print('其中与', name, '最相近的类型为', max(result,key=result.get))