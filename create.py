import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv('data.csv')

data['comb'] = data['actor_1_name'] + ' ' + data['actor_2_name'] + ' '+ data['actor_3_name'] + ' '+ data['director_name'] +' ' + data['genres']


cv = CountVectorizer()
count_matrix = cv.fit_transform(data['comb'])

sim = cosine_similarity(count_matrix)

np.save('similarity_matrix', sim)

data.to_csv('data.csv',index=False)