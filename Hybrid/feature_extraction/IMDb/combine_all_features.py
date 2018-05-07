import json
import numpy as np

with open('language.json', 'r') as f:
    language = json.load(f)

with open('imdb_rat.json', 'r') as f:
    imdb_rat = json.load(f)

with open('plot_feature.json', 'r') as f:
    plot = json.load(f)

with open('rated.json', 'r') as f:
    rated = json.load(f)

mid2idx = {}
with open('unique_movies', 'r') as f:
    data = f.read().splitlines()
    for i in data:
        i = i.split(',')
        mid2idx[i[0]] = int(i[1])

movie_feat_list = []

idx = language['unique'] 
for i in language['list'].keys():
    vec = np.zeros(len(idx))
    lang = language['list'][i]
    for j in lang:
        item = idx[j]
        vec[item] =1.0

    language['list'][i] = list(vec)


idx = rated['unique'] 
for i in rated['list'].keys():
    vec = np.zeros(len(idx))
    item = idx[rated['list'][i]]
    vec[item] =1.0 
    rated['list'][i] = list(vec)


for i in mid2idx.keys():
    mid = str(mid2idx[i])
    #    mid = str(mid)
    vec = language['list'][mid]
    vec+= plot[mid]
    vec+= rated['list'][mid]
    vec.append(imdb_rat[mid])
    vec = np.array(vec)
    movie_feat_list.append(vec)

movie_feat_list = np.array(movie_feat_list)
print movie_feat_list.shape

with open('movie_features.npy', 'wb') as f:
    np.save(f, movie_feat_list)

