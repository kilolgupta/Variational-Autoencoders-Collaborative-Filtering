import json
import numpy as np

with open('./ml20-m/movies.csv', 'r') as f:
    data = f.read().splitlines()

genre_json = {}
for i in data[1:]:
    i = i.split('::')
    m_id = i[0]
    genres = i[2]
    print genres
    genre_data[m_id] = genres.split('|')

with open('genres.json', 'w') as f:
    json.dump(genre_data, f)


genres = set()
for m in genre_data.keys():
    for g in genre_data[m]:
        genres.add(g)
genre2id = dict((genre, i) for (i, genre) in enumerate(genres))

with open('unique_movies', 'r') as f:
    movie_ids = f.read().splitlines()

with open('movie_genres.npy', 'w') as f:
    genre_array = []
    for i in movie_ids:
        i = i.split(',')
        mid = i[0]
        genre_array.append(genre2id[genre_data[mid][0]])
    genre_array = np.array(genre_array)
    np.save(f, genre_array)
    
