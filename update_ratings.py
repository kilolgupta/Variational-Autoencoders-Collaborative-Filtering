import json

with open('movie_data.json', 'r') as f:
    data  = json.load(f)

with open('links.csv', 'r') as f:
    ids = f.read().splitlines()

movie_json = {}
for i in data:
    movie_json[i['imdbID']] = i

movie_ids_list = []
movie_json_ml20 = {}
for i in ids:
    i = i.split(",")
    imdb = i[1]
    ml20 = i[0]
    try:
        movie_json_ml20[ml20] = movie_json['tt'+imdb]
        movie_ids_list.append(ml20)
    except KeyError:
        pass
        
with open('movie_ml20.json', 'w') as f:
    json.dump(movie_json_ml20, f)

with open('ratings.csv', 'r') as f:
    ratings = f.read().splitlines()

with open('new_ratings.csv', 'w') as f:
    for i in ratings:
        j = i
        i = i.split(",")
        movie_id = i[1]
        try :
            a = movie_json_ml20[movie_id]
            f.write(j+"\n")

        except KeyError:
            pass

