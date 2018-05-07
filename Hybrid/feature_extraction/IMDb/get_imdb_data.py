import json
import requests
import time 

def get_movie_data(imdb_id):
	url = 'http://www.omdbapi.com/?i={}&apikey=96725c4f&plot=full'.format(imdb_id)
	response = requests.get(url)
	return response.json()

with open('links.csv', 'r') as f:
	ids = f.read().splitlines()


movie_dict = []
count =  0
for i in ids[1:]:
	i = i.split(",")
	ml_id = i[0]
	imdb_id = 'tt' + i[1]

	print imdb_id
	
	movie = get_movie_data(imdb_id)
	if movie['Response'] == 'True':
		movie_dict.append(movie)

	if count % 500 == 0:
		json_data = open("imdb_data.json", "w")
		movie_dict_json = json.dumps(movie_dict)
		json_data.write(movie_dict_json)
		json_data.close()

	count += 1
	time.sleep(1)
