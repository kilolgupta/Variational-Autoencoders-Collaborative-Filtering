import json
import numpy as np 

def process_lang(string):
    string = string.split(",")
    string = [i.lstrip(' ').rstrip(' ') for i in string]
    return string

with open('imdb_data.json', 'r') as f:
    raw_data = json.load(f)

language = {'unique' : set(), 'list' : {} }
imdb_rat = {}
plot = {}
rated = {'unique' : set(), 'list' : {} }

mid2idx = {}
with open('unique_movies', 'r') as f:
    data = f.read().splitlines()
    for i in data:
        i = i.split(',')
        mid2idx[i[0]] = int(i[1])
        
for mid in mid2idx.keys():
    midx = mid2idx[mid]
    
    lang = process_lang(raw_data[mid]['Language'])
    language['list'][midx] = lang
    for i in lang: 
        language['unique'].add(i)

    if raw_data[mid]['imdbRating'] != 'N/A':
        imdb_rat[midx] = float(raw_data[mid]['imdbRating'])
    else :    
        imdb_rat[midx] = float(0.0)

    plot[midx] = raw_data[mid]['Plot']

    rated['list'][midx] = raw_data[mid]['Rated']
    rated['unique'].add(raw_data[mid]['Rated'])


language['unique'] = dict( (item, i) for (i, item) in enumerate(list(language['unique'])))  
rated['unique'] = dict( (item, i) for (i, item) in enumerate(list(rated['unique'])))  

with open('language.json', 'w') as f:
    json.dump( language,f)

with open('imdb_rat.json', 'w') as f:
    json.dump( imdb_rat ,f)

with open('plot.json', 'w') as f:
    json.dump( plot,f)

with open('rated.json', 'w') as f:
    json.dump( rated ,f)



