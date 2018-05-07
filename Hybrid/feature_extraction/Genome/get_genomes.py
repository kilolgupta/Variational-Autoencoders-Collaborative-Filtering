import json
from operator import itemgetter
import numpy as np
NUM_GENOMES = 1128


#converts tuple to a 1128 dim vector
def get_genome_vec(genome_tup):
    vec = np.zeros(NUM_GENOMES)
    print len(genome_tup)
    for i in genome_tup:

        tag = int(i[0])-1
        vec[tag] = 1
    return vec


with open('./ml-20m/genome-scores.csv', 'r') as f:
    genome_data = f.read().splitlines()

genome_dict = {}

#Collect all the genomes in genomes-scores.csv
for i in genome_data[1:]:
    i = i.split(",")
    mid = i[0]
    tagid = i[1]
    relevance = float(i[2])
    try :
        genome_dict[mid].append((tagid, relevance))
    except:
        genome_dict[mid] = [(tagid, relevance)]

#sort and select genomes
for mid in genome_dict.keys():
    scores = genome_dict[mid]
    scores = sorted(scores , key=itemgetter(1), reverse = True)
    scores = scores[:20]
    genome_dict[mid] = scores
    print len(genome_dict[mid])

with open('genome_scores.json', 'w') as f:
    json.dump(genome_dict, f)

unk = np.array([0]*num_genomes)
movie_embeddings_array = []

#convert list to a one hot vector
with open('unique_movies', 'r') as f:
    movie_id = f.read().splitlines()
    for i in movie_id:
        try:
            i = i.split(',')
            mid = i[0]
            movie_embeddings_array.append(get_genome_vec(genome_dict[mid]))
        except KeyError:
            movie_embeddings_array.append(unk)

movie_embeddings_array = np.array(movie_embeddings_array)
with open('movie_genomes.npy', 'wb') as f:
    np.save(f, movie_embeddings_array)
