import gensim
import sys
from liwc import liwc_score
from vad import vad_score

import json
from nltk.stem.lancaster import LancasterStemmer
from nltk import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import FreqDist as freq
import numpy as np
from nltk.stem import WordNetLemmatizer

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True) 
model.save("word2vec.model")

tokenizer = RegexpTokenizer(r'\w+')    
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ls = liwc_score()
vs = vad_score()

def token_stemmer(inp_sentence):
    token_list = tokenizer.tokenize(inp_sentence)
    token_list = [word for word in token_list if word not in stop_words] 
    stem_list = [lemmatizer.lemmatize(word) for word in token_list]
    return stem_list

def normalize(x):
    return (x - np.min(x))/(np.max(x) - np.min(x))

def get_word2vec(text):
    word_list = token_stemmer(text)
    vec = np.zeros(300)
    count = 1.0
    for word in word_list:
        try:
            vec+= (model[word])
            count += 1.0
            #vec+= model[word]
        except:
            pass
    return vec/count


with open('plot.json', 'r') as f:
    plot_data = json.load(f)

w2vec_list = []
plot_feature = {}
for mid in plot_data.keys():
    text = plot_data[mid]
    
    w2vec = get_word2vec(text)
    w2vec_list.append(w2vec)

    vec = list(ls.get_liwc_score(text))
    vec += list(vs.get_vad_score(text))
    
    plot_feature[mid] = vec

w2vec_list = np.array(w2vec_list)
w2vec_list = (w2vec_list - np.min(w2vec_list))/(np.max(w2vec_list) - np.min(w2vec_list))

mids = liwc_vad_dict.keys()

for i in range(0, len(mids)):
    mid = mids[i]
    vec = list(w2vec_list[i])
    plot_feature[mid] = plot_feature[mid] + vec

with open('plot_feature.json', 'w') as f:
    json.dump(plot_feature, f)


