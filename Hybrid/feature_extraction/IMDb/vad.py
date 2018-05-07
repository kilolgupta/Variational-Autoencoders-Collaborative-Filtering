import json
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import FreqDist as freq

"""with open('warriner.csv', 'r') as f:
    vad_raw = f.read().splitlines()

vad_json = {}
for i in vad_raw[1:]:
    i = i.split(",")
    item = {}
    item['v'] = float(i[2])
    item['a'] = float(i[5])
    item['d'] = float(i[8])
    vad_json[i[1]] = item

with open('warriner.json', 'w') as f:
    json.dump(vad_json, f)"""


class vad_score():
    def __init__(self):
        with open('warriner.json', 'r') as f:
            self.vad = json.load(f)
        self.tokenizer = RegexpTokenizer(r'\w+')    
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def token_stemmer(self, inp_sentence, remove_stop ):
        token_list = self.tokenizer.tokenize(inp_sentence)
        if remove_stop :
            token_list = [word for word in token_list if word not in self.stop_words] 
        lemma_list = [self.lemmatizer.lemmatize(word) for word in token_list]

        return lemma_list

    def get_vad_score(self, inp_sentence, remove_stop= False):
        token_list = self.token_stemmer(inp_sentence, remove_stop)
        fdist = dict(freq(token_list))

        v_sum = 0.0
        a_sum = 0.0
        d_sum = 0.0
        count = 1.0

        for word in fdist.keys():
            try:
                v_sum = self.vad[word]['v']*fdist[word]
                a_sum = self.vad[word]['a']*fdist[word]
                d_sum = self.vad[word]['d']*fdist[word]
                count += fdist[word]
            except KeyError:
                pass
                
        return (v_sum/count, a_sum/count, d_sum/count)


if __name__ == "__main__":
    with open('test', 'r') as f:
        inp_data = f.read()

    l = vad_score()
    print l.get_vad_score(inp_data)

