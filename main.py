import gzip
import tensorflow as tf

from tensorflow import keras
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

def go():
    res = []
    empty = 0
    pth = '/home/sjack/Music/Downloads/reviews_Video_Games_5.json.gz'
    for p in parse(pth):
        try:
            res.append(p['reviewText'])
        except KeyError:
            empty += 1
    print('Valid reviews: ' + str(len(res)))
    print('Empty reviews: ' + str(empty))
    tokener = Tokenizer(num_words=5000, oov_token='<<OOV>>', lower=True)
    tokener.fit_on_texts(res)
    seq  =  tokener.texts_to_sequences(res[1:100])
    seq = pad_sequences(seq, padding='post', truncating='post')
    #print(tokener.word_index)
    print(seq)
    

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield eval(l)

if __name__ == "__main__":
    go()
