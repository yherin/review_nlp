from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import gzip

def go():
    res = []
    maps = []
    empty = 0
    count = 0
    test_size = 400000

    pth = '/home/sjack/Music/Downloads/reviews_Video_Games.json.gz'
    for p in parse(pth):
        try:
            res.append(p['reviewText'])
            verdict = ''
            review = str(p['overall'])
            if review in ('4.0', '5.0'):
                verdict = 'positive'
            elif review in ('3.0'):
                verdict = 'neutral'
            else:
                verdict = 'negative'    
            maps.append((count, verdict))
           # print(p['reviewText'])
            count += 1
            #if count >= 5000:
            #    break
        except KeyError:
            empty += 1
    train = res[:test_size]
    test = res[test_size+1:]
    print('Valid reviews: ' + str(len(train)))
    print('Empty reviews: ' + str(empty))
    counter = CountVectorizer()
    train_counts = counter.fit_transform(train)
    print('shape of counts: ' + str(train_counts.shape))
    #eature_names = counter.get_feature_names()
    #print(feature_names.index('counterstrike'))
    #count frequencies of symbols
    #Term Frequency times Inverse Document Frequency
    transformer = TfidfTransformer()
    df_train_counts = transformer.fit_transform(train_counts)
  #  print('shape of df-counts: ' + str(df_counts.shape))
    y = []
    for m in maps[0:test_size]:
        y.append(m[1])
    classifier = MultinomialNB().fit(df_train_counts,y)
    ##TEST
    test_counts = counter.transform(test)
    test_td_counts = transformer.transform(test_counts)
    predicted = classifier.predict(test_td_counts)
   # for review, score in zip(test[:100], predicted[100]):
   #     print( "{}  ->  {}".format(review, score))
    match = 0
    total = 0
    for num, score in enumerate(predicted):

        actual = maps[test_size+num]
        if (actual[1] == score):
            match += 1
        total += 1

    print('Accuracy: {} with {} test datum'.format(((match / total)*100), total))

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield eval(l)

if __name__ == "__main__":
    go()