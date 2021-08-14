import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
tweet = 'For some quick analysis, creating a corpus could be overkill. If all you need is a word list, there are simpler ways to achieve that goal'
tokenizer = TweetTokenizer(preserve_case=False)
tweet_tokens = tokenizer.tokenize(tweet)
stopwords_english = stopwords.words('english')
print(tweet_tokens)
print()

tweets_clean = []
for word in tweet_tokens:
    if(word not in stopwords_english) and word not in string.punctuation:
            tweets_clean.append(word)
print(tweets_clean)
print()            
            
stemmer = PorterStemmer()
tweets_stem = []
for word in tweets_clean:
    stem_word = stemmer.stem(word)
    tweets_stem.append(stem_word)
print(tweets_stem)