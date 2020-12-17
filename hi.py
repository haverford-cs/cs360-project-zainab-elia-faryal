from os import listdir
from os.path import isfile, join
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
from Corpus import Corpus
from Headline import Headline

example_headline = "Scientists find 800,000-year-old footprints in England"
example_headline2 = "bla bla bla"

h = Headline(example_headline)
h2 = Headline(example_headline2)
#print(h.data)
#print(h2.data)
