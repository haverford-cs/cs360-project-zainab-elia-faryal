from os import listdir
from os.path import isfile, join
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
from Corpus import Corpus
from Headline import Headline


example_headline2 = "bla bla bla"

def main():
    example_headline = "Scientists find 800,000-year-old footprints in England"
    #initializes the corpus and datasets and tests on a single headline
    h = Headline(example_headline)
    #to check dictionary format
    #print(h.data)

if __name__ == "__main__":
    main()
