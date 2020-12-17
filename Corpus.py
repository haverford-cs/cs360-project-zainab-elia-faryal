'''
Author: Elia Anagnostou, Zainab Batool
Date: 12/16/2020
Desc: Creates corpus out of Fox News and CNN technology news headlines.
'''

from os import listdir
from os.path import isfile, join
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
import csv
import random
import re


fox_path = 'FOX/Raw/foxnews_Science_technology/'
cnn_path = 'CNN/Raw/cnn_technology/'
fox_files = [f for f in listdir(fox_path) if isfile(join(fox_path, f))]
cnn_files = [f for f in listdir(cnn_path) if isfile(join(cnn_path, f))]

txtfiles = []
headlines = []
labels = []

class Corpus:
    def __init__(self):
        headlines = process_headlines(fox_path, cnn_path)
        #print(headlines)
        train_headlines, test_headlines, train_labels, test_labels = split_headlines(headlines, labels)
        corpus = process_corpus(train_headlines)
        #print(corpus)
        form_dataset_csv(train_headlines, corpus, train_labels,"headlines_train.csv")
        form_dataset_csv(test_headlines, corpus, test_labels,"headlines_test.csv")
        self.words = corpus
        self.n = len(corpus)

def process_headlines(fox_path, cnn_path):
    '''
    Input:  fox_path: path to desired Fox News category
            cnn_path: path to desired CNN category
    Returns headlines of news articles found in provided paths.
    '''
    fox_files = [f for f in listdir(fox_path) if isfile(join(fox_path, f))]
    cnn_files = [f for f in listdir(cnn_path) if isfile(join(cnn_path, f))]

    txtfiles = []
    headlines = []

    for f in fox_files:
        if f[-3:] == 'txt':
            txtfiles.append((f,'fox'))

    for f in cnn_files:
        if f[-3:] == 'txt':
            txtfiles.append((f,'cnn'))

    random.shuffle(txtfiles)
    for file in txtfiles:
        if file[1] == 'fox':
            f = open(fox_path + file[0], "r", encoding='utf-8')
            # 0 for Fox and 1 for CNN as label
            labels.append(0)
        else:
            f = open(cnn_path + file[0], "r", encoding='utf-8')
            labels.append(1)
        str_f = f.read()

        if '<TITLE>' in str_f:
            start = str_f.index('<TITLE>') + 7
        if '</TITLE>' in str_f:
            end = str_f.index('</TITLE')

        headlines.append(str_f[start:end])
    print(len(labels))
    print(len(headlines))
    return headlines

def process_corpus(train_headlines):
    '''
    Input:  headlines: list of strings that represent news headlines
    Returns list of all words found in headlines minus stopwords,
    numbers, and duplicates.
    '''

    stop_words = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r"\w+")
    word_tokens = tokenizer.tokenize(' '.join(train_headlines))
    #print(word_tokens)
    remove_stopwords = [w for w in word_tokens if not w in stop_words]
    remove_numbers = [w.lower() for w in remove_stopwords if (not w.isdigit() and not len(w)==1)]
    corpus = set(remove_numbers) #remove duplicates

    # for w in corpus:
    #     print(w)

    print(len(remove_stopwords))
    print(len(remove_numbers))
    print(len(corpus))
    print(len(train_headlines))
    #print(headlines)
    #print(corpus)
    return corpus

def split_headlines(headlines, labels):
    #80 % of dataset is training rest is testing
    train_length = int(len(headlines)*0.8)
    train_headlines = headlines[:train_length]
    test_headlines = headlines[train_length:]
    train_labels = labels[:train_length]
    test_labels = labels[train_length:]
    return train_headlines, test_headlines, train_labels, test_labels

    #change corpus to only 80 % of the headlines- training headlines
    # use these to form the Corpus
    # form a csv of vectors of train and test examples with features
    # feed that to logistic regression

def form_dataset_csv (headlines, corpus, labels, filename):
    #print(word_tokens)
    with open(filename, mode= 'w') as our_file:
        file_writer = csv.writer(our_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i,h in enumerate(headlines):
            #print("headline")
            #print(h)
            feature_values = []
            h_words = [re.sub("[^\\w]", "",w.lower()) for w in h.split()]
            #print(h_words)
            for w in corpus:
                if w in h_words:
                    feature_values.append(1)
                else:
                    feature_values.append(0)
            feature_values.append(labels[i])
            file_writer.writerow(feature_values)
