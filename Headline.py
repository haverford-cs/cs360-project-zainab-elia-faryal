'''
Authors: Elia Anagnostou, ...
Date: 12/16/2020
Description: Stores featurization of headline.
'''

from Corpus import Corpus

c = Corpus()
dict = {}

for w in c.words:
    dict[w] = 0

class Headline:
    '''
    Input: h (headline string)
    self.data stores which corpus words can be found in the headline
    '''
    def __init__(self, h):
        updated_dict = check_words(h)
        self.data = updated_dict

def check_words(h):
    '''
    Checks which words from corpus exist in headline. Marks headlines
    words with 1, non-headline words with 0 in dictionary dict.
    '''
    h_words = [w.lower() for w in h.split()]
    for h_word in h_words:
        for w in c.words:
            if h_word == w and dict[h_word] != 1:
                dict[h_word] = 1
    return dict
