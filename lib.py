
'''
Description:    Script that uses NLTK stopwords for identifying languages.
Author:         Erion Çano
Reproduce:      Tested on Ubuntu 23.10 with python=3.11.6, nltk=3.8.1  
Run:            python lib.py (few seconds runtime)
'''

from utils import *
from nltk import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.corpus import stopwords

if __name__ == '__main__': 

    '''
    dictionary to store the computed language scores like {'english': 0.64, 'german': 0.43, 'italian': 0.71, 'french': 0.55, ...}
    '''
    lang_scores = dict()

    # get the target text entered by the user
    target_text = input("Please type or paste your text below:\n\n")

    # split based on whitespaces
    words = target_text.split()		
    # remove word repetitions from the text 	
    words = set(words)	

    '''			
    For each language in nltk.corpus, we compute the number of matching stopwords appearing in the given text divided with its total number of words and store each score in the dictionary above.
    '''
    for lang in stopwords.fileids():
        # the stopwords of the current language
        stops = set(stopwords.words(lang))
        overlap_length = words.intersection(stops)
        # compute the matching score for each language
        lang_scores[lang] = len(overlap_length) / len(words)

    # find and print the language with highes matching score
    top_score_lang = max(lang_scores, key=lang_scores.get)
    print(f"\nThe language of this text is probably: {top_score_lang}")
