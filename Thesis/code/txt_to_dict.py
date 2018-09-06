#-*- coding: utf-8 -*-
from collections import defaultdict
from pprint import pprint  # pretty-printer
from stopwords import *
import re


#with open('C:/Aditya/Personal/research/bunch_of_documents.txt', 'r') as myfile:
    #data=myfile.read()
 # remove common words and tokenize
#stoplist = set('for a of the and to in is that we if you not they'.split())

with open('C:/Aditya/Personal/research/bunch_of_documents.txt', 'r') as data:
        for line in data:
            a = line.rstrip().lower()
            result = re.sub('[^a-zA-Z]', ' ', a)
            result = result.split()#creates a list of words
            b = removeStopwords(result, stopwords)
            print(b)