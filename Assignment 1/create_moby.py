#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 14:36:32 2018

@author: Flore
"""

from __future__ import division
    
__authors__ = ['floredelasteyrie','romainboyer','shenguadu']
__emails__  = ['flore.delasteyrie@essec.edu','romain.boyer1@essec.edu','shengua.du@essec.edu']


####### A SUPPRIMER #########
from tqdm import tqdm


# Pour utiliser dès maintenant la future fonctionnalité de Python :
# Le signe '/' donne un float même si on divise deux integers. 
# Il faut utiliser '//' pour avoir un integer en sortie.
import argparse
# Sert à gérer correctement les arguments passés dans les lignes de 
# commande au lancement des programmes Python dans une console
from scipy.special import expit
# Aussi appelée la 'logistic function'. Elle est définie comme expit(x) = 1/(1+exp(-x))
# (son inverse est la fonction logit)
from sklearn.preprocessing import normalize
# Normalise des arrays élément par élément. On peut sélectionner la 
# norme souhaitée : norm : ‘l1’, ‘l2’, or ‘max’, optional (‘l2’ by default)
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk
import time


#%%

# Import RegexpTokenizer from nltk.tokenize
from nltk.tokenize import RegexpTokenizer


tokenizer = RegexpTokenizer('\w+')
file = open('moby.txt', 'rt')
text = file.read()

sentences = text.split('.')
tokens = []
for sentence in sentences:
    tokens.append(tokenizer.tokenize(sentence))
file.close()

file = open('moby_7599.txt', 'w')
for k in range(7599):
    for token in tokens[k]:
        file.write(token+' ')
    file.write('\n')
file.close()


#%%
    
nltk_stopwords = ['chapter', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
                  "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 
                  'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 
                  'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 
                  'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 
                  'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 
                  'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 
                  'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 
                  'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
                  'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
                  'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 
                  'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 
                  'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 
                  'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
                  'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', 
                  "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 
                  'won', "won't", 'wouldn', "wouldn't"]

#%%
##########################################################################################

from nltk.tokenize import word_tokenize


def text2sentences(path):
    ''' Tokenization / pre-processing '''
    sentences = []
    with open(path) as f:
        for l in tqdm(f):
            ''' Transform the sentence in words '''
            tokens = word_tokenize(l)
            ''' Lowercase all the characters of the words '''
            tokens = [w.lower() for w in tokens]
            ''' Delete numbers and punctations '''
            words = [word for word in tokens if word.isalpha()]
            ''' Delete words defined as stopwords  '''
            words = [w for w in words if not w in nltk_stopwords]
            sentences.append(words)
    return sentences

def sentences2input(sentences):
    ''' Create the dataframe with the words' input vector
    And the dictionnary with the words' frequencies '''
    frequencies = {}
    for sentence in tqdm(sentences):
        for word in sentence:
            if word in frequencies:
                frequencies[word] += 1
            else:
                frequencies[word] = 1
    
    ''' Create a dataframe with as columns all unique words and as their unique vector '''
    X = pd.DataFrame(np.eye(len(frequencies)), index=frequencies.keys())
    ''' Create a dataframe we'll be using for sentence2output '''
    Y = pd.DataFrame(np.zeros((len(frequencies),len(frequencies))),columns=frequencies.keys(), index=frequencies.keys())
    
    ''' Create a dictionary with the probability for each word '''
    tot = np.sum(list(frequencies.values()))
    #    for key, value in frequencies.iteritems() :
    for key, value in frequencies.items():
        frequencies[key] = value / tot
    
    return X,Y,frequencies

def sentence2output(sentences, Y, window=3):
    ''' Creates a dataframe with on each line the probability for being 
    near another word. '''
    
    pairs = [] 
    for sentence in tqdm(sentences):
        #print(sentence)
        for i in range(len(sentence)):
            for k in range(1,window):
                
                ''' To avoid errors for the last words of a sentence '''
                if (i+k+1 <= len(sentence)):
                    #print(sentence[i],sentence[i+k])
                    ''' We add one for the next k words '''
                    Y.loc[sentence[i],sentence[i+k]] += 1 
                    
                ''' To avoid errors for the first words of a sentence '''
                if (i-k+1 > 0 ):
                    #print(sentence[i],sentence[i-k])
                    ''' We add one for the previous k words '''
                    Y.loc[sentence[i],sentence[i-k]] += 1 
                    
    ''' For each word, we divide the count of other words by the total counts  '''        
    Y = Y.div(Y.sum(axis=1),axis=0)
    return Y

def loadPairs(path):
    data = pd.read_csv(path,delimiter='\t')
    pairs = zip(data['word1'],data['word2'],data['similarity'])
    return pairs

#%%
    
sentence = text2sentences('moby_new.txt')
X,Y,freq = sentences2input(sentence)
Y = sentence2output(sentence,Y)