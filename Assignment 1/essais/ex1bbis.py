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
##########################################################################################
# Import data set

do_it = False 
if do_it == True :
    '''
    Creer un dataset à partir de brown
    A supprimer à la fin
    '''
    # Creer un dataset de range = range_
    from nltk.corpus import brown
    range_ = 50
    file = open('brown50.txt', 'w') 
    doc = []

    for i in tqdm(range(range_)):
        
        s = ''
        p = brown.sents()[i]
        for j in p:
            s += str(j)+' '
        file.write(s+'\n')
    file.close()

#%%
nltk_stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
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
    
sentence = text2sentences('brown50.txt')
X,Y,freq = sentences2input(sentence)
Y = sentence2output(sentence,Y)

#%%
 
# une boucle
def neural_net(X=X, Y=Y, neurons=50, epoch=5, learning_rate=0.5):

    X_shape = X.shape[0]
    w_1 = np.random.random((X_shape, neurons))
    w_2 = np.random.random((neurons, X_shape))

    for epoch_ in range(epoch):
        loss = []

    # début de la boucle pour calculer la loss
        for row in tqdm(range(X_shape)):
            # calcul de tous les exit par le NN pour la row == row
            time1 = time.clock()
            hidden_net = np.dot(X.iloc[row, :], w_1)
            hidden_out = expit(hidden_net)
            exit_net = np.dot(hidden_net, w_2)
            exit_out = expit(exit_net)
            time2 = time.clock()
            # calcul de la loss, et ajout à la liste loss pour row==row
            loss_row = 0.5*(Y.iloc[row, :].values - exit_out)**2
            loss.append(loss_row.mean())
            time3 = time.clock()
            # backpropagation sur row en modifiant w_1 et w_2 a la fin de chaque backpropa
            # modif de w_2
            w_2_ = np.zeros((neurons, X_shape))
            time4 = time.clock()
            for i in range(neurons):
                w_2_[i, :]=-(Y.iloc[row, :]-exit_out)*exit_out*(1-exit_out)*hidden_out[i]
            time5 = time.clock()
            for j in range(neurons):
                w_1[row, j]=learning_rate*(-(sum((Y.iloc[row, :]-exit_out)*exit_out*(1-exit_out)*w_2[j,:])*hidden_out[j]*(1-hidden_out[j])*X.iloc[row, row]))
            time6 = time.clock()
            w_2 -= learning_rate*w_2_
            time7 = time.clock()
            
            print("1", time2-time1)
            print("2", time3-time2)
            print("3", time4-time3)
            print("4", time5-time4)
            print("5", time6-time5)
            print("6", time7-time6)
            

        loss = np.array(loss)
        print('Epoch : '+str(epoch_))
        print('Loss : '+str(loss.mean()))
        
    return w_1, w_2
#%%
    
w_1, w_2 = neural_net(X=X, Y=Y, neurons=50, epoch=1, learning_rate=0.5)

#%%

def predic(X=X, w_1=w_1, w_2=w_2):
    return expit(np.dot(expit(np.dot(X, w_1)), w_2))

#%%
##########################################################################################

class mSkipGram:
    def __init__(self,sentences, nEmbed=100, negativeRate=5, winSize = 5, minCount = 5):
        self.sentences = sentences
        self.nEmbed = nEmbed 
        self.negativeRate = negativeRate
        self.winSize = winSize
        self.minCount = minCount


    def train(self,stepsize=1, epochs=1):
        raise NotImplementedError('implement it!')

    def save(self,path):
        raise NotImplementedError('implement it!')

    def similarity(self,word1,word2):
        """
            computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """
        
        w1 = word1.train()
        w2 = word2.train()
        similarity = 1 - normalize(w1-w2, norm='l2')
        return similarity

    @staticmethod
    def load(path):
        raise NotImplementedError('implement it!')
      
#%%
##########################################################################################        
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        sentences = text2sentences(opts.text)
        sg = mSkipGram(sentences)
        sg.train()
        sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)

        sg = mSkipGram.load(opts.model)
        for a,b,_ in pairs:
            print sg.similarity(a,b)