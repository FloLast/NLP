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
    
sentence = text2sentences('brown1000.txt')
X,Y,freq = sentences2input(sentence)
Y = sentence2output(sentence,Y)

#%%
        
def neural_network(X, Y, w_1, w_2, neurons=30, learning_rate = 0.1):
    for n in range(X.shape[0]):
        # FORWARD PROPAGATION #
        
        ### INPUT -----> HIDDEN ###
        net_h = []
        out_h = []
        for a in range(neurons):
            net_h.append(sum(w_1[a,k]*X.iloc[n,k] for k in range(X.shape[0])))
            out_h.append(expit(net_h[a]))
        
        ### HIDDEN -----> OUTPUT ### 
        net_o = []
        out_o = []
        for k in range(X.shape[0]):
            net_o.append(sum(w_2[k,a]*out_h[a] for a in range(neurons)))
            out_o.append(expit(net_o[k]))
        
        ### TOTAL ERROR ###
        loss = []
        for k in range(X.shape[0]):
            loss.append(0.5*(Y.iloc[n,k]-out_o[k])**2)
        tot_loss = np.array(loss).mean()
        
        # BACKWARD PROPAGATION #
        
        ### OUTPUT LAYER ###
        Etot_outo = []
        outo_neto = []
        for k in range(X.shape[0]):
            Etot_outo.append(-(Y.iloc[n,k] - out_o[k]))
            outo_neto.append(out_o[k]*(1-out_o[k]))
        
        w_2_prop = np.zeros((X.shape[0],neurons))
        for a in range(neurons):
            for k in range(X.shape[0]):
                w_2_prop[k][a] -= learning_rate * Etot_outo[k] * outo_neto[k] * out_h[a]
        
        ### HIDDEN LAYER ### 
        Etot_outh = []
        for a in range(neurons):
            Eo_outh = []
            for k in range(X.shape[0]):
                Eo_outh.append(-(Y.iloc[n,k] - out_o[k]) * out_o[k]*(1-out_o[k]) * w_2[k,a])
            Etot_outh.append(sum(Eo_outh))
            outh_neth = []
            for k in range(X.shape[0]):
                outh_neth.append(out_h[a]*(1-out_h[a]))
         
        for k in range(X.shape[0]):
            for a in range(neurons):
                w_1[a,k] -= learning_rate * Etot_outh[a] * outh_neth[k] * X.iloc[n,k]
        w_2 = w_2_prop
        return w_1, w_2, tot_loss

#%% 
       
neurons = 100
epochs = 100
w_1 = np.random.random((neurons, X.shape[0]))
w_2 = np.random.random((X.shape[0],neurons))
plot_loss = []   


time1 = time.clock()
for l in range(1,epochs):
    timeb1 = time.clock()
    print("Epoch number %d" % l)
    w_1, w_2, tot_loss = neural_network(X, Y, w_1=w_1, w_2=w_2, neurons=neurons)
    plot_loss.append(tot_loss)
    timeb2 = time.clock()
    timeB = timeb2-timeb1
    print("Loss = %f" % tot_loss)
    print("### epoch ran in %f s" % timeB)
time2 = time.clock()
timeA = time2-time1
print('##################################')
print('Training time = %f s \n' % timeA)

    
#%% 

plt.plot(plot_loss[1:])
plt.xlabel('Epochs')
plt.ylabel('Loss')

#%%

def predict(X, w_1, w_2):
    ### INPUT -----> HIDDEN ###
    net_h = []
    out_h = []
    for a in range(neurons):
        net_h.append(sum(w_1[a,k]*X[k] for k in range(X.shape[0])))
        out_h.append(expit(net_h[a]))
    
    ### HIDDEN -----> OUTPUT ### 
    net_o = []
    out_o = []
    for k in range(X.shape[0]):
        net_o.append(sum(w_2[k,a]*out_h[a] for a in range(neurons)))
        out_o.append(expit(net_o[k]))
    return out_o

#%%
    
X_test = X[308]
result = predict(X_test, w_1, w_2)
similar = np.argmax(result)
similar

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