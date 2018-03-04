#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
NATURAL LANGUAGE PROCESSING

Assignment N°1

"""

from __future__ import division

####### A SUPPRIMER #########
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

### FUNCTIONS TO IMPORT
import argparse
from scipy.special import expit
from sklearn.preprocessing import normalize
from nltk.tokenize import word_tokenize

import pandas as pd
import numpy as np


__authors__ = ['floredelasteyrie','romainboyer']
__emails__  = ['flore.delasteyrie@essec.edu','romain.boyer1@essec.edu']

##########################################################################################
#%% USEFUL FUNCTIONS

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
    sentences = [x for x in sentences if len(x)>1]
    return sentences


def loadPairs(path):
    data = pd.read_csv(path,delimiter='\t')
    pairs = zip(data['word1'],data['word2'],data['similarity'])
    return pairs

##########################################################################################
#%% DATA PREPROCESSING

X, Y, freq, all_words = sentences2input(sentence)
Y = sentence2output(sentence,Y)

##########################################################################################
#%%


class mSkipGram:
    def __init__(self,sentences, nEmbed=100, negativeRate=5, winSize = 5, minCount = 5):
        # The sentences as the output of text2sentences()
        self.sentences = sentences
        # Number of neurons
        self.nEmbed = nEmbed 
        # Number of words to select for negative sampling
        self.negativeRate = negativeRate
        # The size of 
        self.winSize = winSize
        self.minCount = minCount
        
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

    def sentences2input(sentences):
        ''' Create the dataframe with the words' input vector
        And the dictionnary with the words' frequencies '''
        frequencies = {}
        all_words = []
        for sentence in tqdm(sentences):
            for word in sentence:
                all_words.append(word)
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
        
        return X,Y,frequencies, all_words
    
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

    def train(self,stepsize=1, epochs=1):
        
        X_shape = X.shape[0]
        all_words_len = len(all_words)
        
        for epoch_ in range(epoch):
            loss = []
            time12=[]
            time24=[]
            time45=[]
        # début de la boucle pour calculer la loss
            for row in tqdm(range(X_shape)):
                
                '''FORWARD'''
                
                '''https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/'''
                time1 = time.clock()
                
                # Computation of all exits for row == row
                hidden_net = np.dot(X.iloc[row, :], w_1)
                hidden_out = expit(hidden_net)
                exit_net = np.dot(hidden_net, w_2)
                exit_out = expit(exit_net)
                
                # Computation of the loss for this row, and adding it to the list 'loss'
                loss_row = 0.5*(Y.iloc[row, :].values - exit_out)**2
                loss.append(loss_row.mean())
                
                # Creation of w_2_ to modify w_2
                w_2_ = np.zeros((neurons, X_shape))
                
                '''NEGATIVE SAMPLING'''
                time2 = time.clock()
                
                ### Creation of 'chosen_words' : 
                ### 5 random words (with probability 0) + the most probable word
                chosen_words = []
                
                while len(chosen_words)<5: 
                    # Choose a random word
                    rand = np.random.randint(0, all_words_len)
                    # Verify its probability is 0 and that it's not the row word and it's not already chosen
                    if (Y.iloc[row,:][all_words[rand]] == 0) & (Y.columns[row]!=all_words[rand]) & (all_words[rand] not in chosen_words):
                        chosen_words.append(all_words[rand])
                chosen_words.append(Y.iloc[0,:].argmax())
                    
                ### Retrieve the indexes of the chosen words (10^-5 sec par iteration)
                words_index = {}
                for word in chosen_words:
                    words_index[word] = X.index.get_loc(word)
                
                '''BACKWARD'''
                
                '''http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/'''
                time4 = time.clock()
                
                for key, value in words_index.items():
                    
                    w_2_[:, value] = -(Y.iloc[row, value]-exit_out[value])*exit_out[value]*(1-exit_out[value])*hidden_out  
                    w_1[row, :] -= learning_rate*(-1)*(Y.iloc[row, value]-exit_out[value])*exit_out[value]*(1-exit_out[value])*w_2[:, value]*hidden_out*(1-hidden_out)                
                w_2 -= learning_rate*w_2_   
                     
                '''END'''
                time5 = time.clock()
            
                time12.append(time2-time1)
                time24.append(time4-time2)
                time45.append(time5-time4)
        
                
            print('\nForward  : mean '+str(np.mean(time12))+' / std : '+str(np.std(time12)))
            print('Negative : mean '+str(np.mean(time24))+' / std : '+str(np.std(time24)))
            print('Backward : mean '+str(np.mean(time45))+' / std : '+str(np.std(time45)))
            print('Total (min) : '+str((np.sum(time12)+np.sum(time24)+np.sum(time45))/60))
            
            
            loss = np.array(loss) 
            print('Epoch : '+str(epoch_))
            print('Loss : '+str(loss.mean()))
            print('-'*30)
        return w_1, w_2
    
    def predict(X, w_1=w_1, w_2=w_2):
        return expit(np.dot(expit(np.dot(X, w_1)), w_2))

    def save(self,path):
        raise NotImplementedError('implement it!')

    def similarity(self,word1,word2):
        """
            computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """
        if (word1 not in all_words):
            print("'%s' not in the text" % word1)
            pass
        if (word2 not in all_words):
            print("'%s' not in the text" % word2)
            pass
        else:
            w1 = predict(word1)
            w2 = predict(word2)
            similarity = 1 - normalize(w1-w2, norm='l2')
            return similarity

    @staticmethod
    def load(path):
        raise NotImplementedError('implement it!')
      
##########################################################################################  
#%%
      
        
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