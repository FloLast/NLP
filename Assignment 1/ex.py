#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
NATURAL LANGUAGE PROCESSING

Assignment N°1

"""

from __future__ import division

####### A SUPPRIMER #########
from tqdm import tqdm
import time

### FUNCTIONS TO IMPORT
import argparse
from scipy.special import expit
from sklearn.preprocessing import normalize
from nltk.tokenize import word_tokenize

import pandas as pd
import numpy as np
import json


__authors__ = ['floredelasteyrie','romainboyer']
__emails__  = ['flore.delasteyrie@essec.edu','romain.boyer1@essec.edu']

  
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


def loadPairs(path):
    data = pd.read_csv(path,delimiter='\t')
    pairs = zip(data['word1'],data['word2'],data['similarity'])
    return pairs


class mySkipGram:
    def __init__(self,sentences, nEmbed=30, negativeRate=5, winSize = 5, minCount = 2):
        # The sentences as the output of text2sentences()
        self.sentences = sentences
        # Number of neurons
        self.nEmbed = nEmbed 
        # Number of words to select for negative sampling
        self.negativeRate = negativeRate
        # The size of the window for the skip-gram
        self.winSize = winSize
        # The minimum number of times a word has to appear to be taken into account
        self.minCount = minCount

        

        ''' Create the dataframe with the words' input vector
        And the dictionnary with the words' frequencies '''
        
        print("Starting to create the databases")
        self.frequencies = {}
        self.all_words = []
        for sentence in tqdm(self.sentences):
            for word in sentence:
                if word in self.frequencies:
                    self.frequencies[word] += 1
                else:
                    self.frequencies[word] = 1
        self.frequencies = {k:self.frequencies[k] for k in self.frequencies if self.frequencies[k]>self.minCount}
        
        sentences2=[]
        sentence2=[]
        for sentence in self.sentences:
            sentence2=[]
            for word in sentence:
                if word in self.frequencies.keys():
                    self.all_words.append(word)
                    sentence2.append(word)
            sentences2.append(sentence2)
        
        sentences2 = [x for x in sentences2 if len(x)>1]
        self.all_words = list(set(self.all_words))
        
        ''' Create a dataframe with as columns all unique words and as their unique vector '''
        self.X = pd.DataFrame(np.eye(len(self.frequencies)), index=self.frequencies.keys())
        ''' Create a dataframe we'll be using for sentence2output '''
        self.Y = pd.DataFrame(np.zeros((len(self.frequencies),len(self.frequencies))),columns=self.frequencies.keys(), index=self.frequencies.keys())
        print("Databases created")        

        
        
        ''' Creates a dataframe with on each line the probability for being 
        near another word. '''
        
        print("Starting skip-gram")
        for sentence in tqdm(sentences2):
            for i in range(len(sentence)):
                for k in range(1,self.winSize):
                    
                    ''' To avoid errors for the last words of a sentence '''
                    if (i+k+1 <= len(sentence)):
                        #print(sentence[i],sentence[i+k])
                        ''' We add one for the next k words '''
                        self.Y.loc[sentence[i],sentence[i+k]] += 1 
                        
                    ''' To avoid errors for the first words of a sentence '''
                    if (i-k+1 > 0 ):
                        #print(sentence[i],sentence[i-k])
                        ''' We add one for the previous k words '''
                        self.Y.loc[sentence[i],sentence[i-k]] += 1 
                        
        ''' For each word, we divide the count of other words by the total counts  '''        
        self.Y = self.Y.div(self.Y.sum(axis=1),axis=0)
        print("Skip-gram over")
        
        self.X_shape = self.X.shape[0]
        self.w_1 = np.random.random((self.X_shape, self.nEmbed))
        self.w_2 = np.random.random((self.nEmbed, self.X_shape))

    def train(self,stepsize=0.4, epochs=10):
        print('Starting the training')
        # Start the loop on epochs
        for epoch_ in range(epochs):
            print('Epoch n°%d' % epoch_)
            loss = []
            time12=[]
            time24=[]
            time45=[]
            
            # Start the loop on the words
            for row in tqdm(range(self.X_shape)):
                
                '''FORWARD'''
                
                '''https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/'''
                time1 = time.clock()
                
                # Computation of all exits for row == row
                hidden_net = np.dot(self.X.iloc[row, :], self.w_1)
                hidden_out = expit(hidden_net)
                exit_net = np.dot(hidden_net, self.w_2)
                exit_out = expit(exit_net)
                
                # Computation of the loss for this row, and adding it to the list 'loss'
                loss_row = 0.5*(self.Y.iloc[row, :].values - exit_out)**2
                loss.append(loss_row.mean())
                
                # Creation of w_2_ to modify w_2
                w_2_ = np.zeros((self.nEmbed, self.X_shape))
                
                '''NEGATIVE SAMPLING'''
                time2 = time.clock()
                
                ### Creation of 'chosen_words' : 
                ### 5 random words (with probability 0) + the most probable word
                chosen_words = []
                
                while len(chosen_words) < self.negativeRate: 
                    # Choose a random word
                    rand = np.random.randint(0, len(self.all_words))
                    # Verify its probability is 0 and that it's not the row word and it's not already chosen
                    if (self.Y.iloc[row,:][self.all_words[rand]] == 0) & (self.Y.columns[row]!=self.all_words[rand]) & (self.all_words[rand] not in chosen_words):
                        chosen_words.append(self.all_words[rand])
                chosen_words.append(self.Y.iloc[0,:].argmax())
                    
                ### Retrieve the indexes of the chosen words (10^-5 sec par iteration)
                words_index = {}
                for word in chosen_words:
                    words_index[word] = self.X.index.get_loc(word)
                
                '''BACKWARD'''
                
                '''http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/'''
                time4 = time.clock()
                
                for key, value in words_index.items():
                    
                    w_2_[:, value] = -(self.Y.iloc[row, value]-exit_out[value])*exit_out[value]*(1-exit_out[value])*hidden_out  
                    self.w_1[row, :] -= stepsize*(-1)*(self.Y.iloc[row, value]-exit_out[value])*exit_out[value]*(1-exit_out[value])*self.w_2[:, value]*hidden_out*(1-hidden_out)                
                self.w_2 -= stepsize*w_2_   
                     
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
            print('Loss : '+str(loss.mean()))
            print('-'*30)
    pass

    def save(self, path):
        print("Start saving")
        parameters = self.__dict__.copy()
        
        # Converting types
        parameters['w_1'] = parameters['w_1'].tolist()
        parameters['w_2'] = parameters['w_2'].tolist()
        
        print(str(parameters))
        
        # Writing file
        with open(path, 'w') as file:
            file.write(str(parameters))
        file.close()
        print("Saving OK")
        pass

    def similarity(self,word1,word2):
        """
            computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """
        if (word1 not in self.all_words):
            print("'%s' not in the text" % word1)
            pass
        if (word2 not in self.all_words):
            print("'%s' not in the text" % word2)
            pass
        else:
            w1 = expit(np.dot(expit(np.dot(word1, self.w_1)), self.w_2))
            w2 = expit(np.dot(expit(np.dot(word2, self.w_1)), self.w_2))
            similarity = 1 - normalize(w1-w2, norm='l2')
            return similarity

    @staticmethod
    def load(path):
        print("Start loading")
        # Opening file
        with open(path, 'r') as file:
            parameters = file.read()
        file.close
        new_skip_gram = mySkipGram.__new__(mySkipGram)
        
        # Convert to dict
        print(parameters)
        
        #parameters['w_1'] = np.array(parameters['w_1']).reshape(parameters["X_shape"], parameters["nEmbed"])
        #parameters['w_2'] = np.array(parameters['w_2']).reshape(parameters["nEmbed"], parameters["X_shape"])
        
        # setting attributes to the new instance
        for attribute, attribute_value in parameters.items():
            setattr(new_skip_gram, attribute, attribute_value)
        print("Loading OK")
        return new_skip_gram
 
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        sentences = text2sentences(opts.text)
        sg = mySkipGram(sentences)
        sg.train()
        sg.save(opts.model)
        mySkipGram.load(opts.model)

    else:
        pairs = loadPairs(opts.text)

        sg = mySkipGram.load(opts.model)
        for a,b,_ in pairs:
            print(sg.similarity(a,b))