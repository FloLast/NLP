#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
NATURAL LANGUAGE PROCESSING

Assignment N°1
"""

from __future__ import division

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
        for l in f:
            # Transform the sentence in words
            tokens = word_tokenize(l)
            # Lowercase all the characters of the words
            tokens = [w.lower() for w in tokens]
            # Delete numbers and punctations
            words = [word for word in tokens if word.isalpha()]
            # Delete words defined as stopwords
            words = [w for w in words if not w in nltk_stopwords]
            sentences.append(words)
    return sentences


def loadPairs(path):
    data = pd.read_csv(path,delimiter='\t')
    pairs = zip(data['word1'],data['word2'],data['similarity'])
    return pairs


def similarity(word1, word2, parameters):
        """
        Computes similiarity between the two words. Unknown words are mapped to one common vector
        """
        if (word1 not in parameters['X']):
            print("'%s' not in the text" % word1)
            pass
        if (word2 not in parameters['X']):
            print("'%s' not in the text" % word2)
            pass
        
        else:
            word1_ = np.zeros(len(parameters['X']))
            word1_[parameters['X'].index(word1)]=1
            word2_ = np.zeros(len(parameters['X']))
            word2_[parameters['X'].index(word2)]=1

            w1 = expit(np.dot(expit(np.dot(word1_, parameters['w_1'])), parameters['w_2']))
            w2 = expit(np.dot(expit(np.dot(word2_, parameters['w_1'])), parameters['w_2']))
            
            return (1+np.dot(w1, w2)/np.linalg.norm(w1) / np.linalg.norm(w2))/2



class mySkipGram:
    def __init__(self,sentences, nEmbed=50, negativeRate=7, winSize = 5, minCount = 3):
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
        
        # Creation of frequencies & all_words
        self.frequencies = {}
        self.all_words = []
        for sentence in self.sentences:
            for word in sentence:
                if word in self.frequencies:
                    self.frequencies[word] += 1
                else:
                    self.frequencies[word] = 1
        self.frequencies = {k:self.frequencies[k] for k in self.frequencies if self.frequencies[k]>self.minCount}
        
        # Remove words that are not taken into account
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
        
        # Create a dataframe with as columns all unique words and as their unique vector
        self.X = pd.DataFrame(np.eye(len(self.frequencies)), index=self.frequencies.keys())
        # Create a dataframe we'll be using for sentence2output
        self.Y = pd.DataFrame(np.zeros((len(self.frequencies),len(self.frequencies))),columns=self.frequencies.keys(), index=self.frequencies.keys())
        
        print("Databases created")
        print('-'*50)

        
        
        ''' Creates a dataframe with on each line the probability for being 
        near another word. '''
        print("Starting skip-gram")
        
        for sentence in sentences2:
            for i in range(len(sentence)):
                for k in range(1,self.winSize):
                    
                    # To avoid errors for the last words of a sentence
                    if (i+k+1 <= len(sentence)):
                        # We add one for the next k words
                        self.Y.loc[sentence[i],sentence[i+k]] += 1 
                        
                    ''' To avoid errors for the first words of a sentence '''
                    if (i-k+1 > 0 ):
                        #We add one for the previous k words
                        self.Y.loc[sentence[i],sentence[i-k]] += 1 
                        
        # For each word, we divide the count of other words by the total counts        
        self.Y = self.Y.div(self.Y.sum(axis=1),axis=0)
        
        print("Skip-gram over")
        print('-'*50)
        
        
        
        self.X_shape = self.X.shape[0]
        self.w_1 = np.random.random((self.X_shape, self.nEmbed))
        self.w_2 = np.random.random((self.nEmbed, self.X_shape))


    def train(self,stepsize, epochs):
        print('Starting the training')
        
        # Start the loop on epochs
        for epoch_ in range(epochs):
            print('Epoch n°%d' % epoch_)
            
            loss = []
            
            # Start the loop on the words
            for row in range(self.X_shape):
                
                '''FORWARD'''
                '''https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/'''                
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
                    
                ### Retrieve the indexes of the chosen words
                words_index = {}
                for word in chosen_words:
                    words_index[word] = self.X.index.get_loc(word)
                
                '''BACKWARD'''
                '''http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/'''
                
                for key, value in words_index.items():
                    
                    w_2_[:, value] = -(self.Y.iloc[row, value]-exit_out[value])*exit_out[value]*(1-exit_out[value])*hidden_out  
                    self.w_1[row, :] -= stepsize*(-1)*(self.Y.iloc[row, value]-exit_out[value])*exit_out[value]*(1-exit_out[value])*self.w_2[:, value]*hidden_out*(1-hidden_out)                
                self.w_2 -= stepsize*w_2_   
                     
                '''END'''       

            loss = np.array(loss) 
            print('Loss : '+str(loss.mean()))
            print('-'*30)        
    pass


    def save(self, path):
        print("Start saving")
        parameters = self.__dict__.copy()
        
        param_to_save = {'w_1':parameters['w_1'],
                         'w_2':parameters['w_2'],
                         'X':list(parameters['X'].index)}
        
        # Converting types
        param_to_save['w_1'] = param_to_save['w_1'].tolist()
        param_to_save['w_2'] = param_to_save['w_2'].tolist()
        
        # Writing file
        with open(path, 'w') as fp:
            json.dump(param_to_save, fp)
        fp.close()
        print("Saving OK")
        print('-'*50)
        pass

    

    @staticmethod
    def load(path):
        print("Start loading")
        # Opening file
        with open(path, 'r') as file:
            parameters = json.load(file)
        file.close
        
        print("Loading OK")
        print('-'*50)
        return parameters
 
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        sentences = text2sentences(opts.text)
        sg = mySkipGram(sentences)
        sg.train(1, 75)
        sg.save(opts.model)
        mySkipGram.load(opts.model)

    else:
        pairs = loadPairs(opts.text)

        parameters = mySkipGram.load(opts.model)
        for a,b,_ in pairs:
            print(similarity(a,b))
        