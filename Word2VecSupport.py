# -*- coding: utf-8 -*-

import torch

class Word2Vec:
    def __init__(self, vocabList):
        self.vocabList = vocabList
        self.word_to_index_dictionary = {self.vocabList[i]: i for i in range(len(self.vocabList))}
    '''
    This function creates a dictionary that maps each word to an index. 
    '''
    def make_cbow_data(self, text, window_size, word_to_index):
        
        # identify the first focal word index
        listOfTuples = []
        focal_index = window_size
        left_index = 0
        right_index = 2 * window_size
    
        while right_index < len(text):
            focal_word = [word_to_index[text[focal_index]]]
            context = text[left_index:focal_index] + text[focal_index + 1 : right_index + 1]
            
            # change from word to indices
            for i in range(len(context)):
                context[i] = word_to_index[context[i]]
            
            # append context and focal_word as pytorch tensor
            listOfTuples.append((torch.tensor(context, dtype = torch.long), torch.tensor(focal_word, dtype = torch.long)))
            # update all the indices
            focal_index += 1
            left_index += 1
            right_index += 1
    
        return listOfTuples
    
    '''
    createTupleList function to create a data list that includes all tuples
    '''
    def createTupleList(self, text, window_size, word_to_index):
        data = []
        for i in range(len(text)):
            if (2 * window_size) < len(text[i]):
                listOfTuple = self.make_cbow_data(text[i], window_size, word_to_index)
                for j in range(len(listOfTuple)):
                    data.append(listOfTuple[j])
        return data
    
    '''
    getIndexFromWord function that return the index of the given word in word_to_index_dictionary
    '''
    def getIndexFromWord(self, word):
        return self.word_to_index_dictionary[word]
    
    '''
    getWordFromIndex function that return the word of the given index in word_to_index_dictionary
    '''
    def getWordFromIndex(self, indexVal):
        return list(self.word_to_index_dictionary.keys())[list(self.word_to_index_dictionary.values()).index(indexVal)]
    