"""
 This script contains all the data process functions for SVM part.
 Include tokenize and compute the average word vector for each setence in three sets.
"""
# add the module folder to system path
import sys
sys.path.append('../modules')

import re
import json
import codecs
import pandas as pd
import numpy as np
from word2vec import load_glove
from nltk.tokenize import word_tokenize

def clean_data(string):
    """
     Argument:
             string: the string to be cleaned
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r"\'m", " \'m", string)
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

def tokenize(sentence):
    """
     Argument:
             sentence: the sentence to be tokenized
     Output: 
             the tokenized word list for the sentence
    """
    return word_tokenize(sentence)

def get_dataset_vocab(dataset_path):
    """
     Argument:
             dataset_path: the folder name storing amazon or yelp dataset,
                           like '../data/Amazon' or '../data/Yelp'
     Output:
             vocab2idx: dictionary {vocab: index} 
             len(token_list): the number of vocabularies                      
    """    
    # get the full paths for each dataset
    is_amazon = re.search(r'Amazon', dataset_path)
    if is_amazon is not None:
        dataset_paths = [dataset_path + "/amazon.train.csv",\
                         dataset_path + "/amazon.valid.csv",\
                         dataset_path + "/amazon.test.csv"]
    else:
        dataset_paths = [dataset_path + "/yelp.train.csv",\
                         dataset_path + "/yelp.valid.csv",\
                         dataset_path + "/yelp.test.csv"]    
    
    # read three files tokenize them and get the vocabulary for the dataset
    token_set = set()
    for file in dataset_paths:
        data = pd.read_csv(file, header=None, low_memory=False) 
        # the first column is label, the second one is context
        context_list = np.asarray(data.iloc[:, 1])
        # get all the tokens for all the datasets (here, we have repeated tokens!)
        for context in context_list:
            #try:
            #    tokenize(clean_data(context))
            #except TypeError:
            #    print(context) 
            for token in tokenize(clean_data(context)):
                token_set.add(token)
    
    print("this dataset has ", len(token_set), " vocabularies")
    
    # iterate the set and construct the index for each vacobulary
    vocab2idx = {}
    for i, token in enumerate(token_set):
        vocab2idx[token] = i

    return vocab2idx, len(token_set)    

def construct_w2v_matrix(glove, embedding_dim, vocab_idx_dict, vocab_size):
    """
     Arguments:
              glove: the loaded glove dictionary
              embedding_dim: the embedding dim of glove vector
              vocab_idx_dict: the dictionary of {vocabulary: index}
              vocab_size: the number of vocabularies for this dataset 
     Output:
              W: word matrix, each row is the representative vector for
                 each word.         
    """
    W = np.zeros(shape=(vocab_size, embedding_dim), dtype='float32')
    for word, index in vocab_idx_dict.items():
        try:
            W[index,:] = glove[word]
        except KeyError:
            # if the word not in the glove, then use random vector whose
            # variant is the similar to the glove vectore.
            W[index,:] = np.random.uniform(-0.25,0.25,embedding_dim)

    return W        
    

if __name__ == "__main__":
    #cleaned_sentence = clean_data("Friendly staff, same starbucks fair you get anywhere else. Sometimes the lines can get long.")    
    #tokened_sentence = tokenize(cleaned_sentence)
    #print(cleaned_sentence)
    #print(tokened_sentence)
    vocab_dict, vocab_size = get_dataset_vocab("../data/Amazon")
    with open("../data/Amazon/amazon.vacab.dict.json", 'w') as fout:
        json.dump(vocab_dict, fout)
    glove = load_glove('../data/glove.6B/glove.6B.50d.txt')    
    W = construct_w2v_matrix(glove, 50, vocab_dict, vocab_size)    
    W_list = W.tolist()
    json.dump(W_list, codecs.open("../data/Amazon/amazon.w2v.matrix.json",'w', encoding='utf-8'))
