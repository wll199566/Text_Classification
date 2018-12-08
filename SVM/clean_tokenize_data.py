"""
Create the clean and tokenized datasets
"""

import re
import json
import codecs
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize

from process_data import clean_data
from process_data import tokenize

def clean_tokenize_datasets(dataset_path):
    """
     Argument:
             dataset_path: the folder name storing amazon or yelp dataset,
                           like '../data/Amazon' or '../data/Yelp'
             embedding_size: the dimension of word vector
     Output:
             .csv files containing (label, average word vector)                      
    """    
    # get the full paths, word2idx and embedding matrix for each dataset
    is_amazon = re.search(r'Amazon', dataset_path)
    if is_amazon is not None:
        dataset_paths = [dataset_path + "/amazon.train.csv",\
                         dataset_path + "/amazon.valid.csv",\
                         dataset_path + "/amazon.test.csv"]          
    else:
        dataset_paths = [dataset_path + "/yelp.train.csv",\
                         dataset_path + "/yelp.valid.csv",\
                         dataset_path + "/yelp.test.csv"]                     
            
    # read three files clean and tokenize them
    # then get the average vector for each review. 
    for file in dataset_paths:
        data = pd.read_csv(file, header=None, low_memory=False) 
        # the first column is label, the second one is context
        label_list = np.asarray(data.iloc[:, 0])
        context_list = np.asarray(data.iloc[:, 1])
        # get the average word vector representation for each sample
        cleaned_context_list = []
        for context in context_list:
            cleaned_context_list.append(tokenize(clean_data(context)))    

        # write the whole data into a .csv file.
        # determine the output file name
        if re.search(r'amazon', file) is not None:
            if re.search(r'train', file) is not None:
                output_file = "/amazon.train.cleaned.tokenized.csv"
            elif re.search(r'valid', file) is not None:
                output_file = "/amazon.valid.cleaned.tokenized.csv"
            else:
                output_file = "/amazon.test.cleaned.tokenized.csv"
        else:
            if re.search(r'train', file) is not None:
                output_file = "/yelp.train.cleaned.tokenized.csv"
            elif re.search(r'valid', file) is not None:
                output_file = "/yelp.valid.cleaned.tokenized.csv"
            else:
                output_file = "/yelp.test.cleaned.tokenized.csv"

        clean_token_df = pd.DataFrame()
        clean_token_df['label'] = label_list
        clean_token_df['cleaned_tokenized_review'] = cleaned_context_list
        clean_token_df.to_csv(dataset_path + output_file, encoding='utf-8', index=False)


if __name__ =="__main__":

    datasets = ['../data/Amazon', '../data/Yelp']
    for dataset in datasets:
        clean_tokenize_datasets(dataset)
