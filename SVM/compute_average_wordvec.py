"""
Compute average word vector for each context
"""

import re
import json
import codecs
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize

from process_data import clean_data
from process_data import tokenize

def compute_average_wordvec(dataset_path, embedding_size=50):
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
        word2idx_file = "/amazon.vacab.dict.json"
        embedding_matrix_file = "/amazon.w2v.matrix.json"            
    else:
        dataset_paths = [dataset_path + "/yelp.train.csv",\
                         dataset_path + "/yelp.valid.csv",\
                         dataset_path + "/yelp.test.csv"]                     
        word2idx_file = "/yelp.vacab.dict.json"
        embedding_matrix_file = "/yelp.w2v.matrix.json"
        

    # load  the word2idx json file
    with open(dataset_path+word2idx_file, 'r') as fin:
        word2idx = json.load(fin)
    print("load word2idx")
    print("it has ", len(word2idx), " vocabularies")
    
    # load the embedding matrix file
    with codecs.open(dataset_path + embedding_matrix_file, 'r', encoding='utf-8') as fin:
        obj_text = fin.read()
        embedding_matrix = json.loads(obj_text)
        print("load embedding matrix")
        embedding_matrix = np.array(embedding_matrix)

    # read three files clean and tokenize them
    # then get the average vector for each review. 
    for file in dataset_paths:
        data = pd.read_csv(file, header=None, low_memory=False) 
        # the first column is label, the second one is context
        label_list = np.asarray(data.iloc[:, 0])
        context_list = np.asarray(data.iloc[:, 1])
        # get the average word vector representation for each sample
        average_vec_list = []
        for context in context_list:
            count = 0
            sum_vec = np.zeros(embedding_size)
            for token in tokenize(clean_data(context)):
                index = int(word2idx[token])
                sum_vec += embedding_matrix[index, :]
                count += 1
            average_vec_list.append(sum_vec / count)    

        # write the whole data into a .csv file.
        # determine the output file name
        if re.search(r'amazon', file) is not None:
            if re.search(r'train', file) is not None:
                output_file = "/amazon.train.vector.csv"
            elif re.search(r'valid', file) is not None:
                output_file = "/amazon.valid.vector.csv"
            else:
                output_file = "/amazon.test.vector.csv"
        else:
            if re.search(r'train', file) is not None:
                output_file = "/yelp.train.vector.csv"
            elif re.search(r'valid', file) is not None:
                output_file = "/yelp.valid.vector.csv"
            else:
                output_file = "/yelp.test.vector.csv"

        sample_df = pd.DataFrame()
        sample_df['label'] = label_list
        sample_df['avg_vector'] = average_vec_list
        sample_df.to_csv(dataset_path + output_file, encoding='utf-8', index=False)


if __name__ =="__main__":

    datasets = ['../data/Amazon', '../data/Yelp']
    for dataset in datasets:
        compute_average_wordvec(dataset)



                        
    
     

