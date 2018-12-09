"""
Compute average word vector for each context
"""

import re
import json
import codecs
import csv
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
        dataset_paths = [dataset_path + "/amazon.cleaned.datasets/amazon.train.cleaned.csv",\
                         dataset_path + "/amazon.cleaned.datasets/amazon.valid.cleaned.csv",\
                         dataset_path + "/amazon.cleaned.datasets/amazon.test.cleaned.csv",\
                         dataset_path + "/amazon.cleaned.datasets/amazon.train.cleaned5%.csv",\
                         dataset_path + "/amazon.cleaned.datasets/amazon.train.cleaned20%.csv",\
                         dataset_path + "/amazon.cleaned.datasets/amazon.train.cleaned60%.csv"]
        word2idx_file = "/amazon.vacab.dict.json"
        embedding_matrix_file = "/amazon.w2v.matrix.json"            
    else:
        dataset_paths = [dataset_path + "/yelp.cleaned.datasets/yelp.train.cleaned.csv",\
                         dataset_path + "/yelp.cleaned.datasets/yelp.valid.cleaned.csv",\
                         dataset_path + "/yelp.cleaned.datasets/yelp.test.cleaned.csv",\
                         dataset_path + "/yelp.cleaned.datasets/yelp.train.cleaned5%.csv",\
                         dataset_path + "/yelp.cleaned.datasets/yelp.train.cleaned20%.csv",\
                         dataset_path + "/yelp.cleaned.datasets/yelp.train.cleaned60%.csv"]                     
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
        
        # write the whole data into a .json file.
        # determine the output file name
        if re.search(r'amazon', file) is not None:
            if re.search(r'train', file) is not None:
                if re.search(r'5', file) is not None:
                    output_file = "/amazon.cleaned.datasets/amazon.train.cleaned5%.vector.json"
                elif re.search(r'20', file) is not None:
                    output_file = "/amazon.cleaned.datasets/amazon.train.cleaned20%.vector.json"
                elif re.search(r'60', file) is not None:
                    output_file = "/amazon.cleaned.datasets/amazon.train.cleaned60%.vector.json"
                else:            
                    output_file = "/amazon.cleaned.datasets/amazon.train.cleaned100%vector.json"
            elif re.search(r'valid', file) is not None:
                output_file = "/amazon.cleaned.datasets/amazon.valid.vector.json"
            else:
                output_file = "/amazon.cleaned.datasets/amazon.test.vector.json"
        else:
            if re.search(r'train', file) is not None:
                if re.search(r'5', file) is not None:
                    output_file = "/yelp.cleaned.datasets/yelp.train.cleaned5%.vector.json"
                elif re.search(r'20', file) is not None:
                    output_file = "/yelp.cleaned.datasets/yelp.train.cleaned20%.vector.json"
                elif re.search(r'60', file) is not None:
                    output_file = "/yelp.cleaned.datasets/yelp.train.cleaned60%.vector.json"
                else:            
                    output_file = "/yelp.cleaned.datasets/yelp.train.cleaned100%vector.json"
            elif re.search(r'valid', file) is not None:
                output_file = "/yelp.cleaned.datasets/yelp.valid.vector.json"
            else:
                output_file = "/yelp.cleaned.datasets/yelp.test.vector.json"
                
        
        label_list = []
        context_list = []
        
        # read in files
        with open(file, 'rt', encoding='utf-8') as fin:
            csv_reader = csv.reader(fin, delimiter=',')
            for i, row in enumerate(csv_reader):
                label_list.append(row[0])
                context_list.append(row[1])
        print("there are ", i, "reviews in ", file)

        # remove the header
        #label_list = label_list[1:]
        #context_list = context_list[1:]
        
        # get the average word vector representation for each sample
        # and write it to .json file.
        with open(dataset_path + output_file, 'wt') as fout:
            for i, context in enumerate(context_list):
                count = 0
                sum_vec = np.zeros(embedding_size)
                average_vec_dict = {}
                for token in tokenize(context):
                    index = int(word2idx[token])
                    sum_vec += embedding_matrix[index, :]
                    count += 1
                average_vec_dict['label'] = str(label_list[i])
                average_vec_dict['avg_vec'] = (sum_vec / count).tolist()
                json.dump(average_vec_dict, fout)
                fout.write('\n')    
        print("Finish ", file, "!")        

if __name__ =="__main__":

    datasets = ['../data/Amazon', '../data/Yelp']
    for dataset in datasets:
        compute_average_wordvec(dataset)



                        
    
     

