"""
  Get part of the training datasets for verifying the effectiveness and robustness
  of transfer learning model.
"""

import re
import numpy as np
import csv
import pandas as pd
from sklearn import model_selection

def get_part_training(dataset_path, proportion):
    """
     Argument:
             dataset_path: the folder name storing amazon or yelp dataset,
                           like '../data/Amazon' or '../data/Yelp'             
             proportion: the proportion of the number of desired training samples/ the total number of the original training samples
     Output:
             .csv files containing part of the training set.                      
    """    
    # get the full paths, word2idx and embedding matrix for each dataset
    #is_amazon = re.search(r'Amazon', dataset_path)
    #if is_amazon is not None:
    #    origin_dataset_path = dataset_path + "/amazon.cleaned.datasets/amazon.train.cleaned.csv"          
    #else:
    #    origin_dataset_path = dataset_path + "/yelp.cleaned.datasets/yelp.train.cleaned.csv"                     
    origin_dataset_path = dataset_path + "/yelp.full.cleaned/yelp.full.train.cleaned.csv"

    labels = []
    contexts = []

    # read in data from the original .csv file
    with open(origin_dataset_path, 'r') as fin:
        amazon_csv_train = csv.reader(fin)
        for row in amazon_csv_train:
            labels.append(row[0]) 
            contexts.append(row[1])

    # remove the header
    labels = labels[1:]
    contexts = contexts[1:]
    # split the whole training dataset into the part of it.
    train_x, _, train_y, _ = model_selection.train_test_split(contexts, labels, train_size=proportion)

    # create the data frame
    train_df = pd.DataFrame()
    train_df['label'] = train_y
    train_df['context'] = train_x  

    # get the output filename
    #if re.search(r'amazon', origin_dataset_path) is not None:
    #    output_file = dataset_path + "/amazon.cleaned.datasets/amazon.train.cleaned" + str(int(proportion*100)) + "%.csv"
    #else:
    #    output_file = dataset_path + "/yelp.cleaned.datasets/yelp.train.cleaned" + str(int(proportion*100)) + "%.csv"
    output_file = dataset_path + "/yelp.full.cleaned/yelp.full.cleaned" + str(proportion*100) + "%.csv"
    # write into the file.
    train_df.to_csv(output_file, encoding='utf-8', index=False)                  
    
if __name__ == "__main__":
    
    #datasets = ['../data/Amazon', '../data/Yelp']
    datasets = ['../data/Yelp-full']
    propotions = [0.000005, 0.00005, 0.0005, 0.005, 0.05, 0.2, 0.6]
    for dataset in datasets:
        for proportion in propotions:
            get_part_training(dataset, proportion)
    