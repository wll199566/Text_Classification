# read the data in
import csv
import pandas as pd
from sklearn import model_selection
from sklearn import preprocessing

def prepare_train_dataset(dataset):
    """
     Argument: dataset in {"amazon", "yelp"}
    """
    
    #if dataset == "amazon":
    #    data_root_path = '../data/Amazon/'
    #    train_output = "amazon.train.csv"
    #    valid_output = "amazon.valid.csv"
    #    labels = []
    #    titles = []
    #    contexts = []
    #else:
    #    data_root_path = '../data/Yelp/'
    #    train_output = "yelp.train.csv"
    #    valid_output = "yelp.valid.csv"
    #    labels = []
    #    contexts = []
    data_root_path = '../data/Yelp-full'
    train_output = "/yelp.full.train.csv"
    valid_output = "/yelp.full.valid.csv"
    labels = []
    contexts = []

    # read in data from the original .csv file
    with open(data_root_path + '/train.csv', 'r') as fin:
        amazon_csv_train = csv.reader(fin)
        for row in amazon_csv_train:
            labels.append(row[0])
            #if dataset == "amazon":
            #    titles.append(row[1])
            #    contexts.append(row[2])
            #else: 
            #    contexts.append(row[1])    
            contexts.append(row[1].lower())
    # Combine the title and the contexts together
    #if dataset == "amazon":
    #    for i in range(len(contexts)):
    #        contexts[i] = (titles[i] + ' ' + contexts[i]).lower()

    # get rid of the header
    labels = labels[1:]
    contexts = contexts[1:]

    # split the training dataset into training and validataion datasets
    amazon_train_x, amazon_valid_x, amazon_train_y, amazon_valid_y = model_selection.train_test_split(contexts, labels)

    # get the label-encoded target variable
    encoder = preprocessing.LabelEncoder()
    amazon_train_y = encoder.fit_transform(amazon_train_y)
    amazon_valid_y = encoder.fit_transform(amazon_valid_y)

    # create the dataframe for train
    amazon_df_train = pd.DataFrame()
    amazon_df_train['label'] = amazon_train_y
    amazon_df_train['context'] = amazon_train_x

    # create the dataframe for valid
    amazon_df_valid = pd.DataFrame()
    amazon_df_valid['label'] = amazon_valid_y
    amazon_df_valid['context'] = amazon_valid_x

    # write the dataframe to .csv files
    amazon_df_train.to_csv(data_root_path + train_output, encoding='utf-8', index=False)
    amazon_df_valid.to_csv(data_root_path + valid_output, encoding='utf-8', index=False)

if __name__ == "__main__":

    #datasets = ["amazon", "yelp"]
    #for dataset in datasets:
    #    prepare_train_dataset(dataset)  
    prepare_train_dataset("yelp-full")