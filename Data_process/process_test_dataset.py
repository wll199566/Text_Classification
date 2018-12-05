import csv
import pandas as pd
from sklearn import preprocessing

def prepare_test_dataset(dataset):
    """
     Argument: dataset is in {"amazon", "yelp"}
    """
    
    if dataset == "amazon":
        data_root_path = '../data/Amazon/'
        output_file = 'amazon.test.csv'
        labels = []
        titles = []
        contexts = []
    else:    
        data_root_path = '../data/Yelp/'
        output_file = 'yelp.test.csv'
        labels = []
        contexts = []
    # read in the data from the original .csv file.
    with open(data_root_path + 'test.csv', 'r') as fin:
        amazon_csv_test = csv.reader(fin)
        for row in amazon_csv_test:
            labels.append(row[0])
            if dataset == "amazon":
                titles.append(row[1])
                contexts.append(row[2])
            else:
                contexts.append(row[1])    
    if dataset == "amazon":
        # Combine the title and the contexts together
        for i in range(len(contexts)):
            contexts[i] = (titles[i] + ' ' + contexts[i]).lower()

    # get the label-encoded target variable
    encoder = preprocessing.LabelEncoder()
    amazon_test_y = encoder.fit_transform(labels)

    # create the dataframe for test
    amazon_df_test = pd.DataFrame()
    amazon_df_test['label'] = amazon_test_y
    amazon_df_test['context'] = contexts

    # write the dataframe to .csv files
    amazon_df_test.to_csv(data_root_path + output_file, encoding='utf-8', index=False)

if __name__ == "__main__":

    datasets = ["amazon", "yelp"]
    for dataset in datasets:
        prepare_test_dataset(dataset)    
