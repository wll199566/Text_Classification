"""
 This script includes all the functiions used for training and testing svm.
"""

import json
import pickle
import numpy as np
from sklearn import svm
from sklearn import metrics

def read_dataset_file(file_path):
    """
     Argument:
             file_path: the path of the file, like "../data/Amazon/amazon.cleaned.datasets
             /amazon.cleaned.vector/amazon.train.cleaned100%.vector.json"
     Output: 
             samples: numpy array of training samples
             target_labels: numpy array of training labels
    """
    samples = []
    target_labels = []
    
    with open(file_path, 'rt') as fin:
        for i, row in enumerate(fin, 1):
            #if i == 3:
            #    break
            sample_dict = json.loads(row)
            samples.append(sample_dict['avg_vec'])
            target_labels.append(int(sample_dict['label']))
    
    # convert the list to array
    samples = np.array(samples)
    target_labels = np.array(target_labels)

    return samples, target_labels

def train_model(model, feature_vector_train, labels_train, feature_vector_valid, labels_valid, output_model_file):
    """
     Arguments:
              model: sklearn classifier
              feature_vector_train, feature_vector_valid: training and validation data samples
              labels_train, labels_valid: training and validation labels
              output_model_file: .sav file to output the finalized model 
     Output:
              accuracy: the validation accuracy         
    """
    # fit the training dataset on the classifier
    model.fit(feature_vector_train, labels_train)

    # store the trained model
    pickle.dump(model, open(output_model_file, 'wb'))
    
    # predict the labels on validation dataset
    predictions = model.predict(feature_vector_valid)
    
    # get validation accuracy
    accuracy = metrics.accuracy_score(predictions, labels_valid)

    return accuracy

def test_model(model_file, feature_vector_test, labels_test):
    """
     Arguments:
              model_file: .sav file storing the trained model
              feature_vector_test: numpy array of test samples
              labels_test: numpy array of test labels 
     Output:
              accuracy: test accuracy         
    """
    # load the trained model
    model = pickle.load(open(model_file, 'rb'))

    # get the accuracy of the model on the test dataset
    accuracy = model.score(feature_vector_test, labels_test)

    return accuracy    

if __name__ == "__main__":
    
    train_file = "../data/Amazon/amazon.cleaned.datasets/amazon.cleaned.vector/amazon.train.cleaned100%.vector.json"
    valid_file = "../data/Amazon/amazon.cleaned.datasets/amazon.cleaned.vector/amazon.valid.vector.json"
    test_file = "../data/Amazon/amazon.cleaned.datasets/amazon.cleaned.vector/amazon.test.vector.json"
    
    train_X, train_Y = read_dataset_file(train_file)
    valid_X, valid_Y = read_dataset_file(valid_file)
    test_X, test_Y = read_dataset_file(test_file)

    model_folder = "../data/Amazon/amazon.trained.models"
    model_file = "/linear_svc.sav"    

    # TODO: grid search for searching C for linear SVC 

    model = svm.LinearSVC(penalty='l1', loss='hinge', dual=True, C=1.0, max_iter=1000)

    train_model(model, train_X, train_Y, valid_X, valid_Y, model_folder+model_file)

    #print(samples)
    #print(target_labels)