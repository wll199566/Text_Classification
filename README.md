# Text Classifiacation

To use there code, please do the following preparation for the data:

1. in this project folder, construct a "data" folder.
2. in the data folder, please download the [Amazon training set](https://drive.google.com/file/d/0Bz8a_Dbh9QhbZEwyekt6Q08zMFE/view), [Amazon test set](https://drive.google.com/file/d/0Bz8a_Dbh9QhbVVlPUHFNWTQ4c0k/view) and [Yelp dataset](https://drive.google.com/drive/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M).
3. unzip them, and use the process_test_dataset.py and process_training_dataset.py to create .csv files for training, validation and test file for both datasets. It also available [here](https://drive.google.com/file/d/1-g95Kfl2aidPhpeA_DNEJLtqHd60uktp/view?usp=sharing) and [here](https://drive.google.com/file/d/11kN2iPvC-7Ly2zG9fOvP-u1Yrrc8-vmr/view?usp=sharing).
4. download the GloVe word vectors from [Stanford NLP group website](http://nlp.stanford.edu/data/glove.6B.zip).
5. construct "glove.6B" under data folder, and move all the glove.6B.\*\*.txt to that folder.