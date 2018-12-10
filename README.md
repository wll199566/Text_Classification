# Text Classifiacation

To use this code, please do the following preparation for the data:

1. in this project folder, construct a "data" folder.
2. in the data folder, please download the [Amazon training set](https://drive.google.com/file/d/0Bz8a_Dbh9QhbZEwyekt6Q08zMFE/view), [Amazon test set](https://drive.google.com/file/d/0Bz8a_Dbh9QhbVVlPUHFNWTQ4c0k/view) and [Yelp dataset](https://drive.google.com/drive/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M).
3. unzip them, and use the process_test_dataset.py and process_training_dataset.py to create .csv files for training, validation and test file for both datasets. It also available [here](https://drive.google.com/file/d/1-g95Kfl2aidPhpeA_DNEJLtqHd60uktp/view?usp=sharing) and [here](https://drive.google.com/file/d/11kN2iPvC-7Ly2zG9fOvP-u1Yrrc8-vmr/view?usp=sharing).
4. download the GloVe word vectors from [Stanford NLP group website](http://nlp.stanford.edu/data/glove.6B.zip).
5. construct "glove.6B" under data folder, and move all the glove.6B.\*\*.txt to that folder.

Additional resources:

1. [Amazon](https://drive.google.com/file/d/14K7Nk_pysCmm5EIP9AX6dAdk13bWrO0r/view?usp=sharing) and [Yelp](https://drive.google.com/file/d/1qhFHFnCeRSjgWFlAye1n76sjZUaXyO2J/view?usp=sharing) cleaned and tokenized datasets.
2. [Amazon](https://drive.google.com/file/d/1e---JP3vnaJ8uEaTX27JzMb3qDOixitB/view?usp=sharing) and [Yelp](https://drive.google.com/file/d/1T3yXU9fQUEOrqTz15aLrTf347q4-ujNj/view?usp=sharing) word2idx and word embedding matrix. 
3. Different proportions (5%, 20%, 60%) of the datasets provided in 1, used for transfer learning. ([Amazon](https://drive.google.com/file/d/1bhohvpuX9CIDNZiWIIP2UtqukUrAMPc6/view?usp=sharing), [Yelp](https://drive.google.com/file/d/1PQFLpes6roqM9-toYhqehYnjQ8p9gPsE/view?usp=sharing)).