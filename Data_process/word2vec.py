import numpy as np

def word2vec(glove_path ,word):
    """
      Arguments:
                glove_path: the path storing the downloaded glove.6B.50d.txt
                word: the word waiting for transfering to the GloVe vector
      Outputs: 
                the corresponding GloVe vector for it
    """
    # read in the whole glove and store as a dictionay
    # {key: word, value: glove vector}
    with open(glove_path, 'r') as fin:
        glove = {line.split()[0]: np.fromiter(map(float, line.split()[1:]),dtype=np.float) 
             for line in fin}

    #print(list(glove.keys()))

    return glove[word]

if __name__ == "__main__":

    print( word2vec('../data/glove.6B/glove.6B.50d.txt', 'the'))