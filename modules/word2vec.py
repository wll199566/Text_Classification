import numpy as np

def load_glove(glove_path):
    """
    Argument: 
             glove_path: the path storing the downloaded glove.6B.50d.txt
    Output:
             a dictionary like {key: word, value: glove vector}         
    """
    # read in the whole glove and store as a dictionay
    # {key: word, value: glove vector}
    with open(glove_path, 'r') as fin:
        glove = {line.split()[0]: np.fromiter(map(float, line.split()[1:]),dtype=np.float) 
             for line in fin}

    return glove

def word2vec(glove, word):
    """
      Arguments:
                glove: the glove dictionary like {key: word, value: glove vector} 
                word: the word waiting for transfering to the GloVe vector
      Outputs: 
                the corresponding GloVe vector for it
    """
    
    #print(list(glove.keys()))

    return glove[word]

if __name__ == "__main__":

    glove_vec = load_glove('../data/glove.6B/glove.6B.50d.txt')
    print( word2vec(glove_vec, 'the'))