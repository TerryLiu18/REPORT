"""
Get pre-train embedding from GLOVE
"""
import numpy as np
import torch


def _read_glove_embedding(glove_file):
    '''Fun:read embedding from pre-train by glove
    '''
    ## map word to embedding
    word2embedding = dict()
    with open(glove_file, "r", encoding='utf-8') as f:
        for num,line in enumerate(f):
            values = line.split()
            word = values[0]
            emb = np.asarray(values[1:], dtype='float32')
            word2embedding[word] = emb

    return word2embedding
    

def _get_embedding(glove_file,word_map,embed_dim=100):
    """Fun:get the embedding matrix for our vocabulary
    """
    word2embedding = _read_glove_embedding(glove_file)
    
    ##load glove embedding to embedding matrix
    embedding_matrix = np.zeros((len(word_map), embed_dim))
    for word, i in word_map.items():
        embedding_vector = word2embedding.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    del word2embedding
    
    return torch.FloatTensor(embedding_matrix)
    

