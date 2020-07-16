"""
requirements
pip install numpy
pip install sklearn
pip install allennlp
pip install allennlp-models
"""

#import scipy
import logging
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from allennlp.commands.elmo import ElmoEmbedder


logging.info("Loading of ELMo...")
elmo = ElmoEmbedder() # By default
# or
#options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'
#weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'
#elmo = ElmoEmbedder(options_file=options_file, weight_file=weight_file)
#tokens = ["I", "ate", "an", "apple", "for", "breakfast"]
#vectors = elmo.embed_sentence(tokens)
#scipy.spatial.distance.cosine(vecs1[2][0], vecs2[2][0])


def similarity_matrix(corpus):
    """
    corpus = [["First", "name"], ["Second", "name"], ["Given", "name"], ["Last", "name"]]
    or
    corpus = [["First name"], ["Second name"], ["Given name"], ["Last name"]]
    """
    layer = 2   # the output layer 
    logging.info("The calculation of embeddings")
    elmo_embeddings = [np.mean(elmo.embed_sentence(tokens)[layer], axis=0) for tokens in corpus]
    sims = cosine_similarity(elmo_embeddings, elmo_embeddings)
    return sims


def find_most_similar(text, texts):
    """
    text = "click the label"
    texts = ["long street", "the large city", "small button", "little long button"]
    """
    #corpus = [[t] for t in texts]
    texts = [text] + texts
    corpus = [t.split() for t in texts]
    ext_sims = similarity_matrix(corpus)
    print(ext_sims)
    index = np.argmax(ext_sims[0,1:]) - 1
    sim = ext_sims[0, index+1]
    return index, sim



if __name__ == "__main__":

    corpus = [["street"], ["city"], ["small"], ["little"]]
    print(similarity_matrix(corpus))
    #[[1.0000002  0.5662099  0.2778135  0.2351715 ]
    # [0.5662099  0.9999998  0.24516752 0.23961285]
    # [0.2778135  0.24516752 0.99999976 0.53718054]
    # [0.2351715  0.23961285 0.53718054 0.9999998 ]]

    texts = ["long street", "the large city", "small button", "little long button"]
    print(texts)
    corpus = [[text] for text in texts]
    print(similarity_matrix(corpus))
    corpus = [text.split() for text in texts]
    print(similarity_matrix(corpus))
    """
    [[1.         0.75627625 0.66477364 0.7094424 ]
     [0.75627625 0.99999964 0.66632223 0.7353818 ]
     [0.66477364 0.66632223 1.0000001  0.8111696 ]
     [0.7094424  0.7353818  0.8111696  0.99999976]]
    
    [[1.0000002  0.5136549  0.45660293 0.6065997 ]
     [0.5136549  1.         0.552847   0.4276201 ]
     [0.45660293 0.552847   0.9999996  0.71048474]
     [0.6065997  0.4276201  0.71048474 1.0000001 ]]
    """
    text = "tiny label"
    index, sim = find_most_similar(text, texts)
    print("index={}, cos_similarity={}".format(index, sim))