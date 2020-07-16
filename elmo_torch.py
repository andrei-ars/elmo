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
    text = "huge button"
    texts = ["long street", "the large city", "small button", "large button"]
    """
    #corpus = [[t] for t in texts]
    ext_texts = [text] + texts
    corpus = [t.split() for t in ext_texts]
    sim_matrix = similarity_matrix(corpus)
    sims = sim_matrix[0,1:]
    index = np.argmax(sims)
    print("\ntext: {}".format(text))
    print("texts: {}".format(texts))
    print("similarities: {}".format(sims))
    print("index={}, cos_similarity={}".format(index, sims[index]))
    print("The most similary text to '{}' is text #{}: '{}'".format(text, index, texts[index]))
    return sims


if __name__ == "__main__":

    corpus = [["password"], ["city"], ["small"], ["little"]]
    print(similarity_matrix(corpus))

    texts = ["small town", "the city", "small button", "button above"]
    print(texts)
    corpus = [[text] for text in texts]
    print(similarity_matrix(corpus))
    corpus = [text.split() for text in texts]
    print(similarity_matrix(corpus))

    text = "the huge button"
    texts = ["long street", "the large city", "small button", "large button"]
    sims = find_most_similar(text, texts)

    text = "username"
    texts = ["site administrator", "simple button", "user name", "user login", "user password", "go to website"]
    sims = find_most_similar(text, texts)

    text = "admin"
    texts = ["simple button", "simple user", "just users", "site administrator", "user password", "open website"]
    sims = find_most_similar(text, texts)

    text = "enter a phrase"
    texts = ["simple button", "input text", "site administrator", "user name", "open website"]
    sims = find_most_similar(text, texts)