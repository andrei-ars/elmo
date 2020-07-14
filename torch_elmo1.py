#import scipy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from allennlp.commands.elmo import ElmoEmbedder

options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'
weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'
elmo = ElmoEmbedder(options_file=options_file, weight_file=weight_file)
# or
elmo = ElmoEmbedder() # By default
layer = 2 # output layer 

#tokens = ["I", "ate", "an", "apple", "for", "breakfast"]
#vectors = elmo.embed_sentence(tokens)

corpus = [["First", "name"], ["Second", "name"], ["Given", "name"], ["Last", "name"]]
elmo_embeddings = [np.mean(elmo.embed_sentence(tokens)[layer], axis=0) for tokens in corpus]
sims = cosine_similarity(elmo_embeddings, elmo_embeddings)
print(sims)
print(sims.shape)
#[[1.0000001  0.8678259  0.6146674  0.73796856]
# [0.8678259  0.99999976 0.5989584  0.7129174 ]
# [0.6146674  0.5989584  1.0000002  0.63083917]
# [0.73796856 0.7129174  0.63083917 1.0000002 ]]

corpus = [["street"], ["city"], ["small"], ["little"]]
#[[1.0000002  0.5662099  0.2778135  0.2351715 ]
# [0.5662099  0.9999998  0.24516752 0.23961285]
# [0.2778135  0.24516752 0.99999976 0.53718054]
# [0.2351715  0.23961285 0.53718054 0.9999998 ]]



#vecs1 = elmo.embed_sentence(["Large", "tree"])
#vecs2 = elmo.embed_sentence(["Huge", "tree"])
#vecs3 = elmo.embed_sentence(["Small", "tree"])
#scipy.spatial.distance.cosine(vecs1[2][0], vecs2[2][0])
#scipy.spatial.distance.cosine(vecs1[2][0], vecs3[2][0])
#scipy.spatial.distance.cosine(vecs2[2][0], vecs3[2][0])


corpus = [["long street"], ["the large city"], ["small button"], ["little button"]]
elmo_embeddings = [np.mean(elmo.embed_sentence(tokens)[layer], axis=0) for tokens in corpus]
sims = cosine_similarity(elmo_embeddings, elmo_embeddings)
print(sims)
