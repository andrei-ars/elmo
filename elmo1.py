from sklearn.metrics.pairwise import cosine_similarity

import tensorflow_hub as hub
import tensorflow as tf

elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

def elmo_vectors(x):
    embeddings = elmo(x, signature="default", as_dict=True)["elmo"]

    with tf.device('/device:CPU:0'):  # GPU:0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            # return average of ELMo features
            return sess.run(tf.reduce_mean(embeddings, 1))


#corpus=["I'd like an apple juice",
#        "An apple a day keeps the doctor away",
#         "Eat apple every day",
#         "We buy apples every week",
#         "We use machine learning for text classification",
#         "Text classification is subfield of machine learning"]

"""
corpus=["First name",
        "Second name",
         "Given name",
         "Last name",
         "Surname"]
#[[1.0000004  0.861894   0.6049657  0.72111595 0.4558046 ]
# [0.861894   1.         0.5898464  0.68965757 0.43921626]
# [0.6049657  0.5898464  1.         0.60682935 0.41435632]
# [0.72111595 0.68965757 0.60682935 1.         0.50063854]
# [0.4558046  0.43921626 0.41435632 0.50063854 0.9999999 ]]
"""

#corpus=["The day was bad",
#        "It was the worst day in my life",
#        "Google like search in internet"]
#[[1.         0.65336514 0.26608497]
# [0.65336514 0.9999999  0.295992  ]
# [0.26608497 0.295992   0.9999999 ]]

corpus=["street",
        "city",
        "small",
        "little"]



elmo_embeddings=[]
print(len(corpus))
for i in range(len(corpus)):
    vectors = elmo_vectors([corpus[i]])
    elmo_embeddings.append(vectors[0])
    print("i={}, text={}".format(i, corpus[i]))
    print("shape: {}".format(vectors))

print(elmo_embeddings, len(elmo_embeddings))
print(elmo_embeddings[0].shape)
sims = cosine_similarity(elmo_embeddings, elmo_embeddings)
print(sims)
print(sims.shape)