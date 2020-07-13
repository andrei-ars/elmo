from sklearn.metrics.pairwise import cosine_similarity

import tensorflow_hub as hub
import tensorflow as tf

elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

def elmo_vectors(x):
    embeddings=elmo(x, signature="default", as_dict=True)["elmo"]

    with tf.device('/device:CPU:0'):  # GPU:0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            # return average of ELMo features
            return sess.run(tf.reduce_mean(embeddings,1))


corpus=["I'd like an apple juice",
        "An apple a day keeps the doctor away",
         "Eat apple every day",
         "We buy apples every week",
         "We use machine learning for text classification",
         "Text classification is subfield of machine learning"]


elmo_embeddings=[]
print (len(corpus))
for i in range(len(corpus)):
    print (corpus[i])
    elmo_embeddings.append(elmo_vectors([corpus[i]])[0])

print ( elmo_embeddings, len(elmo_embeddings))
print(elmo_embeddings[0].shape)
sims = cosine_similarity(elmo_embeddings, elmo_embeddings)
print(sims)
print(sims.shape)