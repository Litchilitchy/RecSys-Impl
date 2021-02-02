import tensorflow as tf
import faiss
import numpy as np
from data.random import get_random_data

def retrieval_k(faiss_index, k, user_embedding):
    D,I=faiss_index.search(user_embedding,k)
    return I[0]


def random_test():
    idx_dict={}
    user_feature_num=4
    item_feature_num=3
    user_data, item_data, y = get_random_data(user_feature_num, item_feature_num)
    embed_user=tf.keras.models.load_model('user_embed',custom_objects={'tf':tf})
    embed_item=tf.keras.models.load_model('item_embed',custom_objects={'tf':tf})
    d = 128 # d is the dimension of item_embedding
    index = faiss.IndexFlatL2(d)
    for i in range(10):
        item=item_data[i]
        item=np.reshape(item,(1,item_feature_num))
        item_embedding=embed_item.predict(item)
        idx_dict[i]=i
        index.add(item_embedding)

    user_input=np.reshape(user_data[0],(1,user_feature_num))
    user_embedding=embed_user.predict(user_input)
    k=3
    retrieval_index=retrieval_k(index,k,user_embedding)
    item_idx=[idx_dict[i] for i in retrieval_index]
    print(item_idx)

if __name__=="__main__":
    random_test()