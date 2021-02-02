import tensorflow.keras as keras
import tensorflow as tf

def TwoTowerModel_fb(user_feature_num,item_feature_num):
    # user model
    u_D_300=keras.layers.Dense(300, activation='relu')
    u_D_128=keras.layers.Dense(128, activation='relu')
    u_N=keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))

    u_in=keras.layers.Input(shape=(user_feature_num,),name="user_input")    # user input layer
    u=u_in
    u=u_D_300(u)
    u=u_D_128(u)
    u =u_N(u)

    # item model - item_positive and item_negative share a model
    i_D_300 = keras.layers.Dense(300, activation='relu')
    i_D_128 = keras.layers.Dense(128, activation='relu')
    i_N = keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))

    ip_in = keras.layers.Input(shape=(item_feature_num,), name="item_pos_input")    # item_positive input layer
    ip = ip_in
    ip = i_D_300(ip)
    ip = i_D_128(ip)
    ip = i_N(ip)

    ig_in = keras.layers.Input(shape=(item_feature_num,), name="item_neg_input")    # item_negative input layer
    ig = ig_in
    ig = i_D_300(ig)
    ig = i_D_128(ig)
    ig = i_N(ig)

    inp=keras.layers.Dot(axes=1)
    sub=keras.layers.Subtract()
    u_ip=inp([u,ip])
    u_ig=inp([u,ig])
    out=sub([u_ip,u_ig])

    model=keras.Model([u_in,ip_in,ig_in],out)
    return model

def triplet_loss(y_true, y_pred):
    m = 0.5
    loss = tf.maximum(0.0, y_pred + m)
    return loss