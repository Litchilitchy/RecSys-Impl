import tensorflow as tf
import tensorflow.keras.backend as K

class TwoTowerModel_fb2020(tf.keras.models.Model):

    def __init__(self):
        super().__init__()
        self.t1a = tf.keras.layers.Dense(300, activation='relu')
        self.t1b = tf.keras.layers.Dense(300, activation='relu')
        self.t1c = tf.keras.layers.Dense(128, activation='relu')
        self.t1v = tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))

        self.t2a = tf.keras.layers.Dense(300, activation='relu')
        self.t2b = tf.keras.layers.Dense(300, activation='relu')
        self.t2c = tf.keras.layers.Dense(128, activation='relu')
        self.t2v = tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))

        self.ip = tf.keras.layers.Dot(axes=1)
        self.out = tf.keras.layers.Subtract()

    def call(self, inputs, training=None, mask=None):
        u, i , i_= inputs[0], inputs[1], inputs[2]
        u = self.t1a(u)
        u = self.t1b(u)
        u = self.t1c(u)
        u = self.t1v(u)

        i = self.t2a(i)
        i = self.t2b(i)
        i = self.t2c(i)
        i = self.t2v(i)

        i_ = self.t2a(i_)
        i_ = self.t2b(i_)
        i_ = self.t2c(i_)
        i_ = self.t2v(i_)

        out_positive = self.ip([u, i])
        out_negative = self.ip([u, i_])
        out = self.out([out_negative, out_positive])

        return out

    def triplet_loss(self, y_true, y_pred):
        m=0.5
        loss = tf.maximum(0.0,y_pred+m)
        return loss
