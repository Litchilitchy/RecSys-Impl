import tensorflow as tf
import tensorflow.keras.backend as K


class TwoTowerModel(tf.keras.models.Model):
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

        self.out = tf.keras.layers.Dot(axes=1)

    def call(self, inputs, training=None, mask=None):
        u, i = inputs[0], inputs[1]
        u = self.t1a(u)
        u = self.t1b(u)
        u = self.t1c(u)
        u = self.t1v(u)
        i = self.t2a(i)
        i = self.t2b(i)
        i = self.t2c(i)
        i = self.t2v(i)
        out = self.out([u, i])
        return out

    def batch_loss(self, y_true, y_pred_batch):
        bottom = tf.reduce_sum(tf.exp(y_pred_batch))
        batch_softmax = tf.exp(y_pred_batch) / bottom
        batch_log_likelihood = tf.math.log(batch_softmax)
        loss = -tf.reduce_sum(batch_log_likelihood)
        return loss
