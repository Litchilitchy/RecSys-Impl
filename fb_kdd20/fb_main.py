from fb_kdd20.fb_model_revise import triplet_loss, TwoTowerModel_fb
from data.random import get_random_data
import tensorflow as tf

def main():
    optimizer = tf.keras.optimizers.Adam()
    model = TwoTowerModel_fb(4, 3)
    model.compile(optimizer, loss=triplet_loss)

    user_data, item_pos_data, y = get_random_data(4, 3)
    item_neg_data = item_pos_data

    model.fit([user_data, item_pos_data, item_neg_data], y)
    print(model.summary())
    model.save('two_tower_fb_revise')

    a = model.predict([user_data, item_pos_data, item_neg_data])
    print(a)
    print("Done")

if __name__=="__main__":
    main()
