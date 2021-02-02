import tensorflow as tf
from data.random import get_random_data
from fb_kdd20.fb_model_revise import triplet_loss

def save_embed_model():
    model=tf.keras.models.load_model('two_tower_fb_revise',custom_objects={'triplet_loss':triplet_loss})

    embed_user=tf.keras.models.Model(model.get_layer('user_input').input,model.get_layer('lambda').output)
    embed_item=tf.keras.models.Model(model.get_layer('item_pos_input').input,model.get_layer('lambda_1').output)

    embed_user.save('user_embed')
    embed_item.save('item_embed')
    print("save 2 embed_model successfully!")

def random_test():
    embed_user=tf.keras.models.load_model('user_embed')
    embed_item=tf.keras.models.load_model('item_embed')

    user_data, item_pos_data, y = get_random_data(4, 3)
    user_embedding=embed_user(user_data)
    item_embedding=embed_item(item_pos_data)
    print("******user*******")
    print(user_embedding)
    print("******item*******")
    print(item_embedding)

if __name__=="__main__":
    save_embed_model()
    random_test()
