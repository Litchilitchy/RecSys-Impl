from model import TwoTowerModel_fb2020
import numpy as np

def get_random_data(user_feature_num, item_feature_num):
    """
    user: [0, 1000)
    uf1: [0, 1] gender
    uf2: [0, 10)
    uf3: [0, 20)
    item: [0,10000)
    if1: [0, 50)
    if2: [0, 60)
    """
    def get_random_ndarray(data_size, dis_list, feature_num):
        data_list = []
        for i in range(feature_num):
            arr = np.random.randint(dis_list[i], size=data_size)
            data_list.append(arr)
        data = np.array(data_list)
        return np.transpose(data, axes=(1, 0))
    uf_dis, if_dis, data_size = [1000, 2, 10, 20], [10000, 50, 60], 1000
    y = np.zeros(data_size)
    for i in range(int(data_size/10)):
        y[i] = 1

    return get_random_ndarray(data_size, uf_dis, feature_num=user_feature_num), \
        get_random_ndarray(data_size, if_dis, feature_num=item_feature_num), y

model = TwoTowerModel_fb2020()
user_data, item_pos_data, y = get_random_data(4,3)
item_neg_data=item_pos_data
model.compile(optimizer="adadelta", loss=model.triplet_loss)
model.fit([user_data, item_pos_data, item_neg_data], y)
print(model.summary())
model.save('two_tower_fb')

a = model.predict([user_data, item_pos_data, item_neg_data])

print(a)