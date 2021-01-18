from model import TwoTowerModel
from data.random import get_random_data


model = TwoTowerModel()
user_data, item_data, y = get_random_data(4, 3)
model.compile(optimizer="adadelta", loss=model.batch_loss)
model.fit([user_data, item_data], y)
print(model.summary())

a = model.predict([user_data, item_data])
a