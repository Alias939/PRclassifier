import keras
from keras import datasets, layers, models


#model
shape=(1,1,1) 

premise= layers.Input(shape=shape)
hypothesis= layers.Input(shape=shape)

xp=layers.Dense(100,activation='tanh')(premise)
xh=layers.Dense(100,activation='tanh')(hypothesis)

conc=layers.concatenate([xp,xh])

y= layers.Dense(100,activation='tanh')(conc)
y= layers.Dense(100,activation='tanh')(y)
y= layers.Dense(100,activation='tanh')(y)

output=layers.Flatten()(y)
output = layers.Dense(3, activation='softmax')(output)

model = keras.models.Model([premise,hypothesis], output)
model.summary()