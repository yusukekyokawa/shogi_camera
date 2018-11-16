import keras
from keras import models
from keras import layers
from keras.utils import to_categorical
import numpy as np
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.callbacks import EarlyStopping

komas = np.array([
	np.load('./army/0_gyoku.npz')['arr_0'],
	np.load('./army/1_kin.npz')['arr_0'],
	np.load('./army/2_gin.npz')['arr_0'],
	np.load('./army/3_kei.npz')['arr_0'],
	np.load('./army/4_kyou.npz')['arr_0'],
	np.load('./army/5_kaku.npz')['arr_0'],
	np.load('./army/6_hisya.npz')['arr_0'],
	np.load('./army/7_fu.npz')['arr_0']
	])

ou = np.load('./army/14_ou.npz')['arr_0']
train_koma = ou[0:20000:40]
test_koma = ou[20000:24000:40]
train_label = [0] * 500
test_label = [0] * 100
i = 1
for koma in komas:
	train_label = train_label + [i] * 500
	test_label = test_label + [i] * 100
	train_temp = koma[0:20000:40]
	test_temp = koma[20000:24000:40]
	train_koma = np.vstack((train_koma,train_temp))
	test_koma = np.vstack((test_koma,test_temp))
	i+=1

train_koma = train_koma.astype('float32')
test_koma = test_koma.astype('float32')
train_koma = train_koma/255.0
test_koma = test_koma/255.0

train_label = to_categorical(train_label)
test_label = to_categorical(test_label)

# CNNを構築
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=train_koma.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(9))       # クラスは2個
model.add(Activation('softmax'))

# コンパイル
model.compile(loss='categorical_crossentropy',
              optimizer='SGD',
              metrics=['accuracy'])

# 実行。出力はなしで設定(verbose=0)。
history = model.fit(train_koma, train_label, batch_size=5, epochs=5,
                   validation_data = (test_koma, test_label), verbose = 1, callbacks=[EarlyStopping()])

model_json_str = model.to_json()
open('kg.json', 'w').write(model_json_str)
model.save_weights('kg.h5')
