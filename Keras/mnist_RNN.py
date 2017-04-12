from __future__ import print_function

import keras
from Keras.mnist_loaddata import load_data
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import SimpleRNN
from keras import initializers
from keras.optimizers import RMSprop
import numpy as np

batch_size = 32
num_classes = 10
epochs = 200
hidden_units = 100

learning_rate = 1e-6
clip_norm = 1.0

# the data, shuffled and split between train and test sets
data, label = load_data()
test_split=0.2
np.random.seed(1024)
np.random.shuffle(data)
np.random.seed(1024)
np.random.shuffle(label)

x_train = np.array(data[:int(len(data) * (1 - test_split))])
y_train = np.array(label[:int(len(data) * (1 - test_split))])
x_test = np.array(data[int(len(data) * (1 - test_split)):])
y_test = np.array(label[int(len(data) * (1 - test_split)):])

x_train = x_train.reshape(x_train.shape[0], -1, 1)
x_test = x_test.reshape(x_test.shape[0], -1, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print('Evaluate IRNN...')
model = Sequential()
model.add(SimpleRNN(hidden_units,
                    kernel_initializer=initializers.RandomNormal(stddev=0.001),
                    recurrent_initializer=initializers.Identity(gain=1.0),
                    activation='relu',
                    input_shape=x_train.shape[1:]))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
rmsprop = RMSprop(lr=learning_rate)
model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

scores = model.evaluate(x_test, y_test, verbose=0)
print('IRNN test score:', scores[0])
print('IRNN test accuracy:', scores[1])
