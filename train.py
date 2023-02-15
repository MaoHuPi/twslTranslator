'''
2023 © MaoHuPi
twslTranslator/train.py
'''

'''
load train data
'''
import json
import os
import numpy as np

def tidy(data):
    data2 = []
    for point in data:
        data2.append(point[0])
        data2.append(point[1])
    return(data2)

path = '.'
dataDir = path + '/data'
x_train = []; y_train = []
x_test = []; y_test = []
tags = os.listdir(dataDir)
for handType in tags:
    handTypeDir = f'{dataDir}/{handType}'
    fileNames = os.listdir(handTypeDir)
    for i in range(len(fileNames)):
        fileName = fileNames[i]
        if i < 20:
            [xTarget, yTarget] = [x_train, y_train]
        else:
            [xTarget, yTarget] = [x_test, y_test]
        try:
            file = open(f'{handTypeDir}/{fileName}', 'r', encoding = 'utf-8')
            data = file.read()
            file.close()
            data = json.loads(data)
            data = tidy(data)
            xTarget.append(data)
            yTarget.append(tags.index(handType))
        except Exception as e:
            print(e)
x_train = np.array(x_train)
y_train = np.array(y_train)
y_train_one_hot = np.eye(len(tags))[y_train]
y_test_one_hot = np.eye(len(tags))[y_test]

'''
copy from teacher's example
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Adagrad, SGD

def createModel():
    model = Sequential()
    model.add(Dense(500, Activation('relu'), input_dim = x_train.shape[1]))
    for i in range(5):
        model.add(Dense(100))
    for i in range(2):
        model.add(Dense(500))
    model.add(Dense(len(tags), Activation('softmax')))
    return model
model = createModel()
model.summary()

optim = Adam(lr = 0.005)
# optim = Adagrad(lr = 0.001)
# optim = SGD(lr = 0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=optim,
              metrics=['acc'])

history = model.fit(x_train, y_train_one_hot,
                    batch_size=100,
                    epochs=25,
                    verbose=1,
                    shuffle=True,
                    validation_split=0.1)

history_dict = history.history
history_dict.keys()

import matplotlib.pyplot as plt

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs_ = range(1,len(acc)+1)

# plt.plot(epochs_ , loss , label = 'training loss')
# plt.plot(epochs_ , val_loss , label = 'val los')
# plt.title('training and val loss')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.legend()
# plt.show()

# plt.clf()
# plt.plot(epochs_ , acc , label='train accuracy')
# plt.plot(epochs_ , val_acc , label = 'val accuracy')
# plt.title('train and val acc')
# plt.xlabel('epochs')
# plt.ylabel('acc')
# plt.legend()
# plt.show()

# from sklearn.metrics import accuracy_score
# pred = np.argmax(model.predict(x_test), axis=1)
# print(accuracy_score(y_test, pred))
# print([tags[i] for i in pred])

'''
model method
'''
def mode(*array):
    number, times = np.unique(array, return_counts = True)
    index = np.argmax(times)
    return(number[index])

def predict(datas):
    x_test = []
    for data in datas:
        x_test.append(tidy(data))
    index = mode(*np.argmax(model.predict(x_test), axis=1))
    return(tags[index])