import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import np_utils

df = pd.read_csv('../input/Iris.csv')
df = df.drop(['Id'], axis=1)
df['Species'] = df['Species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
train, test = train_test_split(df, test_size=.2)
X_train = train.drop(['Species'], 1).values
y_train = train['Species'].values
X_test = test.drop(['Species'], 1).values
y_test = test['Species'].values

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

nb_epoch = 100
nb_classes = 3
batch_size = 10

model = Sequential()
model.add(Dense(32, input_dim=4))
model.add(Activation('relu'))
model.add(Dropout(.2))


model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(.25))

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, nb_epoch=nb_epoch, verbose=2)

score = model.evaluate(X_test, y_test)
print('Score: ', score[1]*100)