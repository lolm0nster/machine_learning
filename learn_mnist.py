from sklearn import datasets
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Activation
from keras.optimizers import SGD  
from keras.models import Sequential
import keras

mnist = fetch_openml('mnist_784', version=1,)
#(X_train,Y_train),(X_test, Y_test) = mnist.load_data()
n = len(mnist.data)
N = 10000
indices = np.random.permutation(range(n))[:N] 
#print(indices)
X = mnist.data[indices]
y = mnist.target[indices]
Y = np.eye(10)[y.astype(int)]
print(X,Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)

'''
モデル設定
'''
n_in = len(X[0])
n_hidden = 2000
n_out = len(Y[0])

model = Sequential()
model.add(Dense(n_out))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
	      optimizer = SGD(lr=0.01),
	      metrics = ['accuracy'])


'''
モデル学習
'''
epochs = 1000
batch_size = 100

model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)


'''
予測精度の評価
'''
loss_and_metrics = model.evaluate(X_test, Y_test)
print(loss_and_metrics)



