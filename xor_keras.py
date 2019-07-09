import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import SGD

model = Sequential()
#input-hide
model.add(Dense(input_dim=2, units=2))
model.add(Activation('sigmoid'))
#hide-output
model.add(Dense(units=1))
model.add(Activation('sigmoid'))

X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0],[1],[1],[0]])
model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1))
model.fit(X,Y, epochs=8000, batch_size=2)
classes = model.predict_classes(X,batch_size=2)
prob = model.predict_proba(X, batch_size=2)

print('classified:')
print(Y == classes)
print()
print('output probability')
print(prob)
