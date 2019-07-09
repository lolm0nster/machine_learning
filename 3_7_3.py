from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
N=300
X,Y = datasets.make_moons(N, noise=0.3)
x = np.array(X)
y = np.array(Y)
print(X)
print(y)
plt.scatter(x[:,0],x[:,1],c=y)
plt.show()
