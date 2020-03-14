import numpy as np
import matplotlib.pyplot as plt

def Perceptron(x,y):
    n,d=x.shape
    x_Wbias = np.c_[x, np.ones([n])]
    w = np.zeros([d+1])
    t_step_error=[]
    iter_count = 0
    sum_error = 1
    while sum_error != 0:
        sum_error = 0
        for i in range(n):
            if y[i]*(w.T@x_Wbias[i]) <= 0:
                w += y[i]*x_Wbias[i]
                sum_error += 1
                continue
        t_step_error.append(sum_error)
        iter_count += 1
    return w, t_step_error, iter_count

#Example
from sklearn.datasets import make_blobs
[x,y] = make_blobs(random_state = 42)

y[y==0]=-1
w, t_step_error, iter_count = Perceptron(x,y)
x1_plot = np.linspace(np.min(x[:,0]),np.max(x[:,0]))
x2_plot = np.linspace(np.min(x[:,1]),np.max(x[:,1]))
y_plot = (w[2]*np.ones(len(x1_plot))+w[0]*x1_plot)/(-w[1])

plt.figure()
plt.plot(x1_plot,y_plot)
plt.plot(x[y==-1,0],x[y==-1,1],'ro')
plt.plot(x[y==1,0],x[y==1,1],'bo')
plt.xlabel("x1")
plt.ylabel("x2")

plt.show()
