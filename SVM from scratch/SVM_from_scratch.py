import numpy as np
import matplotlib.pyplot as plt
import sklearn
import matplotlib
from sklearn.model_selection import train_test_split

def svm_with_sgd(X,y,lam=0,epochs=1000,step_size=0.01):
    n,d = X.shape
    w = np.random.uniform(0, 1, size=d)
    b = np.random.rand(1)
    for epoch in range(epochs):
        permutation = np.arange(n)
        np.random.shuffle(permutation)
        X_perm = X[permutation]
        y_perm = y[permutation]
        for xi, yi in zip(X_perm, y_perm):
            if yi*(w.T@xi+b) < 1:
                w_sub_gradient = -(yi * xi) + lam * w
                b_sub_gradient = -yi
                w = w - step_size * w_sub_gradient
                b = b - step_size * b_sub_gradient
    return w, b

def calculate_error(w,b,x,y):
    pred = np.sign(x@w.T+b)
    error = 1 - sum(pred==y)/len(pred)
    return error

np.random.seed(2)
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
X = X[y!=0]
y = y[y!=0]
y [y==2] = -1
X = X[:, 2:4]

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=0)
print(' ')

##### Check my functions #####
w,b = svm_with_sgd(X_train,y_train,lam=0.1,epochs=10000,step_size=0.01)
train_error = calculate_error(w,b,X_train,y_train)

plt.figure()
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired)
x = np.linspace(3, 7)
y = -(1/w[1])*(w[0]*x+b)
plt.plot(x,y, label = 'svm_with_sgd function separator')

from sklearn.svm import SVC
clf = SVC(kernel='linear',C=10)
clf.fit(X_train, y_train)

print('Weights vector of my func: W = ', w, 'b = ' , b)
print('Weights vector of sklearn: W = ', clf.coef_, 'b = ' , clf.intercept_)

print('my func error: ', train_error)
print('sklearn error: ', 1-clf.score(X_train,y_train))

w = clf.coef_
b = clf.intercept_
x = np.linspace(3, 7)
y = -(1/w[0,1])*(w[0,0]*x+b[0])
plt.plot(x,y,'--', color = 'navy', label = 'sklearn.svm separator')
plt.grid()
plt.legend()
plt.xlabel("x1")
plt.ylabel("x2")
plt.title('Validate svm_with_sgd function comparing to sklearn builtin function')
#plt.show()
print(' ')
#########################################################

colors = ['navy', 'blue', 'dodgerblue', 'skyblue', 'c']
lambdas = [0,0.05,0.1,0.2,0.5]

min_x1, max_x1 = X_train[:, 0].min(), X_train[:, 0].max()
min_x2, max_x2 = X_train[:, 1].min(), X_train[:, 1].max()
x = np.linspace(min_x1 - 0.2, max_x1 + 0.2)
w_all = []
b_all = []
Train_Error_dict = {}
Test_Error_dict = {}

for lam in lambdas:
    w, b = svm_with_sgd(X_train, y_train, lam=lam)
    w_all.append(w)
    b_all.append(b)
    train_error = calculate_error(w,b,X_train,y_train)
    Train_Error_dict[lam] = train_error
    test_error = calculate_error(w, b, X_test, y_test)
    Test_Error_dict[lam] = test_error

plt.figure()
i=0
for i in range(len(w_all)):
    w = w_all[i]
    b = b_all[i]
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired)
    y = -(1 / w[1]) * (w[0] * x + b)
    lam = lambdas[i]
    plt.plot(x, y, label=r'$\lambda$ = ' + str(lam), color=colors[i])
    i += 1
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title('Train Data')
    plt.grid()
    plt.legend()

plt.figure()
i=0
for i in range(len(w_all)):
    w = w_all[i]
    b = b_all[i]
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Paired)
    y = -(1 / w[1]) * (w[0] * x + b)
    lam = lambdas[i]
    plt.plot(x, y, label= r'$\lambda$ = ' + str(lam), color=colors[i])
    i += 1
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title('Test Data')
    plt.grid()
    plt.legend()

#ploting errors bar plot
print(Train_Error_dict)
print(Test_Error_dict)
plt.figure()
xticks = [r'$\lambda$ = ' + str(lam) for lam in lambdas]
plt.bar(xticks, Train_Error_dict.values(), align='edge', width=-0.2,  label='train')
plt.bar(xticks, Test_Error_dict.values(), align='edge', width=0.2, label='test')

for i, v in enumerate(Train_Error_dict.values()):
    plt.text(i-0.3, 1.01*v.max(), str('%.3f' %v), size = 8)

for i, v in enumerate(Test_Error_dict.values()):
    plt.text(i, 1.01*v.max(), str('%.3f' %v), size = 8)

plt.legend(loc = 'lower right')
plt.ylabel("Model Error")
plt.title('Models Errors According to Regularization parameter - $\lambda$')

#ploting mergins width
mergins = []
plt.figure()
for w in w_all:
    mergins_width = 2/np.linalg.norm(w)
    mergins.append(mergins_width)
plt.bar(xticks, mergins, align='center', width=0.3)
for i, v in enumerate(mergins):
    plt.text(i-0.1, 1.01*v.max(), str('%.1f' %v), size = 10)

plt.ylabel("Mergins Width")
plt.title('Mergins Width According to Regularization parameter - $\lambda$')

plt.show()
