import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import model_selection
from sklearn import svm
from sklearn import datasets
from keras.datasets import mnist

def cross_validation_error(X, y, model, folds):
    Train_Error_Avg = 0
    Test_Error_Avg = 0
    kf = sklearn.model_selection.KFold(folds)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        Train_Error = 1 - model.score(X_train, y_train)
        Test_Error = 1 - model.score(X_test, y_test)
        Train_Error_Avg += Train_Error
        Test_Error_Avg += Test_Error
    return [Train_Error_Avg/folds,Test_Error_Avg/folds]

def SVM_results(X_train, y_train, X_test, y_test):
    models = {'svm_linear':                 svm.SVC(kernel='linear'),
              'svm_poly_d_2':               svm.SVC(kernel='poly',degree=2),
              'svm_poly_d_4':               svm.SVC(kernel='poly',degree=4),
              'svm_poly_d_6':               svm.SVC(kernel='poly',degree=6),
              'svm_poly_d_8':               svm.SVC(kernel='poly',degree=8),
              'svm_poly_d_10':              svm.SVC(kernel='poly',degree=10),
              'svm_rbf_gamma_0.001':        svm.SVC(kernel='rbf',gamma=0.001),
              'svm_rbf_gamma_0.01':         svm.SVC(kernel='rbf',gamma=0.01),
              'svm_rbf_gamma_0.1':          svm.SVC(kernel='rbf',gamma=0.1),
              'svm_rbf_gamma_1':            svm.SVC(kernel='rbf',gamma=1),
              'svm_rbf_gamma_10':           svm.SVC(kernel='rbf',gamma=10)}

    results = {}
    for model_name, model in models.items():
        temp_results_list = cross_validation_error(X_train, y_train, model, folds=5)
        model.fit(X_train,y_train)
        temp_results_list.append(1 - model.score(X_test, y_test))
        results[model_name] = temp_results_list
    return results

def load_mnist():
    np.random.seed(2)
    (X, y), (_, _) = mnist.load_data()
    indexes = np.random.choice(len(X), 8000, replace=False)
    X = X[indexes]
    y = y[indexes]
    X = X.reshape(len(X), -1)
    return X, y

X,y = load_mnist()
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=98)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = SVM_results(X_train_scaled,y_train,X_test_scaled,y_test)

# results = {'svm_linear': [0.0, 0.1003125, 0.07374999999999998],
#            'svm_poly_d_2': [0.05425781249999999, 0.078125, 0.05312499999999998],
#            'svm_poly_d_4': [0.02855468750000001, 0.059843750000000015, 0.043749999999999956],
#            'svm_poly_d_6': [0.01980468750000002, 0.05265625, 0.03749999999999998],
#            'svm_poly_d_8': [0.016250000000000007, 0.052031249999999994, 0.03312499999999996],
#            'svm_poly_d_10': [0.015078125000000008, 0.053281250000000016, 0.036250000000000004],
#            'svm_rbf_gamma_0.001': [0.057421874999999976, 0.07812499999999997, 0.05874999999999997],
#            'svm_rbf_gamma_0.01': [0.0009375000000000133, 0.044218750000000015, 0.028750000000000053],
#            'svm_rbf_gamma_0.1': [0.0, 0.8125, 0.790625],
#            'svm_rbf_gamma_1': [0.0, 0.8853125000000001, 0.880625],
#            'svm_rbf_gamma_10': [0.0, 0.8853125000000001, 0.880625]}



df = pd.DataFrame(results,index=['Train Error','Validation Error','Test Error']).T
df.index=[x.replace('svm_','').replace('gamma_','$\gamma$=').replace('_','\n',1) for x in df.index]

df.plot(kind="bar",zorder=3, figsize=(10,6))
plt.yticks(np.arange(0, 1, 0.1))
plt.title('SVM Models Results')
plt.xticks(rotation='horizontal')
plt.grid(axis='y',linestyle='--',zorder=0)
plt.show()
