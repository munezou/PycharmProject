from __future__ import print_function
from src import gen_data, train, evaluate


# generate the header file with public keyword
cdef public double calc_accuracy(int n_samples):
    print("[cybridge.pyx] calc_accuracy")
    x_train, x_test, y_train, y_test = gen_data(n_samples)
    clf = train(x_train, y_train)
    confmat = evaluate(clf, x_test, y_test)
    tn, fp, fn, tp = confmat.ravel()
    return float(tn + tp) / confmat.ravel().sum()