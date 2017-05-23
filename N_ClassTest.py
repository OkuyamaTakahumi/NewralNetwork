# coding:utf-8
import numpy as np
#np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
#%matplotlib inline


import N_ClassTrain

if __name__ == '__main__':
    unit = []
    class_num = 3
    W = np.load('./Models/%s_class_W_H3.npy'%(class_num))
    B = np.load('./Models/%s_class_B_H3.npy'%(class_num))
    x = np.load('./Dataset/%s_class/x.npy'%(class_num))
    label = np.load('./Dataset/%s_class/label.npy'%(class_num))


    for i in range(len(W)):
        unit.append(W[i].shape[0])
    unit.append(W[-1].shape[1])

    print "x.shape = ", x.shape
    print "label.shape = ", label.shape
    for i in range(len(W)):
        print "W",i+1,".shape = ",W[i].shape
        print "B",i+1,".shape = ",B[i].shape
        print "\n"

    nn = N_ClassTrain.N_Class_NN(unit)
    nn.draw_test(x, label, W, B)
