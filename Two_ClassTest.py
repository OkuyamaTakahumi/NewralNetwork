# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

import Two_ClassTrain

if __name__ == '__main__':
    unit = []
    W = np.load('./Models/2_class_W_H20.npy')
    B = np.load('./Models/2_class_B_H20.npy')
    x = np.load('./Dataset/2_class/x.npy')
    label = np.load('./Dataset/2_class/label.npy')

    for i in range(len(W)):
        unit.append(W[i].shape[0])
    unit.append(W[-1].shape[1])

    print "x.shape = ", x.shape
    print "label.shape = ", label.shape
    for i in range(len(W)):
        print "W",i+1,".shape = ",W[i].shape
        print "B",i+1,".shape = ",B[i].shape
        print "\n"

    nn = Two_ClassTrain.Two_Class_NN(unit)
    nn.draw_test(x, label, W, B)
