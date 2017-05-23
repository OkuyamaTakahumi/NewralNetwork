# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt


import XorTrain



if __name__ == '__main__':
    unit = []
    W = np.load('./Models/XOR_W_H3.npy')
    B = np.load('./Models/XOR_B_H3.npy')
    for i in range(len(W)):
        unit.append(W[i].shape[0])
    unit.append(W[-1].shape[1])

    for i in range(len(W)):
        print "W",i+1,".shape = ",W[i].shape
        print "B",i+1,".shape = ",B[i].shape
        print "\n"


    # testデータを用意
    test_data = []
    for i in range(50):
        test = np.random.randint(0,2,2)
        test_data.append(test)
    test_data = np.asarray(test_data)


    nn = XorTrain.XOR_NN(unit)


    nn.test(test_data, W, B)
