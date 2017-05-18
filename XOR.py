# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import NN as NN

if __name__ == '__main__':
    # [入力層のユニット数,隠れ層のユニット数,出力層のユニット数]
    unit = [2, 3, 1]

    minibatch = 4  # ミニバッチのサンプル数
    N = 4  # サンプル数
    iterations = 100000  # 学習回数
    eta = 0.3  # 学習率
    M = 0.1  # モメンタム

    # 入力データを用意
    x1 = np.array([0, 0])
    x2 = np.array([0, 1])
    x3 = np.array([1, 0])
    x4 = np.array([1, 1])
    x = np.vstack((x1, x2, x3, x4))
    x = x.astype(np.float32)
    print "x = \n", x

    # 教師ベクトルを用意
    label1 = 0
    label2 = 1
    label3 = 1
    label4 = 0
    label = np.vstack((label1, label2, label3, label4))
    label = label.astype(np.float32)
    print "label = \n", label

# 一つに
    dataset = np.column_stack((x, label))
    print "dataset = \n", dataset
    print ("\n")
    np.random.shuffle(dataset)  # データ点の順番をシャッフル

    nn = NN.Newral_Network(unit)

    nn.train(dataset, N, iterations, minibatch, eta, M)

    # testデータを用意
    test1 = np.array([0, 0])
    test2 = np.array([0, 1])
    test3 = np.array([1, 0])
    test4 = np.array([1, 1])
    test_data = np.vstack((test1, test2, test3, test4))
    nn.test(test_data)

    nn.getWeights()
