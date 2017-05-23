# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

import NN


class Two_Class_NN(NN.Newral_Network):
    def __init__(self, unit):
        print "-------------------------"
        print "2_Class Newral Network"
        super(Two_Class_NN, self).__init__(unit)

    # 順伝搬
    def forward(self, _inputs):
        self.Z = []
        self.Z.append(_inputs)
        for i in range(len(self.unit) - 1):
            u = self.U(self.Z[i], self.W[i], self.B[i])
            z = self.F.sigmoid(u)
            self.Z.append(z)
        return np.round(z)

    # 誤差の計算
    def calc_loss(self, label):
        error = label * np.log(self.Z[-1]) + \
            (1 - label) * np.log(1 - self.Z[-1])
        return -np.mean(error)

    # デルタの計算
    def calc_delta(self, delta_dash, w, z, label, output):
        # delta_dash : 1つ先の層のデルタ
        # w : pre_deltaとdeltaを繋ぐネットの重み
        # z : wへ向かう出力
        if(output):
            delta = z - label
        else:
            delta = np.dot(delta_dash, w.T) * self.F.dsigmoid(z)
        return delta


if __name__ == '__main__':
    # [入力層のユニット数,隠れ層のユニット数,出力層のユニット数]
    unit = [2, 3, 1]

    minibatch = 20  # ミニバッチのサンプル数
    iterations = 50000  # 学習回数
    eta = 0.1  # 学習率
    M = 0.1  # モメンタム

    x = np.load('./Dataset/2_class/x.npy')
    label = np.load('./Dataset/2_class/label.npy')
    print "x.shape = ", x.shape
    print "label.shape = ", label.shape
    N = x.shape[0]  # サンプル数=400
# 一つに
    dataset = np.column_stack((x, label))
    print "dataset.shape = ", dataset.shape
    print ("\n")
    # np.random.shuffle(dataset)  # データ点の順番をシャッフル

    nn = Two_Class_NN(unit)

    nn.train(dataset, N, iterations, minibatch, eta, M)

    # nn.getWeights()
    nn.save_weight("2_class")
    plt.show()
    nn.draw_test(x, label, nn.W, nn.B)
