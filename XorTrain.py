# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

import NN

gpu = 0
if(gpu):
    gpu_device = 0
    cuda.get_device(gpu_device).use()
    # model.to_gpu(gpu_device)
    xp = cuda.cupy
else:
    import numpy as np


class XOR_NN(NN.Newral_Network):
    def __init__(self, unit):
        print "-------------------------"
        print "XOR Newrarl Network"
        # (データ型の名前,self)
        super(XOR_NN, self).__init__(unit)

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

    minibatch = 4  # ミニバッチのサンプル数
    N = 4  # サンプル数
    iterations = 10000  # 学習回数
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
    label1 = np.array([0])
    label2 = np.array([1])
    label3 = np.array([1])
    label4 = np.array([0])
    label = np.vstack((label1, label2, label3, label4))
    label = label.astype(np.float32)
    print "label = \n", label

# 一つに
    dataset = np.column_stack((x, label))
    print "dataset = \n", dataset
    print ("\n")
    # np.random.shuffle(dataset)  # データ点の順番をシャッフル

    nn = XOR_NN(unit)

    nn.train(dataset, N, iterations, minibatch, eta, M)

    nn.getWeights()

    nn.save_weight("XOR")
