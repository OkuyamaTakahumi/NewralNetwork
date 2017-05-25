# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import NN


class N_Class_NN(NN.Newral_Network):
    # コンストラクタを定義しなければ親のコンストラクタが呼び出される
    def __init__(self, unit):
        print "-------------------------"
        print "N_Class Newral Network"
        super(N_Class_NN,self).__init__(unit)

    # 順伝搬
    def forward(self, _inputs):
        self.Z = []
        self.Z.append(_inputs)
        for i in range(len(self.unit) - 1):
            u = self.U(self.Z[i], self.W[i], self.B[i])
            if(i != len(self.unit) - 2):
                z = np.tanh(u)
            else:
                z = self.F.softmax(u)
            self.Z.append(z)
        if z.ndim == 1:
            return np.argmax(z)
        else:  # ndim = 2
            return np.argmax(z, axis=1)

    # 誤差の計算
    def calc_loss(self, label):
        error = np.sum(label * np.log(self.Z[-1]), axis=1)
        return -np.mean(error)

    # デルタの計算
    def calc_delta(self, delta_dash, w, z, label, output):
        # delta_dash : 1つ先の層のデルタ
        # w : pre_deltaとdeltaを繋ぐネットの重み
        # z : wへ向かう出力
        if(output):
            delta = z - label
        else:
            delta = np.dot(delta_dash, w.T) * self.F.dtanh(z)
        return delta


if __name__ == '__main__':
    # [入力層のユニット数,隠れ層のユニット数,出力層のユニット数]
    class_num = 3
    unit = [2, 3,  class_num]
    save_name="%s_class"%(class_num)

    minibatch = 20  # ミニバッチのサンプル数
    iterations = 3000  # 学習回数
    eta = 0.1  # 学習率
    M = 0.1  # モメンタム

    x = np.load('./Dataset/%s_class/x.npy'%(class_num))
    N = x.shape[0]

    label_dash = np.load('./Dataset/%s_class/label.npy'%(class_num))
    label = np.zeros((N, unit[-1]))
    for i in range(N):
        label[i, label_dash[i]] = 1

    print "x.shape = ", x.shape
    print "label.shape = ", label.shape

# 一つに
    dataset = np.column_stack((x, label))
    print "dataset.shape = ", dataset.shape
    print ("\n")
    np.random.shuffle(dataset)  # データ点の順番をシャッフル

    nn = N_Class_NN(unit)

    nn.train(dataset, N, iterations, minibatch, eta, M)
    nn.getWeights()
    nn.save_weight(save_name)
    plt.show()
    nn.draw_test(x, label_dash, nn.W, nn.B)
