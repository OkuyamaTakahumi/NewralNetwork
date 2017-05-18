# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

# rd.seed(0)


class Function(object):
    def sigmoid(self, a):
        return 1.0 / (1.0 + np.exp(-a))

    def tanh(self, a):
        return (np.exp(a) - np.exp(-a)) / (np.exp(a) + np.exp(-a))

    def dtanh(self, a):
        return 1.0 - a**2

    def dsigmoid(self, a):
        return (1 - a) * a

    def softmax(self, x):
        return (np.exp(x).T / np.sum(np.exp(x), axis=1)).T


class Newral_Network:
    def __init__(self, unit):
        self.F = Function()
        self.unit = unit
        self.W = []
        self.B = []
        for i in range(len(self.unit) - 1):
            w = np.random.rand(self.unit[i], self.unit[i + 1])
            self.W.append(w)
            b = np.random.rand(self.unit[i + 1])
            self.B.append(b)

    # ユニットへの総入力を返す関数
    def U(self, x, w, b):
        return np.dot(x, w) + b

    # 順伝搬
    def forward(self, _inputs):
        self.Z = []
        self.Z.append(_inputs)
        for i in range(len(self.unit) - 1):
            u = self.U(self.Z[i], self.W[i], self.B[i])
            z = self.F.sigmoid(u)
            self.Z.append(z)
        return z

    # 勾配の計算
    def calc_grad(self, w, b, z, delta):
        w_grad = np.zeros_like(w)
        b_grad = np.zeros_like(b)
        N = float(z.shape[0])
        w_grad = np.dot(z.T, delta) / N
        b_grad = np.mean(delta, axis=0)
        return w_grad, b_grad

    # デルタの計算
    def calc_delta(self, delta_dash, w, z):
        # delta_dash : 1つ先の層のデルタ
        # w : pre_deltaとdeltaを繋ぐネットの重み
        # z : wへ向かう出力
        delta = np.dot(delta_dash, w.T) * self.F.dsigmoid(z)
        return delta

    # 誤差逆伝搬
    def backPropagate(self, _label, eta, M):
        # calculate output_delta and error terms
        error = self.Z[-1] - _label
        W_grad = []
        B_grad = []
        for i in range(len(self.W)):
            w_grad = np.zeros_like(self.W[i])
            W_grad.append(w_grad)
            b_grad = np.zeros_like(self.W[i])
            B_grad.append(b_grad)

        for i in range(len(self.W)):
            if i == 0:
                delta = error * self.F.dsigmoid(self.Z[-(i + 1)])
            else:
                delta = self.calc_delta(delta, self.W[-(i)], self.Z[-(i + 1)])
            W_grad[-(i + 1)], B_grad[-(i + 1)] = self.calc_grad(self.W[-(i + 1)],
                                                                self.B[-(i + 1)], self.Z[-(i + 2)], delta)

        # update weights
        for i in range(len(self.W)):
            self.W[i] = self.W[i] - eta * W_grad[i]
            self.B[i] = self.B[i] - eta * B_grad[i]

        error = np.mean(np.abs((error**2) / 2.0), axis=0)
        return error[0]

    # パラメータの値を取得
    def getWeights(self):
        for i in range(len(self.W)):
            print "w", i + 1, ":"
            print self.W[i]
            print "b", i + 1, ":"
            print self.B[i]

    def train(self, dataset, N, iterations=1000, minibatch=4, eta=0.5, M=0.1):
        print "-----Train-----"
        # 入力データ
        inputs = dataset[:, :self.unit[0]]

        # 訓練データ
        label = dataset[:, self.unit[0]:]

        errors = []
        for val in range(iterations):
            sum_error = 0
            for index in range(0, N, minibatch):
                _inputs = inputs[index: index + minibatch]
                _label = label[index: index + minibatch]

                self.forward(_inputs)

                error = self.backPropagate(_label, eta, M)
                errors.append(error)
                sum_error += error * minibatch
            mean_error = sum_error / N
            print "epoch", val + 1, " : Loss = ", mean_error
        print "\n"

        errors = np.asarray(errors)
        #plt.plot(errors[:, 0])
        #plt.plot(errors[:, 1])
        #plt.plot(errors[:, 2])
        # plt.show()

    def test(self, x):
        print "-----Test-----"
        print "input = \n", x
        print "output = \n", self.forward(x)
        print ("\n")
