# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

# rd.seed(0)


class Function(object):
    def sigmoid(self, x):
        return 1. / (1 + np.exp(-x))

    def dsigmoid(self, x):
        return x * (1. - x)

    def tanh(self, x):
        return np.tanh(x)

    def dtanh(self, x):
        return 1. - x * x

    def softmax(self, x):
        e = np.exp(x - np.max(x))  # prevent overflow
        if e.ndim == 1:
            return e / np.sum(e, axis=0)
        else:
            return e / np.array([np.sum(e, axis=1)]).T  # ndim = 2

    def ReLU(self, x):
        return x * (x > 0)

    def dReLU(self, x):
        return 1. * (x > 0)


class Newral_Network(object):
    def __init__(self, unit):
        print "Number of layer = ",len(unit)
        print unit
        print "-------------------------"
        self.F = Function()
        self.unit = unit
        self.W = []
        self.B = []
        self.dW = []

        for i in range(len(self.unit) - 1):
            w = np.random.rand(self.unit[i], self.unit[i + 1])
            self.W.append(w)
            dw = np.random.rand(self.unit[i], self.unit[i + 1])
            self.dW.append(dw)
            b = np.random.rand(self.unit[i + 1])
            self.B.append(b)

    # ユニットへの総入力を返す関数
    def U(self, x, w, b):
        return np.dot(x, w) + b

    # 勾配の計算
    def calc_grad(self, w, b, z, delta):
        w_grad = np.zeros_like(w)
        b_grad = np.zeros_like(b)
        N = float(z.shape[0])
        w_grad = np.dot(z.T, delta) / N
        b_grad = np.mean(delta, axis=0)
        return w_grad, b_grad

    # 誤差逆伝搬
    def backPropagate(self, _label, eta, M):
        # calculate output_delta and error terms
        W_grad = []
        B_grad = []
        for i in range(len(self.W)):
            w_grad = np.zeros_like(self.W[i])
            W_grad.append(w_grad)
            b_grad = np.zeros_like(self.W[i])
            B_grad.append(b_grad)
        output = True
        delta = np.zeros_like(self.Z[-1])
        for i in range(len(self.W)):
            delta = self.calc_delta(
                delta, self.W[-(i)], self.Z[-(i + 1)], _label, output)

            W_grad[-(i + 1)], B_grad[-(i + 1)] \
                = self.calc_grad(self.W[-(i + 1)], self.B[-(i + 1)], self.Z[-(i + 2)], delta)

            output = False

        # update weights
        for i in range(len(self.W)):
            self.W[i] = self.W[i] - eta * W_grad[i] + M * self.dW[i]
            self.B[i] = self.B[i] - eta * B_grad[i]
            self.dW[i] = -eta * W_grad[i] + M * self.dW[i]

    def train(self, dataset, N, iterations=1000, minibatch=4, eta=0.5, M=0.1):
        print "-----Train-----"
        # 入力データ
        inputs = dataset[:, :self.unit[0]]

        # 訓練データ
        label = dataset[:, self.unit[0]:]

        errors = []
        for val in range(iterations):
            minibatch_errors = []
            for index in range(0, N, minibatch):
                _inputs = inputs[index: index + minibatch]
                _label = label[index: index + minibatch]
                self.forward(_inputs)
                self.backPropagate(_label, eta, M)

                loss = self.calc_loss(_label)
                minibatch_errors.append(loss)
            En = sum(minibatch_errors) / len(minibatch_errors)
            print "epoch", val + 1, " : Loss = ", En
            errors.append(En)
        print "\n"
        errors = np.asarray(errors)
        plt.plot(errors)

    # パラメータの値を取得
    def getWeights(self):
        for i in range(len(self.W)):
            print "W", i + 1, ":"
            print self.W[i]
            print "\n"
            print "B", i + 1, ":"
            print self.B[i]
            print "\n"

    def save_weight(self, name):
        # 重みの保存
        import datetime
        today_detail = datetime.datetime.today()
        s = today_detail.strftime("%m-%d-%H-%M")
        np.save('Models/%s_W_%s.npy' % (name, s), self.W)
        np.save('Models/%s_B_%s.npy' % (name, s), self.B)
        print "Weight is saved!!"
        for i in range(len(self.W)):
            print "W", i + 1, ".shape = ", self.W[i].shape
            print "B", i + 1, ".shape = ", self.B[i].shape
        print "\n"

    def draw_test(self, x, label, W, B):
        self.W = W
        self.B = B
        x1_max = max(x[:, 0]) + 0.5
        x2_max = max(x[:, 1]) + 0.5
        x1_min = min(x[:, 0]) - 0.5
        x2_min = min(x[:, 1]) - 0.5
        xx, yy = np.meshgrid(np.arange(x1_min, x1_max, 0.01),
                             np.arange(x2_min, x2_max, 0.01))
        Z = self.forward(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        #plt.contour(xx, yy, Z)
        plt.contourf(xx, yy, Z, cmap=plt.cm.jet)
        plt.scatter(x[:, 0], x[:, 1], c=label, cmap=plt.cm.jet)
        plt.show()

    def test(self, x, W, B):
        self.W = W
        self.B = B
        print "-----Test-----"
        for i in range(x.shape[0]):
            print "input = ", x[i]
            print "output = ", int(self.forward(x[i])[0])
            print "\n"
