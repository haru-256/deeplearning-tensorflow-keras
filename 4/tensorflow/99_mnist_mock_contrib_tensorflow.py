import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# import matplotlib.pyplot as plt

np.random.seed(0)
tf.set_random_seed(1234)


# Earlystoppingを行うクラス
class EarlyStopping():
    def __init__(self, patience=0, verbose=0):
        """ Early stoppingを行うクラス．
        初期化でモデルの構成を決める．

        Parameters
        -----------------
        patience : integer
            何回連続してval_lossが上昇するのを強要するか

        verbose : integer
            earlystoppingをしたことを教えるかどうか
        """
        self._step = 0
        self._loss = float('inf')
        self.patience = patience
        self.verbose = verbose

    def validate(self, loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print('early stopping')
                return True
        else:
            self._step = 0  # 連続して誤差が増加すると終了するため常に初期化
            self._loss = loss

        return False


class DNN(object):
    def __init__(self, n_in, n_hiddens, n_out):
        """多クラス問題のDeep Neural Network を行うクラス．
        初期化でモデルの構成を決める．

        Parameters
        -----------------
        n_in : integer
            入力素子(unit)の数.データの次元数に相当

        n_hiddens : array-like 要素はinteger
            隠れ素子の数のリスト．各インデックスがlayerに相当

        n_out : integer
            出力素子の数．クラス数に相当

        """
        self.n_in = n_in
        self.n_hiddens = n_hiddens
        self.n_out = n_out
        self.weights_ = []  # 各層の重みWのリスト．要素が各層の重みに対応
        self.biases_ = []  # 各層のバイアスbのリスト．要素が各層のバイアスに対応

        self._history = {
            'val_accuracy': [],
            'val_loss': []
        }

    def _weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def _bias_variable(self, shape):
        initial = tf.zeros(shape)
        return tf.Variable(initial)

    def _inference(self, x, keep_prob):
        """
        推論処理の定義
        """
        # 入力層 - 隠れ層、隠れ層 - 隠れ層
        for i, n_hidden in enumerate(self.n_hiddens):
            if i == 0:
                input = x
                input_dim = self.n_in
            else:
                input = output
                input_dim = self.n_hiddens[i - 1]

            self.weights_.append(self._weight_variable([input_dim, n_hidden]))
            self.biases_.append(self._bias_variable([n_hidden]))

            h = tf.nn.relu(tf.matmul(
                input, self.weights_[-1]) + self.biases_[-1])
            output = tf.nn.dropout(h, keep_prob)

        # 隠れ層 - 出力層
        self.weights_.append(
            self._weight_variable([self.n_hiddens[-1], self.n_out]))
        self.biases_.append(self._bias_variable([self.n_out]))

        y = tf.nn.softmax(tf.matmul(
            output, self.weights_[-1]) + self.biases_[-1])
        return y

    def _loss(self, y, t):

        weight_decay = tf.reduce_sum(
            [tf.nn.l2_loss(w) for w in self.weights_]
        )
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(
            t * tf.log(tf.clip_by_value(y, 1e-10, 1.0)),
            reduction_indices=[1]))

        return cross_entropy + self.lambda_ * weight_decay

    def _training(self, loss):
        optimizer = tf.train.GradientDescentOptimizer(0.005)
        return optimizer.minimize(loss)

    def _accuracy(self, y, t):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def fit(self, X_train, Y_train,
            nb_epoch=100, batch_size=100, p_keep=0.5,
            lambda_=0.001, verbose=1, patience=10):
        """
        学習を行う関数
        """
        # 検証データを用いた評価をするために検証データを作る．
        X_train, X_validation, Y_train, Y_validation = \
            train_test_split(X_train, Y_train, test_size=0.2)

        x = tf.placeholder(tf.float32, shape=[None, self.n_in])
        t = tf.placeholder(tf.float32, shape=[None, self.n_out])
        keep_prob = tf.placeholder(tf.float32)  # drop_outしない確率

        self._x = x
        self._t = t
        self._keep_prob = keep_prob
        self.lambda_ = lambda_

        y = self._inference(x, keep_prob)
        self.loss_ = self._loss(y, t)
        train_step = self._training(self.loss_)
        self.accuracy_ = self._accuracy(y, t)
        early_stopping = EarlyStopping(patience=patience, verbose=1)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        self._sess = sess

        N_train = len(X_train)
        n_batches = N_train // batch_size

        for epoch in range(nb_epoch):
            X_, Y_ = shuffle(X_train, Y_train)

            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size

                self._sess.run(train_step, feed_dict={
                    x: X_[start:end],  # self._x: X_[start:end]でも良い
                    t: Y_[start:end],  # self._t: Y_[start:end]でも良い
                    keep_prob: p_keep  # self._keep_prob: p_keepでも良い
                })
            val_loss = self.loss_.eval(session=self._sess, feed_dict={
                x: X_validation,  # self._x: X_validationでも良い
                t: Y_validation,  # self._t: Y_validationでも良い
                keep_prob: 1.0  # self._keep_prob: 1.0でも良い
            })
            val_accuracy = self.accuracy_.eval(session=self._sess, feed_dict={
                x: X_validation,   # self._x: X_validationでも良い
                t: Y_validation,  # self._t: Y_validationでも良い
                keep_prob: 1.0  # self._keep_prob: 1.0でも良い
            })
            self._history['val_loss'].append(val_loss)
            self._history['val_accuracy'].append(val_accuracy)

            if verbose:
                print('epoch:', epoch,
                      ' validation loss:', val_loss,
                      ' validation accuracy:', val_accuracy)
            # Early Stopping チェック
            if early_stopping.validate(val_loss):
                break

        return self._history

    def evaluate(self, X_test, Y_test):
        return self.accuracy_.eval(session=self._sess, feed_dict={
            self._x: X_test,
            self._t: Y_test,
            self._keep_prob: 1.0
        })


if __name__ == '__main__':
    '''
    データの生成
    '''
    mnist = datasets.fetch_mldata('MNIST original', data_home='.')

    n = len(mnist.data)
    N = 30000  # MNISTの一部を使う
    indices = np.random.permutation(range(n))[:N]  # ランダムにN枚を選択

    X = mnist.data[indices]
    y = mnist.target[indices]
    Y = np.eye(10)[y.astype(int)]  # 1-of-K 表現に変換

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)

    '''
    モデル設定
    '''
    model = DNN(n_in=len(X[0]),
                n_hiddens=[200, 200, 200],
                n_out=len(Y[0]))

    '''
    モデル学習
    '''
    model.fit(X_train, Y_train,
              nb_epoch=200,
              batch_size=200,
              p_keep=0.5)

    '''
    予測精度の評価
    '''
    acc = model.evaluate(X_test, Y_test)
    print('accuracy: ', acc)
